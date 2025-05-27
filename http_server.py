import asyncio
import os
import sys
from pathlib import Path
from aiohttp import web, WSMsgType
import aiohttp_cors
import json
import logging

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from webrtc_server import MuseTalkWebRTCServer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MuseTalkHTTPServer:
    """HTTP server that serves the web interface and handles WebRTC signaling"""
    
    def __init__(self, config_path: str = "configs/inference/realtime.yaml"):
        self.webrtc_server = MuseTalkWebRTCServer(config_path)
        self.app = web.Application()
        self.setup_routes()
        self.setup_cors()
        
    def setup_routes(self):
        """Setup HTTP routes"""
        # Serve static files
        self.app.router.add_get('/', self.serve_index)
        self.app.router.add_static('/static/', path='static/', name='static')
        
        # WebSocket endpoint for signaling
        self.app.router.add_get('/ws', self.websocket_handler)
        
    def setup_cors(self):
        """Setup CORS for cross-origin requests"""
        cors = aiohttp_cors.setup(self.app, defaults={
            "*": aiohttp_cors.ResourceOptions(
                allow_credentials=True,
                expose_headers="*",
                allow_headers="*",
                allow_methods="*"
            )
        })
        
        # Add CORS to all routes
        for route in list(self.app.router.routes()):
            cors.add(route)
    
    async def serve_index(self, request):
        """Serve the main HTML page"""
        try:
            static_dir = Path(__file__).parent / 'static'
            index_path = static_dir / 'index.html'
            
            if not index_path.exists():
                return web.Response(text="Index file not found", status=404)
                
            with open(index_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            return web.Response(text=content, content_type='text/html')
            
        except Exception as e:
            logger.error(f"Error serving index: {e}")
            return web.Response(text="Internal server error", status=500)
    
    async def websocket_handler(self, request):
        """Handle WebSocket connections for WebRTC signaling"""
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        
        client_id = f"client_{id(ws)}"
        logger.info(f"New WebSocket connection: {client_id}")
        
        try:
            async for msg in ws:
                if msg.type == WSMsgType.TEXT:
                    try:
                        data = json.loads(msg.data)
                        await self.handle_signaling_message(ws, client_id, data)
                    except json.JSONDecodeError:
                        logger.error(f"Invalid JSON from {client_id}")
                        await ws.send_str(json.dumps({
                            "type": "error",
                            "message": "Invalid JSON format"
                        }))
                elif msg.type == WSMsgType.ERROR:
                    logger.error(f"WebSocket error from {client_id}: {ws.exception()}")
                    
        except Exception as e:
            logger.error(f"WebSocket handler error for {client_id}: {e}")
        finally:
            logger.info(f"WebSocket connection closed: {client_id}")
            # Cleanup peer connection if exists
            if client_id in self.webrtc_server.peer_connections:
                try:
                    await self.webrtc_server.peer_connections[client_id].close()
                    del self.webrtc_server.peer_connections[client_id]
                except:
                    pass
            if client_id in self.webrtc_server.video_tracks:
                del self.webrtc_server.video_tracks[client_id]
        
        return ws
    
    async def handle_signaling_message(self, ws, client_id: str, data: dict):
        """Handle WebRTC signaling messages"""
        try:
            message_type = data.get("type")
            
            if message_type == "get_avatars":
                # Send available avatars
                avatar_list = [
                    {
                        "id": avatar_id,
                        "name": avatar_id,
                        "video_path": self.webrtc_server.config[avatar_id]["video_path"]
                    }
                    for avatar_id in self.webrtc_server.avatars.keys()
                ]
                await ws.send_str(json.dumps({
                    "type": "avatars",
                    "avatars": avatar_list
                }))
                
            elif message_type == "select_avatar":
                avatar_id = data.get("avatar_id")
                if avatar_id in self.webrtc_server.avatars:
                    # Create video track for this client
                    from webrtc_server import MuseTalkVideoTrack
                    video_track = MuseTalkVideoTrack(
                        self.webrtc_server.avatars[avatar_id],
                        self.webrtc_server.audio_processor,
                        self.webrtc_server.models
                    )
                    self.webrtc_server.video_tracks[client_id] = video_track
                    
                    await ws.send_str(json.dumps({
                        "type": "avatar_selected",
                        "avatar_id": avatar_id
                    }))
                else:
                    await ws.send_str(json.dumps({
                        "type": "error",
                        "message": f"Avatar {avatar_id} not found"
                    }))
                    
            elif message_type == "offer":
                # Handle WebRTC offer
                await self.handle_offer(ws, client_id, data)
                
            elif message_type == "answer":
                # Handle WebRTC answer
                await self.handle_answer(client_id, data)
                
            elif message_type == "ice_candidate":
                # Handle ICE candidate
                await self.handle_ice_candidate(client_id, data)
                
        except Exception as e:
            logger.error(f"Error handling signaling message from {client_id}: {e}")
            await ws.send_str(json.dumps({
                "type": "error",
                "message": str(e)
            }))
    
    async def handle_offer(self, ws, client_id: str, data: dict):
        """Handle WebRTC offer"""
        try:
            from aiortc import RTCPeerConnection, RTCSessionDescription
            from av import AudioFrame
            
            pc = RTCPeerConnection(configuration=self.webrtc_server.rtc_config)
            self.webrtc_server.peer_connections[client_id] = pc
            
            # Add video track if available
            if client_id in self.webrtc_server.video_tracks:
                pc.addTrack(self.webrtc_server.video_tracks[client_id])
            
            # Handle incoming audio track
            @pc.on("track")
            async def on_track(track):
                logger.info(f"Received {track.kind} track from {client_id}")
                if track.kind == "audio" and client_id in self.webrtc_server.video_tracks:
                    # Process audio frames
                    video_track = self.webrtc_server.video_tracks[client_id]
                    
                    async def process_audio():
                        try:
                            while True:
                                try:
                                    frame = await track.recv()
                                    if isinstance(frame, AudioFrame):
                                        # Convert audio frame to numpy array
                                        audio_data = frame.to_ndarray()
                                        if len(audio_data.shape) > 1:
                                            audio_data = audio_data.mean(axis=1)  # Convert to mono
                                        
                                        # Process audio chunk
                                        await video_track.process_audio_chunk(
                                            audio_data, frame.sample_rate
                                        )
                                except Exception as e:
                                    logger.error(f"Error receiving audio frame: {e}")
                                    break
                        except Exception as e:
                            logger.error(f"Error processing audio: {e}")
                    
                    # Start audio processing task
                    asyncio.create_task(process_audio())
            
            # Set remote description
            await pc.setRemoteDescription(RTCSessionDescription(
                sdp=data["sdp"], type=data["type"]
            ))
            
            # Create answer
            answer = await pc.createAnswer()
            await pc.setLocalDescription(answer)
            
            # Send answer
            await ws.send_str(json.dumps({
                "type": "answer",
                "sdp": pc.localDescription.sdp,
            }))
            
        except Exception as e:
            logger.error(f"Error handling offer: {e}")
            await ws.send_str(json.dumps({
                "type": "error",
                "message": str(e)
            }))
    
    async def handle_answer(self, client_id: str, data: dict):
        """Handle WebRTC answer"""
        if client_id in self.webrtc_server.peer_connections:
            from aiortc import RTCSessionDescription
            pc = self.webrtc_server.peer_connections[client_id]
            await pc.setRemoteDescription(RTCSessionDescription(
                sdp=data["sdp"], type=data["type"]
            ))
    
    async def handle_ice_candidate(self, client_id: str, data: dict):
        """Handle ICE candidate"""
        if client_id in self.webrtc_server.peer_connections:
            pc = self.webrtc_server.peer_connections[client_id]
            candidate = data.get("candidate")
            if candidate:
                try:
                    # For now, skip ICE candidate processing as it's not critical for basic functionality
                    # The WebRTC connection can still work without explicit ICE candidate handling
                    logger.debug(f"Skipping ICE candidate processing: {candidate}")
                except Exception as e:
                    logger.error(f"Error processing ICE candidate: {e}")
    
    async def start_server(self, host: str = "localhost", port: int = 8080):
        """Start the HTTP server"""
        # Initialize WebRTC server components
        await self.webrtc_server.initialize_models()
        await self.webrtc_server.initialize_avatars()
        
        logger.info(f"Starting HTTP server on http://{host}:{port}")
        
        runner = web.AppRunner(self.app)
        await runner.setup()
        
        site = web.TCPSite(runner, host, port)
        await site.start()
        
        logger.info(f"Server started successfully at http://{host}:{port}")
        logger.info("Open your browser and navigate to the URL above to start streaming")
        
        # Keep the server running
        try:
            await asyncio.Future()  # Run forever
        except KeyboardInterrupt:
            logger.info("Shutting down server...")
        finally:
            await runner.cleanup()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="MuseTalk WebRTC HTTP Server")
    parser.add_argument("--host", type=str, default="localhost", help="Server host")
    parser.add_argument("--port", type=int, default=8080, help="Server port")
    parser.add_argument("--config", type=str, default="configs/inference/realtime.yaml", help="Config file")
    
    args = parser.parse_args()
    
    server = MuseTalkHTTPServer(args.config)
    asyncio.run(server.start_server(args.host, args.port))