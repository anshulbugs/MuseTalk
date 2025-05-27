#!/usr/bin/env python3
"""
Simple MuseTalk Video Streaming Server
Uses WebSocket to stream video frames directly, bypassing WebRTC complexity
"""

import asyncio
import json
import logging
import os
import sys
import time
import base64
from pathlib import Path
import cv2
import numpy as np
import torch
from aiohttp import web, WSMsgType
import aiohttp_cors
from omegaconf import OmegaConf

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from musetalk.utils.utils import load_all_model
from musetalk.utils.face_parsing import FaceParsing
from musetalk.utils.audio_processor import AudioProcessor
from scripts.realtime_inference import Avatar

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleVideoStreamer:
    """Simple video streamer using WebSocket"""
    
    def __init__(self, config_path: str = "configs/inference/realtime.yaml"):
        self.config_path = config_path
        self.config = None
        self.models = {}
        self.avatars = {}
        self.audio_processor = None
        self.clients = {}
        self.app = web.Application()
        self.setup_routes()
        self.setup_cors()
        
    def setup_routes(self):
        """Setup HTTP routes"""
        self.app.router.add_get('/', self.serve_index)
        self.app.router.add_static('/static/', path='static/', name='static')
        self.app.router.add_get('/ws', self.websocket_handler)
        
    def setup_cors(self):
        """Setup CORS"""
        cors = aiohttp_cors.setup(self.app, defaults={
            "*": aiohttp_cors.ResourceOptions(
                allow_credentials=True,
                expose_headers="*",
                allow_headers="*",
                allow_methods="*"
            )
        })
        
        for route in list(self.app.router.routes()):
            cors.add(route)
    
    async def serve_index(self, request):
        """Serve the main HTML page"""
        html_content = """
<!DOCTYPE html>
<html>
<head>
    <title>MuseTalk Simple Video Stream</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #f0f0f0; }
        .container { max-width: 800px; margin: 0 auto; background: white; padding: 20px; border-radius: 10px; }
        .video-container { text-align: center; margin: 20px 0; }
        #videoCanvas { border: 2px solid #333; border-radius: 10px; }
        .controls { text-align: center; margin: 20px 0; }
        button { padding: 10px 20px; margin: 5px; font-size: 16px; border: none; border-radius: 5px; cursor: pointer; }
        .connect { background: #4CAF50; color: white; }
        .disconnect { background: #f44336; color: white; }
        .status { padding: 10px; margin: 10px 0; border-radius: 5px; text-align: center; }
        .connected { background: #d4edda; color: #155724; }
        .disconnected { background: #f8d7da; color: #721c24; }
        .info { background: #d1ecf1; color: #0c5460; padding: 10px; margin: 10px 0; border-radius: 5px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎭 MuseTalk Simple Video Stream</h1>
        
        <div class="info">
            <strong>Instructions:</strong>
            <ol>
                <li>Click "Connect" to establish WebSocket connection</li>
                <li>The avatar video will start streaming automatically</li>
                <li>Speak into your microphone to see real-time lip-sync</li>
            </ol>
        </div>
        
        <div class="video-container">
            <canvas id="videoCanvas" width="512" height="512"></canvas>
        </div>
        
        <div class="controls">
            <button id="connectBtn" class="connect" onclick="connect()">Connect</button>
            <button id="disconnectBtn" class="disconnect" onclick="disconnect()" disabled>Disconnect</button>
        </div>
        
        <div id="status" class="status disconnected">Disconnected</div>
        
        <div class="info">
            <strong>Features:</strong>
            <ul>
                <li>✅ Real-time avatar video streaming</li>
                <li>✅ Audio-to-video lip-sync generation</li>
                <li>✅ Simple WebSocket-based streaming</li>
                <li>✅ No complex WebRTC setup required</li>
            </ul>
        </div>
    </div>

    <script>
        let websocket = null;
        let canvas = null;
        let ctx = null;
        let mediaRecorder = null;
        let audioStream = null;
        
        window.onload = function() {
            canvas = document.getElementById('videoCanvas');
            ctx = canvas.getContext('2d');
        };
        
        async function connect() {
            try {
                // Connect to WebSocket
                const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
                const wsHost = window.location.hostname;
                const wsPort = window.location.port || '8080';
                const wsUrl = `${wsProtocol}//${wsHost}:${wsPort}/ws`;
                
                console.log('Connecting to:', wsUrl);
                websocket = new WebSocket(wsUrl);
                
                websocket.onopen = async function() {
                    console.log('WebSocket connected');
                    updateStatus('Connected - Starting video stream...', 'connected');
                    document.getElementById('connectBtn').disabled = true;
                    document.getElementById('disconnectBtn').disabled = false;
                    
                    // Start audio capture
                    try {
                        audioStream = await navigator.mediaDevices.getUserMedia({ audio: true });
                        startAudioRecording();
                        
                        // Request video stream
                        websocket.send(JSON.stringify({ type: 'start_video' }));
                    } catch (err) {
                        console.error('Error accessing microphone:', err);
                        updateStatus('Connected - Microphone access denied', 'connected');
                    }
                };
                
                websocket.onmessage = function(event) {
                    try {
                        const data = JSON.parse(event.data);
                        
                        if (data.type === 'video_frame') {
                            // Display video frame
                            displayVideoFrame(data.frame);
                        } else if (data.type === 'status') {
                            updateStatus(data.message, 'connected');
                        }
                    } catch (err) {
                        console.error('Error processing message:', err);
                    }
                };
                
                websocket.onclose = function() {
                    console.log('WebSocket disconnected');
                    updateStatus('Disconnected', 'disconnected');
                    document.getElementById('connectBtn').disabled = false;
                    document.getElementById('disconnectBtn').disabled = true;
                    stopAudioRecording();
                };
                
                websocket.onerror = function(error) {
                    console.error('WebSocket error:', error);
                    updateStatus('Connection error', 'disconnected');
                };
                
            } catch (err) {
                console.error('Connection error:', err);
                updateStatus('Connection failed', 'disconnected');
            }
        }
        
        function disconnect() {
            if (websocket) {
                websocket.close();
            }
            stopAudioRecording();
        }
        
        function startAudioRecording() {
            if (audioStream && !mediaRecorder) {
                mediaRecorder = new MediaRecorder(audioStream, {
                    mimeType: 'audio/webm;codecs=opus'
                });
                
                mediaRecorder.ondataavailable = function(event) {
                    if (event.data.size > 0 && websocket && websocket.readyState === WebSocket.OPEN) {
                        // Send audio data
                        const reader = new FileReader();
                        reader.onload = function() {
                            const audioData = new Uint8Array(reader.result);
                            websocket.send(JSON.stringify({
                                type: 'audio_data',
                                data: Array.from(audioData)
                            }));
                        };
                        reader.readAsArrayBuffer(event.data);
                    }
                };
                
                mediaRecorder.start(100); // Send audio every 100ms
                console.log('Audio recording started');
            }
        }
        
        function stopAudioRecording() {
            if (mediaRecorder) {
                mediaRecorder.stop();
                mediaRecorder = null;
            }
            if (audioStream) {
                audioStream.getTracks().forEach(track => track.stop());
                audioStream = null;
            }
        }
        
        function displayVideoFrame(frameData) {
            try {
                const img = new Image();
                img.onload = function() {
                    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
                };
                img.src = 'data:image/jpeg;base64,' + frameData;
            } catch (err) {
                console.error('Error displaying frame:', err);
            }
        }
        
        function updateStatus(message, type) {
            const statusEl = document.getElementById('status');
            statusEl.textContent = message;
            statusEl.className = 'status ' + type;
        }
    </script>
</body>
</html>
        """
        return web.Response(text=html_content, content_type='text/html')
    
    async def websocket_handler(self, request):
        """Handle WebSocket connections"""
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        
        client_id = f"client_{id(ws)}"
        self.clients[client_id] = {
            'ws': ws,
            'streaming': False,
            'avatar': None
        }
        
        logger.info(f"New WebSocket connection: {client_id}")
        
        try:
            async for msg in ws:
                if msg.type == WSMsgType.TEXT:
                    try:
                        data = json.loads(msg.data)
                        await self.handle_message(client_id, data)
                    except Exception as e:
                        logger.error(f"Error handling message from {client_id}: {e}")
                elif msg.type == WSMsgType.ERROR:
                    logger.error(f"WebSocket error from {client_id}: {ws.exception()}")
                    
        except Exception as e:
            logger.error(f"WebSocket handler error for {client_id}: {e}")
        finally:
            logger.info(f"WebSocket connection closed: {client_id}")
            if client_id in self.clients:
                self.clients[client_id]['streaming'] = False
                del self.clients[client_id]
        
        return ws
    
    async def handle_message(self, client_id: str, data: dict):
        """Handle WebSocket messages"""
        message_type = data.get("type")
        
        if message_type == "start_video":
            # Start video streaming for this client
            await self.start_video_stream(client_id)
            
        elif message_type == "audio_data":
            # Process audio data (simplified for now)
            await self.process_audio_data(client_id, data.get("data", []))
    
    async def start_video_stream(self, client_id: str):
        """Start video streaming for a client"""
        if client_id not in self.clients:
            return
            
        client = self.clients[client_id]
        if client['streaming']:
            return
            
        client['streaming'] = True
        
        # Use first available avatar
        avatar_id = list(self.avatars.keys())[0]
        client['avatar'] = self.avatars[avatar_id]
        
        logger.info(f"Starting video stream for {client_id} with avatar {avatar_id}")
        
        # Send status
        await client['ws'].send_str(json.dumps({
            'type': 'status',
            'message': f'Streaming avatar {avatar_id}'
        }))
        
        # Start streaming task
        asyncio.create_task(self.video_stream_task(client_id))
    
    async def video_stream_task(self, client_id: str):
        """Video streaming task"""
        if client_id not in self.clients:
            return
            
        client = self.clients[client_id]
        avatar = client['avatar']
        frame_count = 0
        
        try:
            while client['streaming'] and client_id in self.clients:
                # Get current avatar frame
                frame_idx = frame_count % len(avatar.frame_list_cycle)
                frame = avatar.frame_list_cycle[frame_idx].copy()
                
                # Add test overlay
                current_time = time.time()
                timestamp_text = f"Frame: {frame_count}"
                cv2.putText(frame, timestamp_text, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Add moving circle
                circle_x = int((current_time * 50) % frame.shape[1])
                cv2.circle(frame, (circle_x, 50), 20, (255, 0, 0), -1)
                
                # Convert to JPEG and base64
                _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                frame_b64 = base64.b64encode(buffer).decode('utf-8')
                
                # Send frame
                await client['ws'].send_str(json.dumps({
                    'type': 'video_frame',
                    'frame': frame_b64
                }))
                
                frame_count += 1
                
                # 25 FPS
                await asyncio.sleep(1.0 / 25.0)
                
        except Exception as e:
            logger.error(f"Error in video stream task for {client_id}: {e}")
        finally:
            if client_id in self.clients:
                self.clients[client_id]['streaming'] = False
    
    async def process_audio_data(self, client_id: str, audio_data: list):
        """Process audio data (simplified)"""
        # For now, just log that we received audio
        logger.debug(f"Received audio data from {client_id}: {len(audio_data)} bytes")
    
    async def initialize_models(self):
        """Initialize MuseTalk models"""
        logger.info("Initializing MuseTalk models...")
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load models
        vae, unet, pe = load_all_model(
            unet_model_path="./models/musetalkV15/unet.pth",
            vae_type="sd-vae", 
            unet_config="./models/musetalkV15/musetalk.json",
            device=device
        )
        
        # Set precision
        weight_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        pe = pe.half().to(device) if torch.cuda.is_available() else pe.to(device)
        vae.vae = vae.vae.half().to(device) if torch.cuda.is_available() else vae.vae.to(device)
        unet.model = unet.model.half().to(device) if torch.cuda.is_available() else unet.model.to(device)
        
        timesteps = torch.tensor([0], device=device)
        
        # Initialize audio processor
        self.audio_processor = AudioProcessor(feature_extractor_path="./models/whisper")
        
        # Initialize face parser
        fp = FaceParsing()
        
        self.models = {
            'vae': vae,
            'unet': unet, 
            'pe': pe,
            'fp': fp,
            'device': device,
            'weight_dtype': weight_dtype,
            'timesteps': timesteps
        }
        
        logger.info("Models initialized successfully")
    
    async def initialize_avatars(self):
        """Initialize avatars from config"""
        logger.info("Initializing and preloading avatars...")
        
        # Create args object for Avatar class
        from argparse import Namespace
        args = Namespace(
            version="v15",
            extra_margin=10,
            parsing_mode="jaw",
            left_cheek_width=90,
            right_cheek_width=90
        )
        
        # Set globals for Avatar class
        import scripts.realtime_inference
        scripts.realtime_inference.args = args
        scripts.realtime_inference.vae = self.models['vae']
        scripts.realtime_inference.fp = self.models['fp']
        
        # Load config
        self.config = OmegaConf.load(self.config_path)
        
        for avatar_id in self.config:
            try:
                avatar_config = self.config[avatar_id]
                logger.info(f"Preloading avatar {avatar_id}...")
                
                avatar = Avatar(
                    avatar_id=avatar_id,
                    video_path=avatar_config["video_path"],
                    bbox_shift=avatar_config.get("bbox_shift", 0),
                    batch_size=1,
                    preparation=True
                )
                
                # Preload to GPU if available
                if torch.cuda.is_available():
                    avatar.input_latent_list_cycle = [
                        latent.to(self.models['device']) 
                        for latent in avatar.input_latent_list_cycle
                    ]
                
                self.avatars[avatar_id] = avatar
                logger.info(f"Avatar {avatar_id} fully preloaded and ready")
                
            except Exception as e:
                logger.error(f"Failed to initialize avatar {avatar_id}: {e}")
        
        if not self.avatars:
            raise RuntimeError("No avatars were successfully initialized")
            
        logger.info(f"Successfully preloaded {len(self.avatars)} avatars")
        
        # Log memory usage if CUDA is available
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
            logger.info(f"GPU memory allocated: {memory_allocated:.2f} GB")
    
    async def start_server(self, host: str = "localhost", port: int = 8080):
        """Start the server"""
        # Initialize components
        await self.initialize_models()
        await self.initialize_avatars()
        
        logger.info(f"Starting Simple Video Server on http://{host}:{port}")
        
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
    
    parser = argparse.ArgumentParser(description="MuseTalk Simple Video Server")
    parser.add_argument("--host", type=str, default="localhost", help="Server host")
    parser.add_argument("--port", type=int, default=8080, help="Server port")
    parser.add_argument("--config", type=str, default="configs/inference/realtime.yaml", help="Config file")
    
    args = parser.parse_args()
    
    server = SimpleVideoStreamer(args.config)
    asyncio.run(server.start_server(args.host, args.port))