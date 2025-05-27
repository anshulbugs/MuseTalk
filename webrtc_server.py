import asyncio
import json
import logging
import os
import sys
import time
from typing import Optional, Dict, Any
import uuid

import cv2
import numpy as np
import torch
from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack, RTCConfiguration, RTCIceServer
from aiortc.contrib.media import MediaPlayer, MediaRelay
from av import VideoFrame, AudioFrame
import websockets
from websockets.server import serve
import argparse
from omegaconf import OmegaConf

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from musetalk.utils.utils import load_all_model
from musetalk.utils.audio_processor import AudioProcessor
from musetalk.utils.face_parsing import FaceParsing
from musetalk.utils.preprocessing import get_landmark_and_bbox, read_imgs, coord_placeholder
from musetalk.utils.blending import get_image_prepare_material, get_image_blending
from transformers import WhisperModel
import pickle
import glob
import shutil
from scripts.realtime_inference import Avatar

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MuseTalkVideoTrack(VideoStreamTrack):
    """Custom video track that generates MuseTalk frames"""
    
    def __init__(self, avatar: Avatar, audio_processor: AudioProcessor, models: Dict[str, Any]):
        super().__init__()
        self.avatar = avatar
        self.audio_processor = audio_processor
        self.models = models
        self.audio_buffer = []
        self.frame_queue = asyncio.Queue(maxsize=10)
        self.current_frame_idx = 0
        self.fps = 25
        self.frame_time = 1.0 / self.fps
        self.last_frame_time = time.time()
        
        # Default frame (first avatar frame)
        self.default_frame = self.avatar.frame_list_cycle[0]
        
    async def recv(self):
        """Generate video frames for WebRTC"""
        current_time = time.time()
        
        # Maintain frame rate
        time_since_last = current_time - self.last_frame_time
        if time_since_last < self.frame_time:
            await asyncio.sleep(self.frame_time - time_since_last)
        
        try:
            # Try to get processed frame from queue
            frame_data = self.frame_queue.get_nowait()
            frame = frame_data
        except asyncio.QueueEmpty:
            # Use default frame if no processed frame available
            frame = self.default_frame
            
        # Convert to WebRTC VideoFrame
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        av_frame = VideoFrame.from_ndarray(frame_rgb, format="rgb24")
        av_frame.pts = int(current_time * 90000)  # 90kHz timestamp
        av_frame.time_base = "1/90000"
        
        self.last_frame_time = time.time()
        return av_frame
    
    async def process_audio_chunk(self, audio_data: np.ndarray, sample_rate: int = 16000):
        """Process audio chunk and generate corresponding video frame"""
        try:
            # Convert audio to features
            whisper_features = self.audio_processor.process_audio_chunk(
                audio_data, sample_rate, self.models['device'], self.models['weight_dtype']
            )
            
            if whisper_features is not None:
                # Generate video frame
                frame = await self.generate_frame(whisper_features)
                if frame is not None:
                    # Add to frame queue (non-blocking)
                    try:
                        self.frame_queue.put_nowait(frame)
                    except asyncio.QueueFull:
                        # Remove oldest frame and add new one
                        try:
                            self.frame_queue.get_nowait()
                            self.frame_queue.put_nowait(frame)
                        except asyncio.QueueEmpty:
                            pass
                            
        except Exception as e:
            logger.error(f"Error processing audio chunk: {e}")
    
    async def generate_frame(self, audio_features):
        """Generate a single frame from audio features"""
        try:
            # Get current avatar frame and coordinates
            frame_idx = self.current_frame_idx % len(self.avatar.frame_list_cycle)
            ori_frame = self.avatar.frame_list_cycle[frame_idx].copy()
            bbox = self.avatar.coord_list_cycle[frame_idx]
            latent = self.avatar.input_latent_list_cycle[frame_idx % len(self.avatar.input_latent_list_cycle)]
            
            # Skip if no valid bbox
            if bbox == coord_placeholder:
                return ori_frame
                
            # Prepare latent batch
            latent_batch = latent.unsqueeze(0).to(
                device=self.models['device'], 
                dtype=self.models['weight_dtype']
            )
            
            # Audio feature batch
            audio_feature_batch = self.models['pe'](audio_features.unsqueeze(0))
            
            # Generate prediction
            with torch.no_grad():
                pred_latents = self.models['unet'].model(
                    latent_batch,
                    self.models['timesteps'],
                    encoder_hidden_states=audio_feature_batch
                ).sample
                
                # Decode latents
                recon = self.models['vae'].decode_latents(pred_latents)
                res_frame = recon[0]
            
            # Blend with original frame
            x1, y1, x2, y2 = bbox
            res_frame_resized = cv2.resize(
                res_frame.astype(np.uint8), 
                (x2 - x1, y2 - y1)
            )
            
            # Get mask and blend
            mask = self.avatar.mask_list_cycle[frame_idx]
            mask_crop_box = self.avatar.mask_coords_list_cycle[frame_idx]
            
            combined_frame = get_image_blending(
                ori_frame, res_frame_resized, bbox, mask, mask_crop_box
            )
            
            self.current_frame_idx += 1
            return combined_frame
            
        except Exception as e:
            logger.error(f"Error generating frame: {e}")
            return self.default_frame

class MuseTalkWebRTCServer:
    """WebRTC server for MuseTalk streaming"""
    
    def __init__(self, config_path: str = "configs/inference/realtime.yaml"):
        self.config = OmegaConf.load(config_path)
        self.avatars = {}
        self.models = {}
        self.audio_processor = None
        self.peer_connections = {}
        self.video_tracks = {}
        
        # WebRTC configuration
        self.rtc_config = RTCConfiguration(
            iceServers=[
                RTCIceServer(urls=["stun:stun.l.google.com:19302"]),
            ]
        )
        
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
        self.audio_processor = RealTimeAudioProcessor(feature_extractor_path="./models/whisper")
        whisper = WhisperModel.from_pretrained("./models/whisper")
        whisper = whisper.to(device=device, dtype=weight_dtype).eval()
        whisper.requires_grad_(False)
        
        # Initialize face parser
        fp = FaceParsing()
        
        self.models = {
            'vae': vae,
            'unet': unet, 
            'pe': pe,
            'whisper': whisper,
            'fp': fp,
            'device': device,
            'weight_dtype': weight_dtype,
            'timesteps': timesteps
        }
        
        logger.info("Models initialized successfully")
    
    async def initialize_avatars(self):
        """Initialize and preload all avatars from config"""
        logger.info("Initializing and preloading avatars...")
        
        for avatar_id in self.config:
            try:
                avatar_config = self.config[avatar_id]
                logger.info(f"Preloading avatar {avatar_id}...")
                
                # Force preparation to True to ensure all materials are preloaded
                avatar = Avatar(
                    avatar_id=avatar_id,
                    video_path=avatar_config["video_path"],
                    bbox_shift=avatar_config.get("bbox_shift", 0),
                    batch_size=1,  # Use batch size 1 for real-time
                    preparation=True  # Always preload materials
                )
                
                # Verify all required materials are loaded
                required_attrs = [
                    'frame_list_cycle', 'coord_list_cycle',
                    'input_latent_list_cycle', 'mask_list_cycle',
                    'mask_coords_list_cycle'
                ]
                
                for attr in required_attrs:
                    if not hasattr(avatar, attr) or getattr(avatar, attr) is None:
                        raise ValueError(f"Avatar {avatar_id} missing required attribute: {attr}")
                
                # Preload materials into GPU memory if available
                if torch.cuda.is_available():
                    logger.info(f"Preloading avatar {avatar_id} materials to GPU...")
                    avatar.input_latent_list_cycle = [
                        latent.to(self.models['device'])
                        for latent in avatar.input_latent_list_cycle
                    ]
                
                self.avatars[avatar_id] = avatar
                logger.info(f"Avatar {avatar_id} fully preloaded and ready")
                
            except Exception as e:
                logger.error(f"Failed to initialize avatar {avatar_id}: {e}")
                # Continue with other avatars even if one fails
                continue
        
        if not self.avatars:
            raise RuntimeError("No avatars were successfully initialized")
            
        logger.info(f"Successfully preloaded {len(self.avatars)} avatars")
        
        # Log memory usage if CUDA is available
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
            logger.info(f"GPU memory allocated: {memory_allocated:.2f} GB")
    
    async def handle_websocket(self, websocket, path):
        """Handle WebSocket connections for signaling"""
        client_id = str(uuid.uuid4())
        logger.info(f"New WebSocket connection: {client_id}")
        
        try:
            async for message in websocket:
                data = json.loads(message)
                await self.handle_signaling_message(websocket, client_id, data)
                
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"WebSocket connection closed: {client_id}")
        except Exception as e:
            logger.error(f"WebSocket error for {client_id}: {e}")
        finally:
            # Cleanup
            if client_id in self.peer_connections:
                await self.peer_connections[client_id].close()
                del self.peer_connections[client_id]
            if client_id in self.video_tracks:
                del self.video_tracks[client_id]
    
    async def handle_signaling_message(self, websocket, client_id: str, data: Dict[str, Any]):
        """Handle WebRTC signaling messages"""
        message_type = data.get("type")
        
        if message_type == "get_avatars":
            # Send available avatars
            avatar_list = [
                {
                    "id": avatar_id,
                    "name": avatar_id,
                    "video_path": self.config[avatar_id]["video_path"]
                }
                for avatar_id in self.avatars.keys()
            ]
            await websocket.send(json.dumps({
                "type": "avatars",
                "avatars": avatar_list
            }))
            
        elif message_type == "select_avatar":
            avatar_id = data.get("avatar_id")
            if avatar_id in self.avatars:
                # Create video track for this client
                video_track = MuseTalkVideoTrack(
                    self.avatars[avatar_id],
                    self.audio_processor,
                    self.models
                )
                self.video_tracks[client_id] = video_track
                
                await websocket.send(json.dumps({
                    "type": "avatar_selected",
                    "avatar_id": avatar_id
                }))
            else:
                await websocket.send(json.dumps({
                    "type": "error",
                    "message": f"Avatar {avatar_id} not found"
                }))
                
        elif message_type == "offer":
            # Handle WebRTC offer
            await self.handle_offer(websocket, client_id, data)
            
        elif message_type == "answer":
            # Handle WebRTC answer
            await self.handle_answer(client_id, data)
            
        elif message_type == "ice_candidate":
            # Handle ICE candidate
            await self.handle_ice_candidate(client_id, data)
    
    async def handle_offer(self, websocket, client_id: str, data: Dict[str, Any]):
        """Handle WebRTC offer"""
        try:
            pc = RTCPeerConnection(configuration=self.rtc_config)
            self.peer_connections[client_id] = pc
            
            # Add video track if available
            if client_id in self.video_tracks:
                pc.addTrack(self.video_tracks[client_id])
            
            # Handle incoming audio track
            @pc.on("track")
            async def on_track(track):
                logger.info(f"Received {track.kind} track from {client_id}")
                if track.kind == "audio" and client_id in self.video_tracks:
                    # Process audio frames
                    video_track = self.video_tracks[client_id]
                    
                    async def process_audio():
                        try:
                            async for frame in track:
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
            await websocket.send(json.dumps({
                "type": "answer",
                "sdp": pc.localDescription.sdp,
            }))
            
        except Exception as e:
            logger.error(f"Error handling offer: {e}")
            await websocket.send(json.dumps({
                "type": "error",
                "message": str(e)
            }))
    
    async def handle_answer(self, client_id: str, data: Dict[str, Any]):
        """Handle WebRTC answer"""
        if client_id in self.peer_connections:
            pc = self.peer_connections[client_id]
            await pc.setRemoteDescription(RTCSessionDescription(
                sdp=data["sdp"], type=data["type"]
            ))
    
    async def handle_ice_candidate(self, client_id: str, data: Dict[str, Any]):
        """Handle ICE candidate"""
        if client_id in self.peer_connections:
            pc = self.peer_connections[client_id]
            candidate = data.get("candidate")
            if candidate:
                await pc.addIceCandidate(candidate)
    
    async def start_server(self, host: str = "localhost", port: int = 8765):
        """Start the WebRTC server"""
        await self.initialize_models()
        await self.initialize_avatars()
        
        logger.info(f"Starting WebRTC server on {host}:{port}")
        
        async with serve(self.handle_websocket, host, port):
            logger.info("WebRTC server started successfully")
            await asyncio.Future()  # Run forever

# Enhanced AudioProcessor for real-time processing
class RealTimeAudioProcessor:
    def __init__(self, feature_extractor_path: str):
        self.feature_extractor_path = feature_extractor_path
        self.buffer_size = 1600  # 100ms at 16kHz
        self.audio_buffer = []
        self.frame_count = 0
        
    def process_audio_chunk(self, audio_data: np.ndarray, sample_rate: int, device, weight_dtype):
        """Process real-time audio chunk"""
        try:
            # Resample to 16kHz if needed
            if sample_rate != 16000:
                audio_data = self.resample_audio(audio_data, sample_rate, 16000)
            
            # Ensure audio_data is 1D
            if len(audio_data.shape) > 1:
                audio_data = audio_data.flatten()
            
            # Add to buffer
            self.audio_buffer.extend(audio_data.tolist())
            
            # Process if we have enough data
            if len(self.audio_buffer) >= self.buffer_size:
                # Take chunk from buffer
                chunk = np.array(self.audio_buffer[:self.buffer_size], dtype=np.float32)
                self.audio_buffer = self.audio_buffer[self.buffer_size//4:]  # 25% overlap for smoother processing
                
                # Normalize audio
                chunk = chunk / (np.max(np.abs(chunk)) + 1e-8)
                
                # Create audio features (simplified for real-time)
                # Generate features based on audio characteristics
                audio_energy = np.mean(chunk ** 2)
                audio_rms = np.sqrt(audio_energy)
                
                # Create more realistic features based on audio content
                # This is a simplified version - in production you'd use actual Whisper
                feature_dim = 384
                seq_len = 50
                
                # Generate features with some audio-based variation
                base_features = torch.randn(1, seq_len, feature_dim, device=device, dtype=weight_dtype)
                
                # Modulate features based on audio energy
                energy_factor = min(audio_rms * 5, 1.0)  # Scale energy to 0-1 range
                base_features = base_features * (0.5 + energy_factor * 0.5)
                
                self.frame_count += 1
                return base_features
                
        except Exception as e:
            logger.error(f"Error processing audio chunk: {e}")
            
        return None
    
    def resample_audio(self, audio_data: np.ndarray, orig_sr: int, target_sr: int):
        """Simple audio resampling"""
        if orig_sr == target_sr:
            return audio_data
            
        # Simple linear interpolation resampling
        ratio = target_sr / orig_sr
        new_length = int(len(audio_data) * ratio)
        if new_length == 0:
            return np.array([])
        indices = np.linspace(0, len(audio_data) - 1, new_length)
        return np.interp(indices, np.arange(len(audio_data)), audio_data)
    
    def get_audio_feature(self, audio_path, weight_dtype=torch.float32):
        """Compatibility method for existing AudioProcessor interface"""
        # This is for file-based processing - not used in real-time streaming
        pass
    
    def get_whisper_chunk(self, *args, **kwargs):
        """Compatibility method for existing AudioProcessor interface"""
        # This is for file-based processing - not used in real-time streaming
        pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost", help="Server host")
    parser.add_argument("--port", type=int, default=8765, help="Server port")
    parser.add_argument("--config", type=str, default="configs/inference/realtime.yaml", help="Config file")
    
    args = parser.parse_args()
    
    server = MuseTalkWebRTCServer(args.config)
    asyncio.run(server.start_server(args.host, args.port))