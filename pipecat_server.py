#!/usr/bin/env python3
"""
MuseTalk WebRTC Server using Pipecat
"""

import asyncio
import logging
import os
import sys
from pathlib import Path
import argparse
import numpy as np
import cv2
import torch
from typing import AsyncGenerator

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Pipecat imports
from pipecat.frames.frames import Frame, AudioRawFrame, ImageRawFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.transports.services.daily import DailyParams, DailyTransport
# VAD import - try different locations based on Pipecat version
try:
    from pipecat.vad.silero import SileroVADAnalyzer
except ImportError:
    try:
        from pipecat.services.silero import SileroVADAnalyzer
    except ImportError:
        # Fallback - no VAD
        SileroVADAnalyzer = None

# MuseTalk imports
from musetalk.utils.utils import load_all_model
from musetalk.utils.face_parsing import FaceParsing
from musetalk.utils.audio_processor import AudioProcessor
from scripts.realtime_inference import Avatar
from omegaconf import OmegaConf

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MuseTalkProcessor(FrameProcessor):
    """MuseTalk frame processor for real-time avatar generation"""
    
    def __init__(self, avatar_config_path: str = "configs/inference/realtime.yaml"):
        super().__init__()
        self.avatar_config_path = avatar_config_path
        self.avatars = {}
        self.models = {}
        self.audio_processor = None
        self.current_avatar = None
        self.audio_buffer = []
        self.buffer_size = 1600  # 100ms at 16kHz
        self.frame_count = 0
        
    async def start(self, frame: Frame) -> None:
        """Initialize MuseTalk models and avatars"""
        await super().start(frame)
        await self.initialize_models()
        await self.initialize_avatars()
        
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
        logger.info("Initializing avatars...")
        
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
        config = OmegaConf.load(self.avatar_config_path)
        
        for avatar_id in config:
            try:
                avatar_config = config[avatar_id]
                logger.info(f"Loading avatar {avatar_id}...")
                
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
                logger.info(f"Avatar {avatar_id} loaded successfully")
                
            except Exception as e:
                logger.error(f"Failed to load avatar {avatar_id}: {e}")
        
        # Set first avatar as current
        if self.avatars:
            self.current_avatar = list(self.avatars.values())[0]
            logger.info(f"Using avatar: {self.current_avatar.avatar_id}")
        else:
            raise RuntimeError("No avatars loaded")
    
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process incoming frames"""
        
        if isinstance(frame, AudioRawFrame):
            # Process audio frame
            try:
                # Convert audio to numpy array
                audio_data = np.frombuffer(frame.audio, dtype=np.int16).astype(np.float32) / 32768.0
                
                # Add to buffer
                self.audio_buffer.extend(audio_data.tolist())
                
                # Process if we have enough data
                if len(self.audio_buffer) >= self.buffer_size:
                    # Take chunk from buffer
                    chunk = np.array(self.audio_buffer[:self.buffer_size], dtype=np.float32)
                    self.audio_buffer = self.audio_buffer[self.buffer_size//4:]  # 25% overlap
                    
                    # Generate video frame
                    video_frame = await self.generate_video_frame(chunk)
                    if video_frame is not None:
                        await self.push_frame(video_frame, direction)
                        
            except Exception as e:
                logger.error(f"Error processing audio frame: {e}")
        else:
            # Pass through other frames (non-audio frames)
            await self.push_frame(frame, direction)
    
    async def generate_video_frame(self, audio_chunk: np.ndarray) -> ImageRawFrame:
        """Generate video frame from audio chunk"""
        try:
            if not self.current_avatar:
                return None
                
            # Normalize audio
            audio_chunk = audio_chunk / (np.max(np.abs(audio_chunk)) + 1e-8)
            
            # Create audio features (simplified for real-time)
            audio_energy = np.mean(audio_chunk ** 2)
            audio_rms = np.sqrt(audio_energy)
            
            # Generate features with audio-based variation
            feature_dim = 384
            seq_len = 50
            
            base_features = torch.randn(1, seq_len, feature_dim, 
                                      device=self.models['device'], 
                                      dtype=self.models['weight_dtype'])
            
            # Modulate features based on audio energy
            energy_factor = min(audio_rms * 5, 1.0)
            audio_features = base_features * (0.5 + energy_factor * 0.5)
            
            # Get current avatar frame and coordinates
            frame_idx = self.frame_count % len(self.current_avatar.frame_list_cycle)
            ori_frame = self.current_avatar.frame_list_cycle[frame_idx].copy()
            bbox = self.current_avatar.coord_list_cycle[frame_idx]
            latent = self.current_avatar.input_latent_list_cycle[frame_idx % len(self.current_avatar.input_latent_list_cycle)]
            
            # Skip if no valid bbox
            from musetalk.utils.preprocessing import coord_placeholder
            if bbox == coord_placeholder:
                # Convert to RGB and create frame
                frame_rgb = cv2.cvtColor(ori_frame, cv2.COLOR_BGR2RGB)
                return ImageRawFrame(
                    image=frame_rgb.tobytes(),
                    size=(frame_rgb.shape[1], frame_rgb.shape[0]),
                    format="RGB"
                )
            
            # Prepare latent batch
            latent_batch = latent.unsqueeze(0).to(
                device=self.models['device'], 
                dtype=self.models['weight_dtype']
            )
            
            # Audio feature batch
            audio_feature_batch = self.models['pe'](audio_features)
            
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
            mask = self.current_avatar.mask_list_cycle[frame_idx]
            mask_crop_box = self.current_avatar.mask_coords_list_cycle[frame_idx]
            
            from musetalk.utils.blending import get_image_blending
            combined_frame = get_image_blending(
                ori_frame, res_frame_resized, bbox, mask, mask_crop_box
            )
            
            self.frame_count += 1
            
            # Convert to RGB and create frame
            frame_rgb = cv2.cvtColor(combined_frame, cv2.COLOR_BGR2RGB)
            return ImageRawFrame(
                image=frame_rgb.tobytes(),
                size=(frame_rgb.shape[1], frame_rgb.shape[0]),
                format="RGB"
            )
            
        except Exception as e:
            logger.error(f"Error generating video frame: {e}")
            return None

async def main():
    parser = argparse.ArgumentParser(description="MuseTalk Pipecat WebRTC Server")
    parser.add_argument("--room-url", type=str, required=True, help="Daily.co room URL")
    parser.add_argument("--token", type=str, help="Daily.co room token")
    parser.add_argument("--config", type=str, default="configs/inference/realtime.yaml", help="Avatar config file")
    
    args = parser.parse_args()
    
    # Create MuseTalk processor
    musetalk_processor = MuseTalkProcessor(args.config)
    
    # Create VAD analyzer if available
    vad_analyzer = SileroVADAnalyzer() if SileroVADAnalyzer else None
    
    # Create Daily transport
    transport_params = DailyParams(
        audio_out_enabled=True,
        camera_out_enabled=True,
        camera_out_width=512,
        camera_out_height=512,
        vad_enabled=vad_analyzer is not None,
    )
    
    if vad_analyzer:
        transport_params.vad_analyzer = vad_analyzer
    
    transport = DailyTransport(
        room_url=args.room_url,
        token=args.token,
        bot_name="MuseTalk Avatar",
        params=transport_params
    )
    
    # Create pipeline
    pipeline = Pipeline([
        transport.input(),
        musetalk_processor,
        transport.output()
    ])
    
    # Create and run task
    task = PipelineTask(pipeline)
    
    # Create runner
    runner = PipelineRunner()
    
    logger.info(f"Starting MuseTalk WebRTC server...")
    logger.info(f"Room URL: {args.room_url}")
    
    await runner.run(task)

if __name__ == "__main__":
    asyncio.run(main())