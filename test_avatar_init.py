#!/usr/bin/env python3
"""
Test script to verify avatar initialization works correctly
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_avatar_creation():
    """Test creating a WebRTC avatar"""
    try:
        from webrtc_avatar import WebRTCAvatar
        from omegaconf import OmegaConf
        
        # Load config
        config_path = "configs/inference/realtime.yaml"
        if not os.path.exists(config_path):
            print(f"Config file not found: {config_path}")
            return False
            
        config = OmegaConf.load(config_path)
        
        # Test with first avatar
        avatar_id = list(config.keys())[0]
        avatar_config = config[avatar_id]
        
        print(f"Testing avatar creation for: {avatar_id}")
        print(f"Video path: {avatar_config['video_path']}")
        
        # Create avatar
        avatar = WebRTCAvatar(
            avatar_id=avatar_id,
            video_path=avatar_config["video_path"],
            bbox_shift=avatar_config.get("bbox_shift", 0),
            batch_size=1,
            preparation=True,
            version="v15",
            extra_margin=10,
            parsing_mode="jaw",
            left_cheek_width=90,
            right_cheek_width=90
        )
        
        print("✓ Avatar created successfully")
        
        # Check required attributes
        required_attrs = [
            'frame_list_cycle', 'coord_list_cycle', 
            'input_latent_list_cycle', 'mask_list_cycle', 
            'mask_coords_list_cycle'
        ]
        
        for attr in required_attrs:
            if hasattr(avatar, attr) and getattr(avatar, attr) is not None:
                print(f"✓ {attr}: {len(getattr(avatar, attr))} items")
            else:
                print(f"✗ {attr}: missing or None")
                return False
        
        return True
        
    except Exception as e:
        print(f"✗ Avatar creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_models_loading():
    """Test loading MuseTalk models"""
    try:
        from musetalk.utils.utils import load_all_model
        import torch
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        
        # Load models
        vae, unet, pe = load_all_model(
            unet_model_path="./models/musetalkV15/unet.pth",
            vae_type="sd-vae", 
            unet_config="./models/musetalkV15/musetalk.json",
            device=device
        )
        
        print("✓ Models loaded successfully")
        return True, (vae, unet, pe)
        
    except Exception as e:
        print(f"✗ Model loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None

def test_latent_computation():
    """Test latent computation with VAE"""
    try:
        # Load models first
        success, models = test_models_loading()
        if not success:
            return False
            
        vae, unet, pe = models
        
        # Create avatar
        from webrtc_avatar import WebRTCAvatar
        from omegaconf import OmegaConf
        
        config = OmegaConf.load("configs/inference/realtime.yaml")
        avatar_id = list(config.keys())[0]
        avatar_config = config[avatar_id]
        
        avatar = WebRTCAvatar(
            avatar_id=avatar_id,
            video_path=avatar_config["video_path"],
            bbox_shift=avatar_config.get("bbox_shift", 0),
            batch_size=1,
            preparation=False,  # Use existing materials
            version="v15"
        )
        
        # Compute latents
        avatar.compute_latents(vae)
        print("✓ Latents computed successfully")
        
        # Check latents
        if hasattr(avatar, 'input_latent_list_cycle'):
            print(f"✓ Latent list has {len(avatar.input_latent_list_cycle)} items")
            if len(avatar.input_latent_list_cycle) > 0:
                first_latent = avatar.input_latent_list_cycle[0]
                print(f"✓ First latent shape: {first_latent.shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ Latent computation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=" * 50)
    print("Avatar Initialization Test")
    print("=" * 50)
    
    print("\n1. Testing avatar creation...")
    avatar_success = test_avatar_creation()
    
    print("\n2. Testing latent computation...")
    latent_success = test_latent_computation()
    
    print("\n" + "=" * 50)
    print("Test Results:")
    print(f"Avatar Creation: {'PASS' if avatar_success else 'FAIL'}")
    print(f"Latent Computation: {'PASS' if latent_success else 'FAIL'}")
    
    if avatar_success and latent_success:
        print("\n🎉 All tests passed! Avatar initialization is working.")
    else:
        print("\n⚠ Some tests failed. Check the errors above.")