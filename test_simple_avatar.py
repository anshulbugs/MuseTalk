#!/usr/bin/env python3
"""
Simple test to verify original Avatar class works with our WebRTC setup
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_original_avatar():
    """Test using the original Avatar class"""
    try:
        from omegaconf import OmegaConf
        from musetalk.utils.utils import load_all_model
        from musetalk.utils.face_parsing import FaceParsing
        from scripts.realtime_inference import Avatar
        from argparse import Namespace
        import torch
        
        print("Loading models...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load models
        vae, unet, pe = load_all_model(
            unet_model_path="./models/musetalkV15/unet.pth",
            vae_type="sd-vae", 
            unet_config="./models/musetalkV15/musetalk.json",
            device=device
        )
        
        # Create face parser
        fp = FaceParsing()
        
        # Create args object
        args = Namespace(
            version="v15",
            extra_margin=10,
            parsing_mode="jaw",
            left_cheek_width=90,
            right_cheek_width=90
        )
        
        # Set globals that Avatar class expects
        import scripts.realtime_inference
        scripts.realtime_inference.args = args
        scripts.realtime_inference.vae = vae
        scripts.realtime_inference.fp = fp
        
        print("✓ Models and globals set")
        
        # Load config
        config = OmegaConf.load("configs/inference/realtime.yaml")
        avatar_id = list(config.keys())[0]
        avatar_config = config[avatar_id]
        
        print(f"Creating avatar: {avatar_id}")
        
        # Create avatar using original class
        avatar = Avatar(
            avatar_id=avatar_id,
            video_path=avatar_config["video_path"],
            bbox_shift=avatar_config.get("bbox_shift", 0),
            batch_size=1,
            preparation=True  # This will create materials if they don't exist
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
        
        print("✓ All avatar materials loaded successfully")
        return True
        
    except Exception as e:
        print(f"✗ Avatar test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=" * 50)
    print("Simple Avatar Test")
    print("=" * 50)
    
    success = test_original_avatar()
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 Avatar test PASSED! Original Avatar class works.")
        print("You can now start the WebRTC server:")
        print("python start_webrtc_server.py")
    else:
        print("⚠ Avatar test FAILED. Check the errors above.")