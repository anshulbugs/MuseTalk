#!/usr/bin/env python3
"""
Test script to verify WebRTC streaming setup for MuseTalk
"""

import sys
import os
import asyncio
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_imports():
    """Test if all required modules can be imported"""
    print("Testing imports...")
    
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__}")
    except ImportError as e:
        print(f"✗ PyTorch: {e}")
        return False
    
    try:
        import cv2
        print(f"✓ OpenCV {cv2.__version__}")
    except ImportError as e:
        print(f"✗ OpenCV: {e}")
        return False
    
    try:
        import numpy as np
        print(f"✓ NumPy {np.__version__}")
    except ImportError as e:
        print(f"✗ NumPy: {e}")
        return False
    
    try:
        import aiortc
        print(f"✓ aiortc {aiortc.__version__}")
    except ImportError as e:
        print(f"✗ aiortc: {e}")
        return False
    
    try:
        import aiohttp
        print(f"✓ aiohttp {aiohttp.__version__}")
    except ImportError as e:
        print(f"✗ aiohttp: {e}")
        return False
    
    try:
        import websockets
        print(f"✓ websockets {websockets.__version__}")
    except ImportError as e:
        print(f"✗ websockets: {e}")
        return False
    
    try:
        from transformers import WhisperModel
        print("✓ transformers (Whisper)")
    except ImportError as e:
        print(f"✗ transformers: {e}")
        return False
    
    try:
        from omegaconf import OmegaConf
        print("✓ omegaconf")
    except ImportError as e:
        print(f"✗ omegaconf: {e}")
        return False
    
    return True

def test_model_files():
    """Test if required model files exist"""
    print("\nTesting model files...")
    
    required_files = [
        "models/musetalkV15/unet.pth",
        "models/musetalkV15/musetalk.json",
        "models/sd-vae/config.json",
        "models/whisper/config.json"
    ]
    
    all_exist = True
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✓ {file_path}")
        else:
            print(f"✗ {file_path} (missing)")
            all_exist = False
    
    return all_exist

def test_config_file():
    """Test if configuration file exists and is valid"""
    print("\nTesting configuration file...")
    
    config_path = "configs/inference/realtime.yaml"
    if not os.path.exists(config_path):
        print(f"✗ {config_path} (missing)")
        return False
    
    try:
        from omegaconf import OmegaConf
        config = OmegaConf.load(config_path)
        print(f"✓ {config_path}")
        
        # Check if at least one avatar is configured
        if len(config) == 0:
            print("✗ No avatars configured")
            return False
        
        print(f"✓ Found {len(config)} avatar(s) configured")
        
        # Check avatar configurations
        for avatar_id, avatar_config in config.items():
            video_path = avatar_config.get("video_path", "")
            if os.path.exists(video_path):
                print(f"✓ Avatar '{avatar_id}' video: {video_path}")
            else:
                print(f"✗ Avatar '{avatar_id}' video missing: {video_path}")
                return False
        
        return True
        
    except Exception as e:
        print(f"✗ Error loading config: {e}")
        return False

def test_gpu_availability():
    """Test GPU availability"""
    print("\nTesting GPU availability...")
    
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            current_device = torch.cuda.current_device()
            gpu_name = torch.cuda.get_device_name(current_device)
            print(f"✓ CUDA available: {gpu_count} GPU(s)")
            print(f"✓ Current GPU: {gpu_name}")
            
            # Test memory allocation
            try:
                test_tensor = torch.randn(100, 100).cuda()
                memory_allocated = torch.cuda.memory_allocated() / 1024**2  # MB
                print(f"✓ GPU memory test passed ({memory_allocated:.1f} MB allocated)")
                del test_tensor
                torch.cuda.empty_cache()
                return True
            except Exception as e:
                print(f"✗ GPU memory test failed: {e}")
                return False
        else:
            print("⚠ CUDA not available, will use CPU")
            return True
            
    except Exception as e:
        print(f"✗ GPU test error: {e}")
        return False

async def test_server_components():
    """Test server components initialization"""
    print("\nTesting server components...")
    
    try:
        from webrtc_server import MuseTalkWebRTCServer
        
        # Test server creation
        server = MuseTalkWebRTCServer("configs/inference/realtime.yaml")
        print("✓ WebRTC server created")
        
        # Test model initialization
        await server.initialize_models()
        print("✓ Models initialized")
        
        # Test avatar initialization
        await server.initialize_avatars()
        print(f"✓ Avatars initialized ({len(server.avatars)} avatars)")
        
        return True
        
    except Exception as e:
        print(f"✗ Server component test failed: {e}")
        return False

def test_static_files():
    """Test if static files exist"""
    print("\nTesting static files...")
    
    static_files = [
        "static/index.html"
    ]
    
    all_exist = True
    for file_path in static_files:
        if os.path.exists(file_path):
            print(f"✓ {file_path}")
        else:
            print(f"✗ {file_path} (missing)")
            all_exist = False
    
    return all_exist

async def main():
    """Run all tests"""
    print("=" * 60)
    print("MuseTalk WebRTC Setup Test")
    print("=" * 60)
    
    tests = [
        ("Import Test", test_imports),
        ("Model Files Test", test_model_files),
        ("Configuration Test", test_config_file),
        ("GPU Test", test_gpu_availability),
        ("Static Files Test", test_static_files),
        ("Server Components Test", test_server_components)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'-' * 40}")
        print(f"Running {test_name}...")
        print(f"{'-' * 40}")
        
        try:
            if asyncio.iscoroutinefunction(test_func):
                result = await test_func()
            else:
                result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print(f"\n{'=' * 60}")
    print("Test Summary")
    print(f"{'=' * 60}")
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        symbol = "✓" if result else "✗"
        print(f"{symbol} {test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Your WebRTC setup is ready.")
        print("\nTo start the server, run:")
        print("python start_webrtc_server.py")
    else:
        print(f"\n⚠ {total - passed} test(s) failed. Please fix the issues above.")
        
        if not test_imports():
            print("\nTo install missing dependencies:")
            print("pip install -r requirements_webrtc.txt")
        
        if not test_model_files():
            print("\nTo download missing models:")
            print("python download_weights.py  # or run download_weights.bat/sh")

if __name__ == "__main__":
    asyncio.run(main())