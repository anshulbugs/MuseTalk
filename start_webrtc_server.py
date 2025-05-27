#!/usr/bin/env python3
"""
MuseTalk WebRTC Streaming Server Startup Script

This script starts the MuseTalk WebRTC streaming server with proper initialization
and error handling.
"""

import asyncio
import sys
import os
import argparse
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

try:
    from http_server import MuseTalkHTTPServer
except ImportError as e:
    print(f"Error importing required modules: {e}")
    print("Please make sure you have installed all dependencies:")
    print("pip install -r requirements_webrtc.txt")
    sys.exit(1)

def setup_logging(level=logging.INFO):
    """Setup logging configuration"""
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('musetalk_webrtc.log')
        ]
    )

def check_dependencies():
    """Check if all required dependencies are available"""
    required_modules = [
        'torch', 'cv2', 'numpy', 'aiortc', 'aiohttp', 
        'websockets', 'transformers', 'omegaconf'
    ]
    
    missing_modules = []
    for module in required_modules:
        try:
            __import__(module)
        except ImportError:
            missing_modules.append(module)
    
    if missing_modules:
        print("Missing required modules:")
        for module in missing_modules:
            print(f"  - {module}")
        print("\nPlease install missing dependencies:")
        print("pip install -r requirements_webrtc.txt")
        return False
    
    return True

def check_model_files():
    """Check if required model files exist"""
    model_paths = [
        "models/musetalkV15/unet.pth",
        "models/musetalkV15/musetalk.json",
        "models/sd-vae/config.json",
        "models/whisper/config.json"
    ]
    
    missing_files = []
    for path in model_paths:
        if not os.path.exists(path):
            missing_files.append(path)
    
    if missing_files:
        print("Missing required model files:")
        for file in missing_files:
            print(f"  - {file}")
        print("\nPlease download the required models using:")
        if sys.platform == "win32":
            print("download_weights.bat")
        else:
            print("./download_weights.sh")
        return False
    
    return True

def check_config_file(config_path):
    """Check if configuration file exists"""
    if not os.path.exists(config_path):
        print(f"Configuration file not found: {config_path}")
        print("Please make sure the configuration file exists.")
        return False
    return True

async def main():
    parser = argparse.ArgumentParser(
        description="MuseTalk WebRTC Streaming Server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python start_webrtc_server.py
  python start_webrtc_server.py --host 0.0.0.0 --port 8080
  python start_webrtc_server.py --config configs/inference/custom.yaml --verbose
        """
    )
    
    parser.add_argument(
        "--host", 
        type=str, 
        default="localhost", 
        help="Server host address (default: localhost)"
    )
    parser.add_argument(
        "--port", 
        type=int, 
        default=8080, 
        help="Server port (default: 8080)"
    )
    parser.add_argument(
        "--config", 
        type=str, 
        default="configs/inference/realtime.yaml", 
        help="Configuration file path (default: configs/inference/realtime.yaml)"
    )
    parser.add_argument(
        "--verbose", 
        action="store_true", 
        help="Enable verbose logging"
    )
    parser.add_argument(
        "--skip-checks", 
        action="store_true", 
        help="Skip dependency and model file checks"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(log_level)
    
    logger = logging.getLogger(__name__)
    
    print("=" * 60)
    print("MuseTalk WebRTC Streaming Server")
    print("=" * 60)
    
    if not args.skip_checks:
        print("Checking dependencies...")
        if not check_dependencies():
            sys.exit(1)
        
        print("Checking model files...")
        if not check_model_files():
            sys.exit(1)
        
        print("Checking configuration file...")
        if not check_config_file(args.config):
            sys.exit(1)
    
    print("All checks passed!")
    print()
    
    try:
        # Create and start server
        server = MuseTalkHTTPServer(args.config)
        
        print(f"Starting server on http://{args.host}:{args.port}")
        print("Press Ctrl+C to stop the server")
        print()
        
        await server.start_server(args.host, args.port)
        
    except KeyboardInterrupt:
        print("\nServer stopped by user")
    except Exception as e:
        logger.error(f"Server error: {e}")
        print(f"Server error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nShutdown complete")
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)