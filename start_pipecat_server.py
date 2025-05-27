#!/usr/bin/env python3
"""
MuseTalk Pipecat WebRTC Server Startup Script
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

def setup_logging(level=logging.INFO):
    """Setup logging configuration"""
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('musetalk_pipecat.log')
        ]
    )

def check_dependencies():
    """Check if required dependencies are available"""
    required_modules = [
        'torch', 'cv2', 'numpy', 'pipecat', 'daily',
        'transformers', 'omegaconf'
    ]
    
    missing_modules = []
    for module in required_modules:
        try:
            if module == 'cv2':
                import cv2
            elif module == 'pipecat':
                import pipecat
            elif module == 'daily':
                import daily
            else:
                __import__(module)
        except ImportError:
            missing_modules.append(module)
    
    if missing_modules:
        print("Missing required modules:")
        for module in missing_modules:
            print(f"  - {module}")
        print("\nPlease install missing dependencies:")
        print("pip install -r requirements_pipecat.txt")
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

async def main():
    parser = argparse.ArgumentParser(
        description="MuseTalk Pipecat WebRTC Server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Using Daily.co room
  python start_pipecat_server.py --room-url https://your-domain.daily.co/room-name
  
  # With authentication token
  python start_pipecat_server.py --room-url https://your-domain.daily.co/room-name --token your-token
  
  # With custom avatar config
  python start_pipecat_server.py --room-url https://your-domain.daily.co/room-name --config custom.yaml
        """
    )
    
    parser.add_argument(
        "--room-url", 
        type=str, 
        help="Daily.co room URL (e.g., https://your-domain.daily.co/room-name)"
    )
    parser.add_argument(
        "--token", 
        type=str, 
        help="Daily.co room token (optional for public rooms)"
    )
    parser.add_argument(
        "--config", 
        type=str, 
        default="configs/inference/realtime.yaml", 
        help="Avatar configuration file (default: configs/inference/realtime.yaml)"
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
    print("MuseTalk Pipecat WebRTC Server")
    print("=" * 60)
    
    if not args.skip_checks:
        print("Checking dependencies...")
        if not check_dependencies():
            sys.exit(1)
        
        print("Checking model files...")
        if not check_model_files():
            sys.exit(1)
    
    if not args.room_url:
        print("Error: --room-url is required")
        print("\nTo create a Daily.co room:")
        print("1. Go to https://dashboard.daily.co/")
        print("2. Create a new room")
        print("3. Copy the room URL")
        print("4. Run: python start_pipecat_server.py --room-url <your-room-url>")
        sys.exit(1)
    
    print("All checks passed!")
    print()
    
    try:
        # Import and start the server
        from pipecat_server import main as pipecat_main
        
        print(f"Starting MuseTalk Pipecat server...")
        print(f"Room URL: {args.room_url}")
        if args.token:
            print("Using authentication token")
        print("Press Ctrl+C to stop the server")
        print()
        
        # Override sys.argv for the pipecat server
        sys.argv = [
            "pipecat_server.py",
            "--room-url", args.room_url,
            "--config", args.config
        ]
        if args.token:
            sys.argv.extend(["--token", args.token])
        
        await pipecat_main()
        
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