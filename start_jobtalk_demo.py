#!/usr/bin/env python3
"""
JobTalk MuseTalk Demo Startup Script
Starts both the MuseTalk server and the JobTalk HTML server
"""

import asyncio
import subprocess
import sys
import time
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def start_musetalk_server():
    """Start the MuseTalk server"""
    logger.info("Starting MuseTalk server...")
    return subprocess.Popen([
        sys.executable, 'simple_video_server.py'
    ], cwd=Path(__file__).parent)

def start_jobtalk_server():
    """Start the JobTalk HTML server"""
    logger.info("Starting JobTalk HTML server...")
    return subprocess.Popen([
        sys.executable, 'serve_jobtalk.py'
    ], cwd=Path(__file__).parent)

async def main():
    logger.info("🎭 Starting JobTalk MuseTalk Demo")
    logger.info("=" * 50)
    
    # Start MuseTalk server
    musetalk_process = start_musetalk_server()
    
    # Wait a bit for MuseTalk to start
    logger.info("Waiting for MuseTalk server to initialize...")
    time.sleep(3)
    
    # Start JobTalk HTML server
    jobtalk_process = start_jobtalk_server()
    
    # Wait a bit for HTML server to start
    time.sleep(2)
    
    logger.info("=" * 50)
    logger.info("🚀 Both servers are starting up!")
    logger.info("📱 Open your browser and go to: http://localhost:3000")
    logger.info("🎤 Click 'Start Interview' to begin the AI interview")
    logger.info("🎭 The avatar will lip-sync with the AI interviewer")
    logger.info("=" * 50)
    logger.info("Press Ctrl+C to stop both servers")
    
    try:
        # Keep running until interrupted
        while True:
            await asyncio.sleep(1)
            
            # Check if processes are still running
            if musetalk_process.poll() is not None:
                logger.error("MuseTalk server stopped unexpectedly")
                break
            if jobtalk_process.poll() is not None:
                logger.error("JobTalk server stopped unexpectedly")
                break
                
    except KeyboardInterrupt:
        logger.info("Shutting down servers...")
        
        # Terminate processes
        musetalk_process.terminate()
        jobtalk_process.terminate()
        
        # Wait for clean shutdown
        try:
            musetalk_process.wait(timeout=5)
            jobtalk_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            logger.warning("Force killing processes...")
            musetalk_process.kill()
            jobtalk_process.kill()
        
        logger.info("Demo stopped successfully")

if __name__ == "__main__":
    asyncio.run(main())