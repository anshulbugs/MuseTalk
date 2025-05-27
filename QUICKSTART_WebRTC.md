# MuseTalk WebRTC Quick Start Guide

Get up and running with MuseTalk WebRTC streaming in minutes!

## Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended)
- Webcam and microphone
- Modern web browser (Chrome, Firefox, Safari, Edge)

## Quick Setup

### 1. Install Dependencies

```bash
# Install WebRTC streaming dependencies
pip install -r requirements_webrtc.txt
```

### 2. Download Models

```bash
# Windows
download_weights.bat

# Linux/Mac
./download_weights.sh
```

### 3. Test Setup

```bash
# Run the setup test
python test_webrtc_setup.py
```

If all tests pass, you're ready to go! If not, follow the error messages to fix any issues.

### 4. Start the Server

```bash
# Start with default settings
python start_webrtc_server.py

# Or with custom settings
python start_webrtc_server.py --host 0.0.0.0 --port 8080 --verbose
```

### 5. Open Web Interface

1. Open your browser
2. Navigate to `http://localhost:8080`
3. Allow camera and microphone permissions when prompted

## Using the Interface

### Step 1: Connect
- Click "Connect" to establish connection with the server
- Wait for "Connected" status

### Step 2: Select Avatar
- Choose an avatar from the dropdown menu
- The avatar will be preloaded and ready for streaming

### Step 3: Start Streaming
- Click "Start Streaming"
- Allow browser permissions for camera/microphone
- You should see your camera feed on the left
- The avatar will appear on the right and respond to your speech

### Step 4: Interact
- Speak into your microphone
- The avatar will generate lip-sync video in real-time
- Audio processing happens automatically

### Step 5: Stop
- Click "Stop Streaming" when finished
- You can select a different avatar and start again

## Troubleshooting

### Common Issues

**"Connection failed"**
- Make sure the server is running
- Check if port 8080 is available
- Try refreshing the browser

**"No avatars available"**
- Check your `configs/inference/realtime.yaml` file
- Ensure avatar video files exist
- Run the test script to verify setup

**"Camera/microphone not working"**
- Allow browser permissions
- Check if other applications are using the devices
- Try refreshing the page

**"Avatar not responding to audio"**
- Check browser console for errors
- Ensure microphone is working and not muted
- Try speaking louder or closer to the microphone

### Performance Tips

**For better performance:**
- Use a CUDA-compatible GPU
- Close other resource-intensive applications
- Use a wired internet connection
- Ensure good lighting for your camera

**For lower latency:**
- Use Chrome browser (best WebRTC support)
- Reduce video resolution if needed
- Ensure stable network connection

## Configuration

### Avatar Configuration

Edit `configs/inference/realtime.yaml`:

```yaml
my_avatar:
  preparation: True  # Set to True for first-time setup
  bbox_shift: 5      # Adjust face detection
  video_path: "path/to/your/video.mp4"
  audio_clips:
    sample: "path/to/sample/audio.wav"
```

### Server Configuration

```bash
# Custom host and port
python start_webrtc_server.py --host 0.0.0.0 --port 9000

# Enable verbose logging
python start_webrtc_server.py --verbose

# Use custom config file
python start_webrtc_server.py --config my_config.yaml
```

## Next Steps

- **Add your own avatars**: Place video files in the `data/video/` directory and update the config
- **Customize the interface**: Modify `static/index.html` for your needs
- **Deploy to production**: Use HTTPS and proper WebRTC TURN servers
- **Integrate with applications**: Use the WebRTC API for custom integrations

## Support

If you encounter issues:

1. Run the test script: `python test_webrtc_setup.py`
2. Check server logs: Look at `musetalk_webrtc.log`
3. Enable verbose logging: `--verbose` flag
4. Check browser console for client-side errors

## Advanced Usage

### Multiple Clients
The server supports multiple simultaneous connections. Each client can select their own avatar and stream independently.

### Custom Audio Processing
Modify the `RealTimeAudioProcessor` class in `webrtc_server.py` to implement custom audio processing or use actual Whisper models.

### Production Deployment
For production use:
- Use HTTPS (required for getUserMedia)
- Configure STUN/TURN servers for NAT traversal
- Set up load balancing for multiple server instances
- Add authentication and rate limiting

Enjoy your real-time MuseTalk streaming experience! 🎉