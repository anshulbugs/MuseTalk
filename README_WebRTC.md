# MuseTalk WebRTC Real-Time Streaming

This document describes the WebRTC streaming implementation for MuseTalk, which enables real-time avatar-based video streaming with audio input processing.

## Overview

The WebRTC streaming system allows you to:
- Select an avatar from pre-configured options
- Stream your audio input in real-time
- Receive synchronized video output with the avatar speaking
- Use any WebRTC-compatible browser for streaming

## Architecture

### Components

1. **WebRTC Server** (`webrtc_server.py`)
   - Handles WebRTC peer connections
   - Manages avatar video track generation
   - Processes real-time audio input
   - Generates synchronized video frames

2. **HTTP Server** (`http_server.py`)
   - Serves the web interface
   - Handles WebSocket signaling
   - Manages client connections

3. **Web Client** (`static/index.html`)
   - Browser-based interface
   - WebRTC peer connection management
   - Audio/video stream handling

4. **Avatar Management**
   - Pre-loads all avatar resources at startup
   - Maintains frame cycles and latent representations
   - Handles real-time frame blending

## Installation

### 1. Install Dependencies

```bash
# Install WebRTC streaming dependencies
pip install -r requirements_webrtc.txt

# Or install individually
pip install aiortc aiohttp aiohttp-cors websockets av librosa soundfile
```

### 2. Download Models

Make sure you have downloaded all required MuseTalk models:

```bash
# Windows
download_weights.bat

# Linux/Mac
./download_weights.sh
```

### 3. Configure Avatars

Edit `configs/inference/realtime.yaml` to configure your avatars:

```yaml
avatar_1:
  preparation: True  # Set to True for first-time setup
  bbox_shift: 5
  video_path: "data/video/person1.mp4"
  audio_clips:
    audio_0: "data/audio/sample1.wav"

avatar_2:
  preparation: True
  bbox_shift: 0
  video_path: "data/video/person2.mp4"
  audio_clips:
    audio_0: "data/audio/sample2.wav"
```

## Usage

### Starting the Server

#### Method 1: Using the startup script (Recommended)

```bash
python start_webrtc_server.py
```

#### Method 2: With custom parameters

```bash
python start_webrtc_server.py --host 0.0.0.0 --port 8080 --verbose
```

#### Method 3: Direct server start

```bash
python http_server.py --host localhost --port 8080
```

### Using the Web Interface

1. **Open Browser**: Navigate to `http://localhost:8080`
2. **Select Avatar**: Choose from available avatars in the dropdown
3. **Connect**: Click "Connect" to establish WebSocket connection
4. **Start Streaming**: Click "Start Streaming" to begin real-time processing
5. **Speak**: Your audio will be processed and the avatar will respond in real-time
6. **Stop**: Click "Stop Streaming" when finished

## Configuration Options

### Server Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--host` | localhost | Server host address |
| `--port` | 8080 | Server port |
| `--config` | configs/inference/realtime.yaml | Configuration file |
| `--verbose` | False | Enable verbose logging |
| `--skip-checks` | False | Skip dependency checks |

### Avatar Configuration

| Parameter | Description |
|-----------|-------------|
| `preparation` | Whether to preprocess avatar materials |
| `bbox_shift` | Face bounding box adjustment |
| `video_path` | Path to reference video/images |
| `audio_clips` | Sample audio files (for testing) |

## Performance Optimization

### GPU Acceleration

The system automatically uses GPU acceleration when available:
- Models are loaded in half-precision (float16) on GPU
- Avatar latents are preloaded to GPU memory
- Real-time inference uses optimized batch processing

### Memory Management

- All avatar resources are preloaded at startup
- Frame queues prevent memory buildup
- Automatic cleanup of disconnected clients

### Latency Optimization

- Batch size of 1 for minimal latency
- Asynchronous audio processing
- Frame-by-frame generation
- WebRTC optimized video encoding

## Troubleshooting

### Common Issues

#### 1. Connection Failed
```
Error: Connection failed. Make sure the server is running.
```
**Solution**: Ensure the server is started and accessible at the specified host/port.

#### 2. Missing Dependencies
```
Error importing required modules: No module named 'aiortc'
```
**Solution**: Install WebRTC dependencies:
```bash
pip install -r requirements_webrtc.txt
```

#### 3. Missing Model Files
```
Missing required model files: models/musetalkV15/unet.pth
```
**Solution**: Download model files:
```bash
# Windows
download_weights.bat

# Linux/Mac  
./download_weights.sh
```

#### 4. Avatar Loading Failed
```
Failed to initialize avatar avatar_1: No face detected
```
**Solution**: 
- Check video file path in configuration
- Adjust `bbox_shift` parameter
- Ensure video contains clear face

#### 5. WebRTC Connection Issues
```
Error: Failed to start streaming: NotAllowedError
```
**Solution**: 
- Allow camera/microphone permissions in browser
- Use HTTPS for production deployment
- Check firewall settings

### Browser Compatibility

Tested browsers:
- Chrome 90+
- Firefox 88+
- Safari 14+
- Edge 90+

### Network Requirements

- Local network: No special requirements
- Internet deployment: HTTPS required for getUserMedia()
- Firewall: Allow ports 8080 (HTTP) and WebRTC ports

## API Reference

### WebSocket Messages

#### Client to Server

```javascript
// Get available avatars
{
  "type": "get_avatars"
}

// Select avatar
{
  "type": "select_avatar", 
  "avatar_id": "avatar_1"
}

// WebRTC offer
{
  "type": "offer",
  "sdp": "..."
}

// ICE candidate
{
  "type": "ice_candidate",
  "candidate": {...}
}
```

#### Server to Client

```javascript
// Avatar list response
{
  "type": "avatars",
  "avatars": [
    {"id": "avatar_1", "name": "Avatar 1", "video_path": "..."}
  ]
}

// Avatar selection confirmation
{
  "type": "avatar_selected",
  "avatar_id": "avatar_1"
}

// WebRTC answer
{
  "type": "answer",
  "sdp": "..."
}

// Error message
{
  "type": "error", 
  "message": "Error description"
}
```

## Development

### Adding New Features

1. **Custom Audio Processing**: Modify `AudioProcessor.process_audio_chunk()`
2. **Video Effects**: Extend `MuseTalkVideoTrack.generate_frame()`
3. **Avatar Management**: Update avatar loading in `initialize_avatars()`
4. **UI Enhancements**: Modify `static/index.html`

### Testing

```bash
# Start server with verbose logging
python start_webrtc_server.py --verbose

# Test with multiple clients
# Open multiple browser tabs to test concurrent connections
```

### Deployment

For production deployment:

1. **Use HTTPS**: Required for getUserMedia() API
2. **Configure STUN/TURN**: For NAT traversal
3. **Load Balancing**: For multiple server instances
4. **Monitoring**: Add logging and metrics

## License

This WebRTC streaming extension follows the same license as the main MuseTalk project.

## Support

For issues and questions:
1. Check this documentation
2. Review server logs (`musetalk_webrtc.log`)
3. Test with verbose logging enabled
4. Check browser developer console for client-side errors