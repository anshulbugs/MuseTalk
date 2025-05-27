# MuseTalk Pipecat WebRTC Integration

This document describes the Pipecat-based WebRTC implementation for MuseTalk, which provides a robust, production-ready solution for real-time avatar streaming.

## Overview

Pipecat is a framework specifically designed for building real-time AI applications with WebRTC. This integration provides:

- **Robust WebRTC handling** - No manual ICE candidate or SDP management
- **Production-ready** - Built for scale and reliability
- **Daily.co integration** - Easy room management and deployment
- **Real-time processing** - Optimized for low-latency avatar generation
- **Voice Activity Detection** - Automatic speech detection

## Architecture

### Components

1. **MuseTalkProcessor** - Core processor that handles audio-to-video generation
2. **Daily Transport** - WebRTC transport layer using Daily.co
3. **VAD Analyzer** - Voice activity detection for better performance
4. **Pipeline** - Pipecat pipeline orchestrating the data flow

### Data Flow

```
Audio Input → VAD → MuseTalkProcessor → Video Output
     ↓              ↓                      ↑
Daily.co ←→ WebRTC Transport ←→ Daily.co Room
```

## Installation

### 1. Install Pipecat Dependencies

```bash
pip install -r requirements_pipecat.txt
```

### 2. Download MuseTalk Models

```bash
# Windows
download_weights.bat

# Linux/Mac
./download_weights.sh
```

### 3. Set up Daily.co Account

1. Go to [Daily.co Dashboard](https://dashboard.daily.co/)
2. Create a free account
3. Create a new room or use the default room
4. Copy the room URL (e.g., `https://your-domain.daily.co/room-name`)

## Usage

### Basic Usage

```bash
# Start with a Daily.co room URL
python start_pipecat_server.py --room-url https://your-domain.daily.co/room-name
```

### Advanced Usage

```bash
# With authentication token
python start_pipecat_server.py \
  --room-url https://your-domain.daily.co/room-name \
  --token your-room-token

# With custom avatar configuration
python start_pipecat_server.py \
  --room-url https://your-domain.daily.co/room-name \
  --config configs/inference/custom.yaml \
  --verbose
```

### Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--room-url` | Daily.co room URL (required) | - |
| `--token` | Daily.co room token (optional) | - |
| `--config` | Avatar configuration file | `configs/inference/realtime.yaml` |
| `--verbose` | Enable verbose logging | False |
| `--skip-checks` | Skip dependency checks | False |

## Daily.co Room Setup

### Creating a Room

1. **Dashboard**: Go to [Daily.co Dashboard](https://dashboard.daily.co/)
2. **Create Room**: Click "Create Room" or use the default room
3. **Room Settings**: Configure as needed (public/private, recording, etc.)
4. **Get URL**: Copy the room URL for use with the server

### Room Configuration

For optimal MuseTalk performance, configure your Daily.co room with:

- **Video**: Enabled (for avatar output)
- **Audio**: Enabled (for voice input)
- **Recording**: Optional (for saving sessions)
- **Max Participants**: Set based on your needs

### Authentication

- **Public Rooms**: No token required
- **Private Rooms**: Generate and use room tokens
- **Meeting Tokens**: For user-specific permissions

## Avatar Configuration

### Basic Configuration

Edit `configs/inference/realtime.yaml`:

```yaml
avatar_1:
  preparation: True
  bbox_shift: 5
  video_path: "data/video/person1.mp4"
  audio_clips:
    sample: "data/audio/sample1.wav"

avatar_2:
  preparation: True
  bbox_shift: 0
  video_path: "data/video/person2.mp4"
  audio_clips:
    sample: "data/audio/sample2.wav"
```

### Avatar Selection

The server automatically uses the first avatar in the configuration. To use a specific avatar:

1. Modify the configuration file to put your desired avatar first
2. Or create a custom configuration file with only your avatar

## Performance Optimization

### GPU Optimization

- **CUDA**: Automatically uses GPU if available
- **Memory**: Avatar materials preloaded to GPU memory
- **Precision**: Half-precision (float16) for faster inference

### Audio Processing

- **Buffer Size**: 100ms audio chunks (1600 samples at 16kHz)
- **Overlap**: 25% overlap for smoother processing
- **VAD**: Voice Activity Detection reduces unnecessary processing

### Network Optimization

- **WebRTC**: Peer-to-peer connections when possible
- **Daily.co**: Global edge network for low latency
- **Adaptive**: Automatic quality adjustment based on network

## Client Connection

### Web Browser

1. **Open Room**: Navigate to your Daily.co room URL in a browser
2. **Join Room**: Click "Join" and allow camera/microphone permissions
3. **See Avatar**: The MuseTalk avatar will appear as a participant
4. **Interact**: Speak to see real-time lip-sync generation

### Mobile Apps

- **Daily.co Mobile**: Use the Daily.co mobile app
- **Custom Apps**: Integrate Daily.co SDK in your mobile app

### API Integration

```javascript
// Example: Join room programmatically
const daily = DailyIframe.createFrame();
await daily.join({ url: 'https://your-domain.daily.co/room-name' });
```

## Monitoring and Debugging

### Logging

```bash
# Enable verbose logging
python start_pipecat_server.py --room-url <url> --verbose

# Check log file
tail -f musetalk_pipecat.log
```

### Performance Metrics

- **GPU Memory**: Monitor CUDA memory usage
- **Audio Latency**: Check audio processing times
- **Frame Rate**: Monitor video generation rate
- **Network**: Daily.co dashboard provides network stats

### Common Issues

**Connection Issues:**
- Check Daily.co room URL
- Verify internet connection
- Check firewall settings

**Performance Issues:**
- Ensure GPU is available and used
- Check system resources (CPU, memory)
- Reduce avatar complexity if needed

**Audio Issues:**
- Verify microphone permissions
- Check audio input levels
- Test with different browsers

## Production Deployment

### Scaling

- **Multiple Instances**: Run multiple servers for different rooms
- **Load Balancing**: Use Daily.co's built-in load balancing
- **Auto-scaling**: Deploy on cloud platforms with auto-scaling

### Security

- **Room Tokens**: Use authentication tokens for private rooms
- **HTTPS**: Daily.co enforces HTTPS for security
- **Permissions**: Configure room permissions appropriately

### Monitoring

- **Daily.co Analytics**: Built-in analytics and monitoring
- **Custom Metrics**: Add application-specific monitoring
- **Alerts**: Set up alerts for system health

## API Reference

### MuseTalkProcessor

```python
class MuseTalkProcessor(FrameProcessor):
    def __init__(self, avatar_config_path: str)
    async def initialize_models(self)
    async def initialize_avatars(self)
    async def process_frame(self, frame: Frame, direction: FrameDirection)
    async def generate_video_frame(self, audio_chunk: np.ndarray)
```

### Key Methods

- `initialize_models()`: Load MuseTalk models
- `initialize_avatars()`: Load and prepare avatar materials
- `process_frame()`: Process incoming audio/video frames
- `generate_video_frame()`: Generate avatar video from audio

## Comparison with Custom WebRTC

| Feature | Custom WebRTC | Pipecat + Daily.co |
|---------|---------------|-------------------|
| **Setup Complexity** | High | Low |
| **WebRTC Handling** | Manual | Automatic |
| **Scalability** | Limited | High |
| **Reliability** | Variable | Production-ready |
| **Maintenance** | High | Low |
| **Features** | Basic | Rich (recording, analytics, etc.) |

## Support and Resources

- **Pipecat Documentation**: [pipecat.ai](https://pipecat.ai)
- **Daily.co Documentation**: [docs.daily.co](https://docs.daily.co)
- **MuseTalk Issues**: Check server logs and error messages
- **Community**: Daily.co and Pipecat community forums

## License

This Pipecat integration follows the same license as the main MuseTalk project.