import React, { useEffect, useRef, useState } from 'react';
import { RTVIClient, RTVIEvent } from '@pipecat-ai/client-js';
import { DailyTransport } from '@pipecat-ai/daily-transport';

const JobTalkMuseTalkIntegration = () => {
  // Pipecat/JobTalk connection states
  const [isConnecting, setIsConnecting] = useState(false);
  const [callActive, setCallActive] = useState(false);
  const [callStatus, setCallStatus] = useState('Ready to start interview');
  
  // MuseTalk avatar states
  const [avatarConnected, setAvatarConnected] = useState(false);
  const [audioEnergy, setAudioEnergy] = useState(0);
  const [mouthState, setMouthState] = useState('closed');
  
  // Refs for Pipecat
  const rtviClientRef = useRef(null);
  const botAudioRef = useRef(null);
  const audioContextRef = useRef(null);
  const analyserRef = useRef(null);
  const animationFrameRef = useRef(null);
  const isConnectingRef = useRef(false);
  
  // Refs for MuseTalk
  const canvasRef = useRef(null);
  const websocketRef = useRef(null);
  
  // Constants
  const TOKEN = 'g90lzsyz';
  const BASE_URL = 'https://aptask.jobtalk.ai/v1';
  const CONNECTION_COOLDOWN_MS = 3000;
  const lastConnectionAttemptTimeRef = useRef(0);

  const log = (message) => {
    console.log(`[JobTalk-MuseTalk]: ${message}`);
  };

  // Initialize MuseTalk connection
  const connectToMuseTalk = async () => {
    try {
      log('Connecting to MuseTalk server...');
      
      // Connect to MuseTalk WebSocket server (adjust URL as needed)
      const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
      const wsUrl = `${wsProtocol}//localhost:8080/ws`; // Your MuseTalk server URL
      
      websocketRef.current = new WebSocket(wsUrl);
      
      websocketRef.current.onopen = () => {
        log('MuseTalk WebSocket connected');
        setAvatarConnected(true);
        
        // Start video stream
        websocketRef.current.send(JSON.stringify({ type: 'start_video' }));
      };
      
      websocketRef.current.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          
          if (data.type === 'video_frame') {
            displayVideoFrame(data.frame);
          }
        } catch (err) {
          console.error('Error processing MuseTalk message:', err);
        }
      };
      
      websocketRef.current.onclose = () => {
        log('MuseTalk WebSocket disconnected');
        setAvatarConnected(false);
      };
      
      websocketRef.current.onerror = (error) => {
        console.error('MuseTalk WebSocket error:', error);
        setAvatarConnected(false);
      };
      
    } catch (err) {
      console.error('Error connecting to MuseTalk:', err);
      setAvatarConnected(false);
    }
  };

  const disconnectFromMuseTalk = () => {
    if (websocketRef.current) {
      websocketRef.current.close();
      websocketRef.current = null;
    }
    setAvatarConnected(false);
  };

  const displayVideoFrame = (frameData) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    const img = new Image();
    
    img.onload = () => {
      ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
    };
    
    img.src = 'data:image/jpeg;base64,' + frameData;
  };

  const sendAudioToMuseTalk = (energy) => {
    if (websocketRef.current && websocketRef.current.readyState === WebSocket.OPEN) {
      // Convert energy to audio data simulation
      const audioDataSize = Math.floor(energy * 1000);
      const audioData = new Array(audioDataSize).fill(128);
      
      websocketRef.current.send(JSON.stringify({
        type: 'audio_data',
        data: audioData
      }));
      
      setAudioEnergy(energy);
      setMouthState(energy > 0.5 ? 'open' : energy > 0.2 ? 'half_open' : 'closed');
    }
  };

  // Setup audio analysis for Pipecat audio responses
  const setupAudioTrack = (track) => {
    if (botAudioRef.current) {
      if (botAudioRef.current.srcObject) {
        const oldTrack = (botAudioRef.current.srcObject).getAudioTracks()[0];
        if (oldTrack?.id === track.id) return;
      }
      botAudioRef.current.srcObject = new MediaStream([track]);

      if (!audioContextRef.current) {
        try {
          audioContextRef.current = new AudioContext();
          const audioContext = audioContextRef.current;
          const source = audioContext.createMediaStreamSource(new MediaStream([track]));
          analyserRef.current = audioContext.createAnalyser();
          const analyser = analyserRef.current;

          analyser.fftSize = 256;
          analyser.smoothingTimeConstant = 0.8;
          source.connect(analyser);

          const dataArray = new Uint8Array(analyser.frequencyBinCount);

          const checkAudioEnergy = () => {
            if (!analyser) return;

            analyser.getByteFrequencyData(dataArray);

            let sum = 0;
            const midStart = Math.floor(dataArray.length * 0.1);
            const midEnd = Math.floor(dataArray.length * 0.7);

            for (let i = midStart; i < midEnd; i++) {
              sum += dataArray[i];
            }

            const energy = sum / (midEnd - midStart) / 255;

            // Send audio energy to MuseTalk for lip-sync
            if (energy > 0.03) {
              sendAudioToMuseTalk(energy * 1.5);
            } else {
              sendAudioToMuseTalk(0);
            }

            animationFrameRef.current = requestAnimationFrame(checkAudioEnergy);
          };

          checkAudioEnergy();
          log('Audio analysis started for MuseTalk lip-sync');
        } catch (error) {
          log(`Error setting up audio analysis: ${error instanceof Error ? error.message : String(error)}`);
        }
      }
    }
  };

  // Start Pipecat connection to JobTalk
  const startConnection = async () => {
    if (isConnectingRef.current) {
      log('Connection already in progress, ignoring duplicate request');
      return;
    }

    const now = Date.now();
    const timeSinceLastAttempt = now - lastConnectionAttemptTimeRef.current;
    if (timeSinceLastAttempt < CONNECTION_COOLDOWN_MS && lastConnectionAttemptTimeRef.current > 0) {
      log(`Connection attempt too soon. Waiting ${CONNECTION_COOLDOWN_MS - timeSinceLastAttempt}ms.`);
      return;
    }

    isConnectingRef.current = true;
    lastConnectionAttemptTimeRef.current = now;

    try {
      setIsConnecting(true);
      setCallStatus('Connecting to JobTalk interview...');
      log('Starting connection to JobTalk');

      const transport = new DailyTransport();

      if (rtviClientRef.current) {
        log('Cleaning up existing client');
        await rtviClientRef.current.disconnect();
        rtviClientRef.current = null;
      }

      const clientParams = {
        baseUrl: BASE_URL,
        endpoints: {
          connect: `/webrtc-screening/start-interview/${TOKEN}`,
        },
        metadata: {
          token: TOKEN
        }
      };

      log('Client params: ' + JSON.stringify(clientParams, null, 2));

      rtviClientRef.current = new RTVIClient({
        transport,
        params: clientParams,
        enableMic: true,
        enableCam: false,
        callbacks: {
          onConnected: () => {
            log('JobTalk client connected');
            setCallStatus('Connected - establishing audio...');
          },
          onDisconnected: () => {
            log('JobTalk client disconnected');
            setCallStatus('Disconnected');
            setIsConnecting(false);
            setCallActive(false);
          },
          onTransportStateChanged: (state) => {
            log(`Transport state: ${state}`);
            if (state === 'ready') {
              const tracks = rtviClientRef.current?.tracks();
              if (tracks?.bot?.audio) {
                setupAudioTrack(tracks.bot.audio);
              }
            }
          },
          onBotConnected: (participant) => {
            log(`Bot connected: ${JSON.stringify(participant)}`);
            setCallStatus('AI Interviewer connected');
          },
          onBotDisconnected: (participant) => {
            log(`Bot disconnected: ${JSON.stringify(participant)}`);
            setCallStatus('AI Interviewer disconnected');
          },
          onBotReady: (data) => {
            log(`Bot ready: ${JSON.stringify(data)}`);
            setTimeout(() => {
              const tracks = rtviClientRef.current?.tracks();
              setIsConnecting(false);
              setCallActive(true);
              setCallStatus('Interview in progress');
              if (tracks?.bot?.audio) {
                setupAudioTrack(tracks.bot.audio);
              }
            }, 500);
          },
          onUserTranscript: (data) => {
            if (data.final) {
              log(`You: ${data.text}`);
            }
          },
          onBotTranscript: (data) => {
            log(`AI Interviewer: ${data.text}`);
          },
          onMessageError: (error) => {
            log(`Message error: ${error}`);
            setCallStatus('Error during interview');
          },
          onError: (error) => {
            log(`Error: ${JSON.stringify(error, null, 2)}`);
            setIsConnecting(false);
            setCallStatus('Connection error');
          }
        }
      });

      rtviClientRef.current.on(RTVIEvent.TrackStarted, (track, participant) => {
        if (!participant?.local) {
          if (track.kind === 'audio') {
            setupAudioTrack(track);
          }
        }
      });

      log('Initializing devices...');
      setCallStatus('Requesting microphone access...');
      await rtviClientRef.current.initDevices();

      log('Connecting to JobTalk bot...');
      await rtviClientRef.current.connect();

      log('JobTalk connection complete');
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : String(error);
      log(`Connection error: ${errorMessage}`);
      setIsConnecting(false);
      setCallStatus('Connection failed. Please try again.');

      if (rtviClientRef.current) {
        try {
          await rtviClientRef.current.disconnect();
          rtviClientRef.current = null;
        } catch (disconnectError) {
          log(`Error during disconnect: ${disconnectError instanceof Error ? disconnectError.message : String(disconnectError)}`);
        }
      }
    } finally {
      isConnectingRef.current = false;
    }
  };

  const endInterview = async () => {
    try {
      log('Ending interview...');
      setCallStatus('Ending interview...');

      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
        animationFrameRef.current = null;
      }

      if (audioContextRef.current && audioContextRef.current.state !== 'closed') {
        try {
          await audioContextRef.current.close();
        } catch (e) {
          log(`Error closing audio context: ${e instanceof Error ? e.message : String(e)}`);
        }
        audioContextRef.current = null;
      }

      if (rtviClientRef.current) {
        await rtviClientRef.current.disconnect();
        rtviClientRef.current = null;
      }

      if (botAudioRef.current && botAudioRef.current.srcObject) {
        const tracks = (botAudioRef.current.srcObject).getTracks();
        tracks.forEach(track => track.stop());
        botAudioRef.current.srcObject = null;
      }

      disconnectFromMuseTalk();

      log('Interview ended');
      setCallActive(false);
      setIsConnecting(false);
      setCallStatus('Interview ended');
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : String(error);
      log(`End interview error: ${errorMessage}`);
      setCallStatus('Error ending interview');
    }
  };

  // Initialize components
  useEffect(() => {
    // Create audio element for bot audio
    botAudioRef.current = document.createElement('audio');
    botAudioRef.current.autoplay = true;
    document.body.appendChild(botAudioRef.current);

    // Connect to MuseTalk
    connectToMuseTalk();

    return () => {
      // Cleanup
      if (botAudioRef.current) {
        if (botAudioRef.current.srcObject) {
          const tracks = (botAudioRef.current.srcObject).getTracks();
          tracks.forEach(track => track.stop());
          botAudioRef.current.srcObject = null;
        }
        document.body.removeChild(botAudioRef.current);
        botAudioRef.current = null;
      }

      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
        animationFrameRef.current = null;
      }

      if (audioContextRef.current && audioContextRef.current.state !== 'closed') {
        audioContextRef.current.close();
        audioContextRef.current = null;
      }

      if (rtviClientRef.current) {
        rtviClientRef.current.disconnect();
        rtviClientRef.current = null;
      }

      disconnectFromMuseTalk();
    };
  }, []);

  return (
    <div className="min-h-screen bg-gray-50 flex items-center justify-center p-4">
      <div className="bg-white rounded-2xl shadow-xl p-8 max-w-2xl w-full">
        <div className="text-center mb-8">
          <h1 className="text-3xl font-bold text-gray-900 mb-2">
            JobTalk AI Interview
          </h1>
          <p className="text-gray-600">
            AI-powered interview with real-time avatar responses
          </p>
        </div>

        <div className="grid md:grid-cols-2 gap-8">
          {/* Avatar Section */}
          <div className="text-center">
            <h3 className="text-lg font-semibold mb-4">AI Interviewer Avatar</h3>
            
            <div className="relative mb-4">
              <canvas
                ref={canvasRef}
                width={300}
                height={300}
                className="w-full max-w-sm mx-auto rounded-lg border-2 border-gray-200 bg-gray-100"
              />
              
              {!avatarConnected && (
                <div className="absolute inset-0 flex items-center justify-center bg-gray-100 bg-opacity-90 rounded-lg">
                  <div className="text-center">
                    <div className="w-8 h-8 border-t-2 border-b-2 border-blue-500 rounded-full animate-spin mx-auto mb-2"></div>
                    <p className="text-sm text-gray-600">Loading Avatar...</p>
                  </div>
                </div>
              )}
            </div>

            {avatarConnected && (
              <div className="space-y-2">
                <div className="flex items-center justify-between text-sm text-gray-600">
                  <span>Speaking:</span>
                  <span className="capitalize font-medium">{mouthState}</span>
                </div>
                
                <div className="w-full bg-gray-200 rounded-full h-2">
                  <div
                    className="bg-blue-500 h-2 rounded-full transition-all duration-200"
                    style={{ width: `${audioEnergy * 100}%` }}
                  ></div>
                </div>
              </div>
            )}
          </div>

          {/* Interview Controls */}
          <div className="space-y-6">
            <div className="text-center">
              <div className="mb-4">
                <div className={`inline-flex items-center px-4 py-2 rounded-full text-sm font-medium ${
                  callActive ? 'bg-green-100 text-green-800' : 
                  isConnecting ? 'bg-yellow-100 text-yellow-800' : 
                  'bg-gray-100 text-gray-800'
                }`}>
                  {callActive ? '🟢 Interview Active' : 
                   isConnecting ? '🟡 Connecting...' : 
                   '⚪ Ready to Start'}
                </div>
              </div>

              <div className="mb-6">
                <p className="text-sm text-gray-600 bg-gray-50 p-3 rounded-lg">
                  {callStatus}
                </p>
              </div>

              <div className="space-y-4">
                {!callActive && !isConnecting && (
                  <button
                    onClick={startConnection}
                    className="w-full py-3 px-6 bg-blue-600 hover:bg-blue-700 text-white font-medium rounded-lg transition-colors"
                  >
                    Start Interview
                  </button>
                )}

                {(callActive || isConnecting) && (
                  <button
                    onClick={endInterview}
                    className="w-full py-3 px-6 bg-red-600 hover:bg-red-700 text-white font-medium rounded-lg transition-colors"
                  >
                    End Interview
                  </button>
                )}
              </div>
            </div>

            <div className="bg-blue-50 p-4 rounded-lg">
              <h4 className="font-medium text-blue-900 mb-2">How it works:</h4>
              <ul className="text-sm text-blue-800 space-y-1">
                <li>• Connects to JobTalk AI interviewer</li>
                <li>• Avatar lip-syncs with AI responses</li>
                <li>• Real-time audio analysis</li>
                <li>• Natural conversation flow</li>
              </ul>
            </div>
          </div>
        </div>

        <div className="mt-8 text-center text-xs text-gray-500">
          <p>
            Using token: {TOKEN} | 
            Base URL: {BASE_URL} | 
            Avatar: {avatarConnected ? 'Connected' : 'Disconnected'}
          </p>
        </div>
      </div>
    </div>
  );
};

export default JobTalkMuseTalkIntegration;