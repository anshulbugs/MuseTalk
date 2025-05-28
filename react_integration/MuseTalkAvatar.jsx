import React, { useEffect, useRef, useState } from 'react';

const MuseTalkAvatar = ({ isActive, onAudioData }) => {
  const canvasRef = useRef(null);
  const websocketRef = useRef(null);
  const [connectionStatus, setConnectionStatus] = useState('disconnected');
  const [audioEnergy, setAudioEnergy] = useState(0);
  const [mouthState, setMouthState] = useState('closed');

  useEffect(() => {
    if (isActive) {
      connectToMuseTalk();
    } else {
      disconnectFromMuseTalk();
    }

    return () => {
      disconnectFromMuseTalk();
    };
  }, [isActive]);

  // Listen for audio playback events from the main interview component
  useEffect(() => {
    const handleAudioPlaybackStart = (event) => {
      const energy = event.detail?.energy || 0.5;
      sendAudioToMuseTalk(energy);
    };

    const handleAudioPlaybackEnd = () => {
      sendAudioToMuseTalk(0);
    };

    window.addEventListener('audioPlaybackStart', handleAudioPlaybackStart);
    window.addEventListener('audioPlaybackEnd', handleAudioPlaybackEnd);

    return () => {
      window.removeEventListener('audioPlaybackStart', handleAudioPlaybackStart);
      window.removeEventListener('audioPlaybackEnd', handleAudioPlaybackEnd);
    };
  }, []);

  const connectToMuseTalk = async () => {
    try {
      setConnectionStatus('connecting');
      
      // Connect to MuseTalk WebSocket server
      const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
      const wsUrl = `${wsProtocol}//localhost:8080/ws`; // Adjust this to your MuseTalk server URL
      
      websocketRef.current = new WebSocket(wsUrl);
      
      websocketRef.current.onopen = () => {
        console.log('MuseTalk WebSocket connected');
        setConnectionStatus('connected');
        
        // Start video stream
        websocketRef.current.send(JSON.stringify({ type: 'start_video' }));
      };
      
      websocketRef.current.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          
          if (data.type === 'video_frame') {
            displayVideoFrame(data.frame);
          } else if (data.type === 'status') {
            console.log('MuseTalk status:', data.message);
          }
        } catch (err) {
          console.error('Error processing MuseTalk message:', err);
        }
      };
      
      websocketRef.current.onclose = () => {
        console.log('MuseTalk WebSocket disconnected');
        setConnectionStatus('disconnected');
      };
      
      websocketRef.current.onerror = (error) => {
        console.error('MuseTalk WebSocket error:', error);
        setConnectionStatus('error');
      };
      
    } catch (err) {
      console.error('Error connecting to MuseTalk:', err);
      setConnectionStatus('error');
    }
  };

  const disconnectFromMuseTalk = () => {
