// Import the RTMS SDK
// Import Node.js modules for file operations
import fs from "fs";
import path from "path";
import { spawn } from 'node:child_process';
import WebSocket from 'ws';
import crypto from 'crypto';

// Parse command line arguments
const parseArgs = () => {
  const args = process.argv.slice(2);
  const result = {
    recording_file_path: null,
    join_payload: null,
  };
  
  for (const arg of args) {
    if (arg.startsWith('--recording_file_path=')) {
      result.recording_file_path = arg.substring('--recording_file_path='.length);
    } else if (arg.startsWith('--join_payload=')) {
      const payloadStr = arg.substring('--join_payload='.length);
      try {
        // Decode from base64 first
        const decodedStr = Buffer.from(payloadStr, 'base64').toString('utf-8');
        result.join_payload = JSON.parse(decodedStr);
      } catch (e) {
        console.error('Failed to parse join_payload as JSON:', e);
        console.error('Raw payload string:', payloadStr);
      }
    }
  }
  
  return result;
};

// Get the arguments
const { recording_file_path, join_payload } = parseArgs();
console.log('Recording will be saved to:', recording_file_path);
console.log('Join payload:', join_payload);

let videoSink = null;
let audioSink = null;
let firstVideoFrameReceived = false;

const audioFd = process.env.AUDIO_FD ? Number(process.env.AUDIO_FD) : null;
const videoFd = process.env.VIDEO_FD ? Number(process.env.VIDEO_FD) : null;

let logVideoTicker = 0;
let logAudioTicker = 0;

if (videoFd != null) {
  videoSink = fs.createWriteStream(null, { fd: videoFd, autoClose: false });
  console.log(`Opened video FD: ${videoFd}`);
}

if (audioFd != null) {
  audioSink = fs.createWriteStream(null, { fd: audioFd, autoClose: false });
  console.log(`Opened audio FD: ${audioFd}`);
}


const CLIENT_ID = process.env.ZM_RTMS_CLIENT;
const CLIENT_SECRET = process.env.ZM_RTMS_SECRET;

function generateSignature(CLIENT_ID, meetingUuid, streamId, CLIENT_SECRET) {
  console.log('Generating signature with parameters:');
  console.log('meetingUuid:', meetingUuid);
  console.log('streamId:', streamId);

  // Create a message string and generate an HMAC SHA256 signature
  const message = `${CLIENT_ID},${meetingUuid},${streamId}`;
  return crypto.createHmac('sha256', CLIENT_SECRET).update(message).digest('hex');
}

// Set up webhook event handler to receive RTMS events from Zoom
// Map to keep track of active WebSocket connections and audio chunks
const activeConnections = new Map();
const audioChunks = new Map();

// Function to connect to the media WebSocket server
function connectToMediaWebSocket(mediaUrl, meetingUuid, streamId, signalingSocket) {
  console.log(`Connecting to media WebSocket at ${mediaUrl}`);

  const mediaWs = new WebSocket(mediaUrl, { rejectUnauthorized: false });

  // Store connection for cleanup later
  if (activeConnections.has(meetingUuid)) {
      activeConnections.get(meetingUuid).media = mediaWs;
  }

  // Initialize audio chunks array for this meeting
  audioChunks.set(meetingUuid, []);

  mediaWs.on('open', () => {
      const signature = generateSignature(
          CLIENT_ID,
          meetingUuid,
          streamId,
          CLIENT_SECRET
      );
      const handshake = {
          msg_type: 3, // DATA_HAND_SHAKE_REQ
          protocol_version: 1,
          meeting_uuid: meetingUuid,
          rtms_stream_id: streamId,
          signature,
          media_type: 1, // MEDIA_DATA_AUDIO
          payload_encryption: false,
      };
      mediaWs.send(JSON.stringify(handshake));
  });

  mediaWs.on('message', (data) => {
      try {
          // Try to parse as JSON first
          const msg = JSON.parse(data.toString());
          //console.log('Media JSON Message:', JSON.stringify(msg, null, 2));

          // Handle successful media handshake
          if (msg.msg_type === 4 && msg.status_code === 0) { // DATA_HAND_SHAKE_RESP
              signalingSocket.send(
                  JSON.stringify({
                      msg_type: 7, // CLIENT_READY_ACK
                      rtms_stream_id: streamId,
                  })
              );
              console.log('Media handshake successful, sent start streaming request');
          }

          // Respond to keep-alive requests
          if (msg.msg_type === 12) { // KEEP_ALIVE_REQ
              mediaWs.send(
                  JSON.stringify({
                      msg_type: 13, // KEEP_ALIVE_RESP
                      timestamp: msg.timestamp,
                  })
              );
              console.log('Responded to Media KEEP_ALIVE_REQ');
          }

          // Handle audio data
          if (msg.msg_type === 14 && msg.content && msg.content.data) {
              // Decode base64 audio data
              const audioData = Buffer.from(msg.content.data, 'base64');
              const chunks = audioChunks.get(meetingUuid);
              if (chunks) {
                  chunks.push(audioData);
                  console.log('Received audio chunk, total chunks:', chunks.length);
              }
          }
      } catch (err) {
          console.error('Error processing media message:', err);
      }
  });

  mediaWs.on('error', (err) => {
      console.error('Media socket error:', err);
  });

  mediaWs.on('close', () => {
      console.log('Media socket closed');
      if (activeConnections.has(meetingUuid)) {
          delete activeConnections.get(meetingUuid).media;
      }
  });
}

function connectToSignalingWebSocket(meetingUuid, streamId, serverUrl) {
  console.error(`Connecting to signaling WebSocket for meeting ${meetingUuid} and stream id ${streamId} and server url ${serverUrl}`);

  const ws = new WebSocket(serverUrl);

  // Store connection for cleanup later
  if (!activeConnections.has(meetingUuid)) {
      activeConnections.set(meetingUuid, {});
  }
  activeConnections.get(meetingUuid).signaling = ws;

  
  ws.on('open', () => {
      console.log(`Signaling WebSocket connection opened for meeting ${meetingUuid}`);
      const signature = generateSignature(
          CLIENT_ID,
          meetingUuid,
          streamId,
          CLIENT_SECRET
      );

      // Send handshake message to the signaling server
      const handshake = {
          msg_type: 1, // SIGNALING_HAND_SHAKE_REQ
          protocol_version: 1,
          meeting_uuid: meetingUuid,
          rtms_stream_id: streamId,
          sequence: Math.floor(Math.random() * 1e9),
          signature,
      };
      ws.send(JSON.stringify(handshake));
      console.log('Sent handshake to signaling server');
  });

  ws.on('message', (data) => {
      const msg = JSON.parse(data);
      console.log('Signaling Message:', JSON.stringify(msg, null, 2));

      // Handle successful handshake response
      if (msg.msg_type === 2 && msg.status_code === 0) { // SIGNALING_HAND_SHAKE_RESP
          const mediaUrl = msg.media_server?.server_urls?.all;
          if (mediaUrl) {
              // Connect to the media WebSocket server using the media URL
              connectToMediaWebSocket(mediaUrl, meetingUuid, streamId, ws);
          }
      }

      // Respond to keep-alive requests
      if (msg.msg_type === 12) { // KEEP_ALIVE_REQ
          const keepAliveResponse = {
              msg_type: 13, // KEEP_ALIVE_RESP
              timestamp: msg.timestamp,
          };
          console.log('Responding to Signaling KEEP_ALIVE_REQ:', keepAliveResponse);
          ws.send(JSON.stringify(keepAliveResponse));
      }
  });

  ws.on('error', (err) => {
      console.error('Signaling socket error:', err);
  });

  ws.on('close', () => {
      console.log('Signaling socket closed');
      if (activeConnections.has(meetingUuid)) {
          delete activeConnections.get(meetingUuid).signaling;
      }
  });
}

console.log('Connecting to signaling WebSocket for meeting', join_payload.meeting_uuid, ' and stream id', join_payload.rtms_stream_id, ' and server urls', join_payload.server_urls);

connectToSignalingWebSocket(join_payload.meeting_uuid, join_payload.rtms_stream_id, join_payload.server_urls);


// Set up command handling via stdin
process.stdin.setEncoding('utf8');
process.stdin.on('data', (data) => {
  const command = data.toString().trim();
  console.log(`Received command: ${command}`);
  
  // Process the command
  if (command === 'leave') {
    if (videoSink) {
      videoSink.end();
      console.log("Closed video pipe");
    }
    if (audioSink) {
      audioSink.end();
      console.log("Closed audio pipe");
    }
    console.log("Meeting ended, closing pipes...");
    process.exit(0); // Exit Node.js process with success code
  } else {
    console.log('Unknown command:', command);
  }
});