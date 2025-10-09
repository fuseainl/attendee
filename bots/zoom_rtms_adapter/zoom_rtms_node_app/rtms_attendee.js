// Import the RTMS SDK
import rtms from "@zoom/rtms";
// Import Node.js modules for file operations
import fs from "fs";
import path from "path";
import { spawn } from 'node:child_process';

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

// Set up webhook event handler to receive RTMS events from Zoom


// Create a client instance for this specific meeting
const client = new rtms.Client();

rtms.onWebhookEvent(({ event, payload }) => {
  console.log(`Received webhook event: ${event}`);

});
client.setVideoParams({
  contentType: rtms.VideoContentType.RAW_VIDEO,
  codec: rtms.VideoCodec.H264,
  resolution: rtms.VideoResolution.HD,
  dataOpt: rtms.VideoDataOption.VIDEO_MIXED_GALLERY_VIEW,
  fps: 15
});


client.onLeave((reason) => {
  console.log(`Client left the meeting with reason: ${reason}`);
});

client.onSessionUpdate((op, sessionInfo) => {
  console.log(`Session updated: ${op} ${JSON.stringify(sessionInfo)}`);
  const sessionUpdateData = {
    op: op,
    sessionInfo: sessionInfo,
    type: "sessionUpdate"
  }
  process.stdout.write("rtmsdata." + JSON.stringify(sessionUpdateData) + '\n' + "*".repeat(10000) + '\n');

});

client.onUserUpdate((op, user) => {
  const userUpdateData = {
    op: op,
    user: user,
    type: "userUpdate"
  }
  process.stdout.write("rtmsdata." + JSON.stringify(userUpdateData) + '\n');
});

client.onTranscriptData((data, timestamp, metadata, user) => {
  const convertedData = data.toString('utf-8');
  
  const transcriptUpdateData = {
    user: user,
    text: convertedData,
    type: "transcriptUpdate",
    id: metadata
  }
  process.stdout.write("rtmsdata." + JSON.stringify(transcriptUpdateData) + '\n');
});

client.onVideoData((data, size, timestamp, metadata) => {
  logVideoTicker++;
  if (logVideoTicker % 100 == 0) {
    // This userName is NOT correct
    console.log(`Video data: ${size} bytes from ${JSON.stringify(metadata)}`);
  }
  if (videoSink) {
    // Create a buffer with username, userId, then int32 length prefix, followed by the video data
    const userName = metadata.userName || '';
    const userId = metadata.userId || -1;
    const userNameBuffer = Buffer.from(userName, 'utf8');
    const userNameLengthBuffer = Buffer.allocUnsafe(4);
    userNameLengthBuffer.writeInt32LE(userNameBuffer.length, 0);
    
    const userIdBuffer = Buffer.allocUnsafe(4);
    userIdBuffer.writeInt32LE(userId, 0);
    
    const lengthBuffer = Buffer.allocUnsafe(4);
    lengthBuffer.writeInt32LE(size, 0);
    
    const frameWithMetadata = Buffer.concat([userNameLengthBuffer, userNameBuffer, userIdBuffer, lengthBuffer, data]);
    
    const writeResult = videoSink.write(frameWithMetadata);
    if (logVideoTicker % 100 == 0) {
      console.log(`Write video result: ${writeResult}`);
    }
  } else {
    console.log('No video sink available');
  }

  if (!firstVideoFrameReceived) {
    firstVideoFrameReceived = true;
    const videoUpdateData = {
      type: "firstVideoFrameReceived",
    }
    process.stdout.write("rtmsdata." + JSON.stringify(videoUpdateData) + '\n');
  }

  /*
  // Write video data to file
  const fileName = `video-${timestamp}.raw`;
  const filePath = path.join(outputDir, fileName);
  fs.writeFileSync(filePath, data);
  
  // Write metadata to a separate file without pretty printing
  const metadataFileName = `video-${timestamp}.metadata`;
  const metadataFilePath = path.join(outputDir, metadataFileName);
  fs.writeFileSync(metadataFilePath, JSON.stringify(metadata));
  
  console.log(`Saved video to ${filePath} and metadata to ${metadataFilePath}`);*/
});
  
client.setAudioParams({
  contentType: rtms.AudioContentType.RAW_AUDIO,
  codec: rtms.AudioCodec.OPUS,
  sampleRate: 16000,
  channel: rtms.AudioChannel.MONO,
  dataOpt: rtms.AudioDataOption.AUDIO_MIXED_STREAM,
  duration: 20,
  frameSize: 320
});

// Set up audio data handler
client.onAudioData((data, size, timestamp, metadata) => {
  logAudioTicker++;
  if (logAudioTicker % 100 == 0) {
    // This userName is correct
    console.log(`Audio data: ${size} bytes from ${metadata.userName}`);
  }

  if (audioSink) {
    // Create a buffer with username, userId, then int32 length prefix, followed by the audio data
    const userName = metadata.userName || '';
    const userId = metadata.userId || -1;
    const userNameBuffer = Buffer.from(userName, 'utf8');
    const userNameLengthBuffer = Buffer.allocUnsafe(4);
    userNameLengthBuffer.writeInt32LE(userNameBuffer.length, 0);
    
    const userIdBuffer = Buffer.allocUnsafe(4);
    userIdBuffer.writeInt32LE(userId, 0);
    
    const lengthBuffer = Buffer.allocUnsafe(4);
    lengthBuffer.writeInt32LE(size, 0);
    
    const frameWithMetadata = Buffer.concat([userNameLengthBuffer, userNameBuffer, userIdBuffer, lengthBuffer, data]);
    
    const writeResult = audioSink.write(frameWithMetadata);
    if (logAudioTicker % 100 == 0) {
      console.log(`Write audio result: ${writeResult}`);
    }
  } else {
    console.log('No audio sink available');
  }

  /*
  // Write audio data to file
  const fileName = `audio-${timestamp}.raw`;
  const filePath = path.join(outputDir, fileName);
  fs.writeFileSync(filePath, data);
  
  // Write metadata to a separate file without pretty printing
  const metadataFileName = `audio-${timestamp}.metadata`;
  const metadataFilePath = path.join(outputDir, metadataFileName);
  fs.writeFileSync(metadataFilePath, JSON.stringify(metadata));
  
  console.log(`Saved audio to ${filePath} and metadata to ${metadataFilePath}`);
  */
}); 

console.log('join_payload', join_payload);

// Join the meeting using the webhook payload directly
let joinResult = client.join(join_payload);
console.log(`Join result: ${joinResult}`);

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