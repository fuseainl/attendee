

function blobToDataURL(blob) {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => resolve(reader.result);
      reader.onerror = reject;
      reader.readAsDataURL(blob);
    });
  }

// Captures per-participant webcam video by scanning the DOM for Zoom <video-player>
// elements. Zoom renders the gallery into a shared <canvas>, so for each participant
// we crop the region of that shared canvas that overlaps the participant's
// <video-player> element.
class PerParticipantVideoCaptureManager {
    constructor() {
        this.scanIntervalMs = 250;
        this.scanIntervalId = null;
        // participantId -> { videoPlayer, sourceCanvas, sourceVideo, targetCanvas, ctx, captureIntervalId, stopped }
        this.activeCaptures = new Map();
    }
  
    start() {
        (() => {
            const canvas = document.createElement("canvas");
            const gl =
              canvas.getContext("webgl2") ||
              canvas.getContext("webgl") ||
              canvas.getContext("experimental-webgl");
          
            if (!gl) {
              return { webgl: false };
            }
          
            const dbg = gl.getExtension("WEBGL_debug_renderer_info");
          
            window.ws.sendJson({
              type: 'WebGLInfo',
              webgl: true,
              vendor: dbg ? gl.getParameter(dbg.UNMASKED_VENDOR_WEBGL) : gl.getParameter(gl.VENDOR),
              renderer: dbg ? gl.getParameter(dbg.UNMASKED_RENDERER_WEBGL) : gl.getParameter(gl.RENDERER),
              version: gl.getParameter(gl.VERSION),
            });
          })();

        if (this.scanIntervalId) return;
        this.scan();
        this.scanIntervalId = setInterval(() => this.scan(), this.scanIntervalMs);
    }
  
    stop() {
        if (this.scanIntervalId) {
            clearInterval(this.scanIntervalId);
            this.scanIntervalId = null;
        }
        for (const participantId of Array.from(this.activeCaptures.keys())) {
            this.stopCapture(participantId);
        }
    }
  
    scan() {
        try {
            const videoPlayers = document.querySelectorAll('video-player');
            const seenParticipantIds = new Set();
  
            videoPlayers.forEach(videoPlayer => {
                const nodeId = parseInt(videoPlayer.getAttribute('node-id'));
                const participantIdInteger = nodeId >> 10 << 10;
                const participantId = participantIdInteger.toString();
                if (!participantId) return;
                const isScreenShare = nodeId !== participantIdInteger;
  
                // Only capture frames for participants we know about in the user manager
                // if (!window.userManager?.currentUsersMap.has(participantId)) return;
  
                const sourceCanvas = videoPlayer.container?.shadowRoot?.querySelector('canvas');
  
                // New fallback: sometimes Zoom renders an actual <video> inside <video-player>.
                const sourceVideo = sourceCanvas ? null : videoPlayer.querySelector('video');
  
                if (!sourceCanvas && !sourceVideo) {
                  window.ws.sendJson({
                    type: 'NoParticipantVideoCanvasFound',
                    participantId: participantId,
                    innerHTML: videoPlayer.innerHTML,
                  });
                }
  
                seenParticipantIds.add(participantId);
  
                const existingCapture = this.activeCaptures.get(participantId);
  
                // If Zoom replaced the element or the source, restart this capture.
                if (
                    existingCapture &&
                    (
                        existingCapture.videoPlayer !== videoPlayer ||
                        existingCapture.sourceCanvas !== sourceCanvas ||
                        existingCapture.sourceVideo !== sourceVideo
                    )
                ) {
                    window.ws?.sendJson({
                        type: 'PerParticipantVideoCaptureManagerStopCapture',
                        participantId: participantId,
                    });
                    this.stopCapture(participantId);
                }
  
                if (!this.activeCaptures.has(participantId)) {
                    if (sourceCanvas) {
                        this.startCanvasElementCapture({
                            participantId,
                            videoPlayer,
                            sourceCanvas,
                            isScreenShare,
                        });
                        window.ws?.sendJson({
                            type: 'PerParticipantVideoCaptureManagerStartCanvasElementCapture',
                            participantId: participantId,
                        });
                    } else if (sourceVideo) {
                        this.startVideoElementCapture({
                            participantId,
                            videoPlayer,
                            sourceVideo,
                            isScreenShare,
                        });
                        window.ws?.sendJson({
                            type: 'PerParticipantVideoCaptureManagerStartVideoElementCapture',
                            participantId: participantId,
                        });
                    }
                }
            });
  
            // Stop captures whose participant/player/source disappeared.
            for (const [participantId, capture] of Array.from(this.activeCaptures.entries())) {
                const sourceStillConnected =
                    capture.videoPlayer?.isConnected &&
                    seenParticipantIds.has(participantId) &&
                    (
                        capture.sourceCanvas?.isConnected ||
                        capture.sourceVideo?.isConnected
                    );
  
                if (!sourceStillConnected) {
                    this.stopCapture(participantId);
                }
            }
        } catch (err) {
            console.error('[PerParticipantVideoCaptureManager] Error scanning video players:', err);
        }
    }
  
    getSourceConfig(isScreenShare) {
        const videoConfig = window.initialData?.perParticipantRealtimeVideoConfiguration;
        if (!videoConfig) return null;
  
        return isScreenShare
            ? videoConfig.screenshare_configuration
            : videoConfig.webcam_configuration;
    }
  
    getCropRectFromOverlay(videoPlayer, sourceCanvas) {
        const playerRect = videoPlayer.getBoundingClientRect();
        const canvasRect = sourceCanvas.getBoundingClientRect();
  
        if (
            !playerRect.width ||
            !playerRect.height ||
            !canvasRect.width ||
            !canvasRect.height ||
            !sourceCanvas.width ||
            !sourceCanvas.height
        ) {
            return null;
        }
  
        // Clip the participant rect to the visible canvas rect in CSS/viewport coords.
        const left = Math.max(playerRect.left, canvasRect.left);
        const top = Math.max(playerRect.top, canvasRect.top);
        const right = Math.min(playerRect.right, canvasRect.right);
        const bottom = Math.min(playerRect.bottom, canvasRect.bottom);
  
        const cssW = right - left;
        const cssH = bottom - top;
  
        if (cssW <= 0 || cssH <= 0) {
            return null;
        }
  
        // Convert CSS pixels to canvas backing-store pixels.
        const scaleX = sourceCanvas.width / canvasRect.width;
        const scaleY = sourceCanvas.height / canvasRect.height;
  
        return {
            sx: Math.round((left - canvasRect.left) * scaleX),
            sy: Math.round((top - canvasRect.top) * scaleY),
            sw: Math.round(cssW * scaleX),
            sh: Math.round(cssH * scaleY),
        };
    }
  
    drawCropLetterboxed({
        sourceCanvas,
        cropRect,
        targetCanvas,
        ctx,
        targetWidth,
        targetHeight,
    }) {
        const { sx, sy, sw, sh } = cropRect;
  
        if (!sw || !sh) return false;
  
        const srcAspect = sw / sh;
        const targetAspect = targetWidth / targetHeight;
  
        let drawW, drawH;
        if (srcAspect > targetAspect) {
            drawW = targetWidth;
            drawH = Math.round(targetWidth / srcAspect);
        } else {
            drawH = targetHeight;
            drawW = Math.round(targetHeight * srcAspect);
        }
  
        const offsetX = Math.round((targetWidth - drawW) / 2);
        const offsetY = Math.round((targetHeight - drawH) / 2);
  
        ctx.fillStyle = 'black';
        ctx.fillRect(0, 0, targetWidth, targetHeight);
  
        ctx.drawImage(
            sourceCanvas,
            sx,
            sy,
            sw,
            sh,
            offsetX,
            offsetY,
            drawW,
            drawH
        );
  
        return true;
    }
  
    drawVideoElementLetterboxed({
        sourceVideo,
        targetCanvas,
        ctx,
        targetWidth,
        targetHeight,
    }) {
        if (
            !sourceVideo.videoWidth ||
            !sourceVideo.videoHeight ||
            sourceVideo.readyState < 2
        ) {
            return false;
        }
  
        const srcAspect = sourceVideo.videoWidth / sourceVideo.videoHeight;
        const targetAspect = targetWidth / targetHeight;
  
        let drawW, drawH;
        if (srcAspect > targetAspect) {
            drawW = targetWidth;
            drawH = Math.round(targetWidth / srcAspect);
        } else {
            drawH = targetHeight;
            drawW = Math.round(targetHeight * srcAspect);
        }
  
        const offsetX = Math.round((targetWidth - drawW) / 2);
        const offsetY = Math.round((targetHeight - drawH) / 2);
  
        ctx.fillStyle = 'black';
        ctx.fillRect(0, 0, targetWidth, targetHeight);
  
        ctx.drawImage(
            sourceVideo,
            offsetX,
            offsetY,
            drawW,
            drawH
        );
  
        return true;
    }
  
    startCanvasElementCapture({ participantId, videoPlayer, sourceCanvas, isScreenShare }) {
        const sourceConfig = this.getSourceConfig(isScreenShare);

        if (!sourceConfig?.enabled) {
            window.ws?.sendJson({
                type: 'PerParticipantVideoCaptureManagerSourceConfigNotEnabled',
                participantId: participantId,
                sourceConfig: sourceConfig,
            });
            return;
        }

        const desiredFPS = sourceConfig.framerate || 1;
        const captureIntervalMs = Math.max(1, Math.round(1000 / desiredFPS));

        const blobToDataURL = (blob) => new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.onload = () => resolve(reader.result);
            reader.onerror = reject;
            reader.readAsDataURL(blob);
        });

        const capture = {
            videoPlayer,
            sourceCanvas,
            sourceVideo: null,
            targetCanvas: null,
            ctx: null,
            captureIntervalId: null,
            stopped: false,
            lastSentAt: 0,
            inFlight: false,
        };

        // Just mark that we got this far
        window.ws?.sendJson({
            type: 'PerParticipantVideoCaptureManagerStartCanvasElementCapturezz',
            participantId: participantId,
        });

        const captureFrame = async () => {
            window.ws?.sendJson({
                type: 'PerParticipantVideoCaptureManagerCaptureFrameqq',
                participantId: participantId,
                stopped: capture.stopped,
                inFlight: capture.inFlight,
            });
            if (capture.stopped) return;
            if (capture.inFlight) return;

            window.ws?.sendJson({
                type: 'PerParticipantVideoCaptureManagerCaptureFrameqq0',
                participantId: participantId,
                stopped: capture.stopped,
                inFlight: capture.inFlight,
            });
            try {
                if (!window.ws?.mediaSendingEnabled) return;

                window.ws?.sendJson({
                    type: 'PerParticipantVideoCaptureManagerCaptureFrameqq1',
                    participantId: participantId,
                    stopped: capture.stopped,
                    inFlight: capture.inFlight,
                });

                if (!videoPlayer.isConnected) {
                    this.stopCapture(participantId);
                    window.ws?.sendJson({
                        type: 'PerParticipantVideoCaptureManagerStopCapturezqqq',
                        participantId: participantId,
                    });
                    return;
                }


                window.ws?.sendJson({
                    type: 'PerParticipantVideoCaptureManagerCaptureFrameqq3',
                    participantId: participantId,
                    stopped: capture.stopped,
                    inFlight: capture.inFlight,
                });

                const sdk = videoPlayer.render?.getSDK?.();
                if (!sdk || typeof sdk.ScreenShot !== 'function') {
                    console.warn('[PerParticipantVideoCaptureManager] Zoom SDK not available on videoPlayer');
                    window.ws?.sendJson({
                        type: 'PerParticipantVideoCaptureManagerZoomSDKNotAvailable',
                        participantId: participantId,
                    });
                    return;
                }


                window.ws?.sendJson({
                    type: 'PerParticipantVideoCaptureManagerCaptureFrameqq55',
                    participantId: participantId,
                    stopped: capture.stopped,
                    inFlight: capture.inFlight,
                });

                const nodeId = videoPlayer.getAttribute('node-id');
                if (!nodeId) {
                    console.warn('[PerParticipantVideoCaptureManager] videoPlayer missing node-id attribute');
                    window.ws?.sendJson({
                        type: 'PerParticipantVideoCaptureManagerVideoPlayerMissingNodeIdAttribute',
                        participantId: participantId,
                    });
                    return;
                }

                capture.inFlight = true;


                window.ws?.sendJson({
                    type: 'PerParticipantVideoCaptureManagerCaptureFrameqq177',
                    participantId: participantId,
                    stopped: capture.stopped,
                    inFlight: capture.inFlight,
                });

                const blob = await sdk.ScreenShot(nodeId, isScreenShare ? 'sharing' : 'video');


                window.ws?.sendJson({
                    type: 'PerParticipantVideoCaptureManagerCaptureFrameqq1113',
                    participantId: participantId,
                    stopped: capture.stopped,
                    inFlight: capture.inFlight,
                });

                if (!blob) {
                    window.ws?.sendJson({
                        type: 'PerParticipantVideoCaptureManagerScreenShotFailed',
                        participantId: participantId,
                    });
                    return;
                }

                const bitmap = await createImageBitmap(blob);
                const jpegCanvas = document.createElement('canvas');
                jpegCanvas.width = bitmap.width;
                jpegCanvas.height = bitmap.height;
                const jpegCtx = jpegCanvas.getContext('2d');
                jpegCtx.drawImage(bitmap, 0, 0);
                bitmap.close?.();
                const dataUrl = jpegCanvas.toDataURL('image/jpeg', 0.05);

                window.ws?.sendJson({
                    type: 'PerParticipantVideoScreenshot',
                    participantId: String(participantId),
                    isScreenShare,
                    nodeId,
                    dataUrl,
                });
                capture.lastSentAt = performance.now();
            } catch (err) {
                console.error('[PerParticipantVideoCaptureManager] Error capturing video frame via SDK:', err);
            } finally {
                capture.inFlight = false;
            }
        };

        capture.captureIntervalId = setInterval(captureFrame, captureIntervalMs);
        this.activeCaptures.set(participantId, capture);

        captureFrame();

        console.log('[PerParticipantVideoCaptureManager] Started capture:', {
            participantId,
            isScreenShare,
            sourceType: 'sdk-screenshot',
            desiredFPS,
        });
    }
  
    startVideoElementCapture({ participantId, videoPlayer, sourceVideo, isScreenShare }) {
        const sourceConfig = this.getSourceConfig(isScreenShare);
  
        if (!sourceConfig?.enabled) {
            return;
        }
  
        const desiredFPS = sourceConfig.framerate || 1;
        const captureIntervalMs = Math.max(1, Math.round(1000 / desiredFPS));
  
        const targetWidth = sourceConfig.width;
        const targetHeight = sourceConfig.height;
        const jpegQuality = (sourceConfig.jpeg_quality ?? 80) / 100;
  
        if (!targetWidth || !targetHeight) {
            console.error('[PerParticipantVideoCaptureManager] Missing target dimensions:', sourceConfig);
            return;
        }
  
        const targetCanvas = document.createElement('canvas');
        targetCanvas.width = targetWidth;
        targetCanvas.height = targetHeight;
  
        const ctx = targetCanvas.getContext('2d', { alpha: false });
        if (!ctx) {
            console.error('[PerParticipantVideoCaptureManager] Could not create target canvas context');
            return;
        }
  
        ctx.imageSmoothingEnabled = true;
        ctx.imageSmoothingQuality = 'high';
  
        const capture = {
            videoPlayer,
            sourceCanvas: null,
            sourceVideo,
            targetCanvas,
            ctx,
            captureIntervalId: null,
            stopped: false,
            lastSentAt: 0,
        };
  
        const captureFrame = () => {
            if (capture.stopped) return;
  
            try {
                if (!window.ws?.mediaSendingEnabled) return;
  
                if (!videoPlayer.isConnected || !sourceVideo.isConnected) {
                    this.stopCapture(participantId);
                    return;
                }
  
                const didDraw = this.drawVideoElementLetterboxed({
                    sourceVideo,
                    targetCanvas,
                    ctx,
                    targetWidth,
                    targetHeight,
                });
  
                if (!didDraw) return;
  
                const base64 = targetCanvas.toDataURL('image/jpeg', jpegQuality).split(',', 2)[1];
                if (!base64) return;
  
                window.ws.sendPerParticipantVideo(participantId, isScreenShare, base64);
                capture.lastSentAt = performance.now();
            } catch (err) {
                console.error('[PerParticipantVideoCaptureManager] Error capturing video element frame:', err);
            }
        };
  
        capture.captureIntervalId = setInterval(captureFrame, captureIntervalMs);
        this.activeCaptures.set(participantId, capture);
  
        captureFrame();
  
        console.log('[PerParticipantVideoCaptureManager] Started capture:', {
            participantId,
            isScreenShare,
            sourceType: 'video',
            targetWidth,
            targetHeight,
            desiredFPS,
        });
    }
  
    stopCapture(participantId) {
        const capture = this.activeCaptures.get(participantId);
        if (!capture) return;
  
        capture.stopped = true;
  
        if (capture.captureIntervalId) {
            clearInterval(capture.captureIntervalId);
            capture.captureIntervalId = null;
        }
  
        this.activeCaptures.delete(participantId);
  
        console.log('[PerParticipantVideoCaptureManager] Stopped capture:', {
            participantId,
        });
    }
  }

class ParticipantSpeechStartStopManager {
    constructor() {
        // Only one active speaker at a time
        this.activeSpeaker = null;
    }

    sendSpeechStartStopEvent(participantId, isSpeechStart, timestamp) {
        window.ws?.sendJson({
            type: 'ParticipantSpeechStartStopEvent',
            participantId: participantId.toString(),
            isSpeechStart: isSpeechStart,
            timestamp: timestamp
        });
    }

    addActiveSpeaker(speakerId) {
        if (this.activeSpeaker === speakerId) {
            return;
        }
        if (this.activeSpeaker) {
            this.sendSpeechStartStopEvent(this.activeSpeaker, false, Date.now());
        }
        this.activeSpeaker = speakerId;
        this.sendSpeechStartStopEvent(this.activeSpeaker, true, Date.now());
    }
}

class DominantSpeakerManager {
    constructor() {
        this.dominantSpeakerStreamId = null;
        this.captionAudioTimes = [];
    }

    getLastSpeakerIdForTimestampMs(timestampMs) {
        // Find the caption audio times that are before timestampMs
        const captionAudioTimesBeforeTimestampMs = this.captionAudioTimes.filter(captionAudioTime => captionAudioTime.timestampMs <= timestampMs);
        if (captionAudioTimesBeforeTimestampMs.length === 0) {
            return null;
        }
        // Return the caption audio time with the highest timestampMs
        return captionAudioTimesBeforeTimestampMs.reduce((max, captionAudioTime) => captionAudioTime.timestampMs > max.timestampMs ? captionAudioTime : max).speakerId;
    }

    addCaptionAudioTime(timestampMs, speakerId) {
        this.captionAudioTimes.push({
            timestampMs: timestampMs,
            speakerId: speakerId
        });
    }

    setDominantSpeakerStreamId(dominantSpeakerStreamId) {
        this.dominantSpeakerStreamId = dominantSpeakerStreamId.toString();
    }

    getDominantSpeaker() {
        return virtualStreamToPhysicalStreamMappingManager.virtualStreamIdToParticipant(this.dominantSpeakerStreamId);
    }
}

const handleAudioTrack = async (event) => {
    let lastAudioFormat = null;  // Track last seen format
    
    try {
      // Create processor to get raw frames
      const processor = new MediaStreamTrackProcessor({ track: event.track });
      const generator = new MediaStreamTrackGenerator({ kind: 'audio' });
      
      // Get readable stream of audio frames
      const readable = processor.readable;
      const writable = generator.writable;
  
      const firstStreamId = event.streams[0]?.id;
      if (!firstStreamId) {
        window.ws?.sendJson({
            type: 'AudioTrackError',
            message: 'No stream ID found for audio track'
        });
        return;
      }
      var userIdForStreamId = null;
      var numAttemptsToMapToUserId = 0;
      
        
      // Transform stream to intercept frames
      const transformStream = new TransformStream({
          async transform(frame, controller) {
              if (!frame) {
                  return;
              }
  
              try {
                  // Check if controller is still active
                  if (controller.desiredSize === null) {
                      frame.close();
                      return;
                  }
  
                  // Copy the audio data
                  const numChannels = frame.numberOfChannels;
                  const numSamples = frame.numberOfFrames;
                  const audioData = new Float32Array(numSamples);
                  
                  // Copy data from each channel
                  // If multi-channel, average all channels together
                  if (numChannels > 1) {
                      // Temporary buffer to hold each channel's data
                      const channelData = new Float32Array(numSamples);
                      
                      // Sum all channels
                      for (let channel = 0; channel < numChannels; channel++) {
                          frame.copyTo(channelData, { planeIndex: channel });
                          for (let i = 0; i < numSamples; i++) {
                              audioData[i] += channelData[i];
                          }
                      }
                      
                      // Average by dividing by number of channels
                      for (let i = 0; i < numSamples; i++) {
                          audioData[i] /= numChannels;
                      }
                  } else {
                      // If already mono, just copy the data
                      frame.copyTo(audioData, { planeIndex: 0 });
                  }
  
                  // console.log('frame', frame)
                  // console.log('audioData', audioData)
  
                  // Check if audio format has changed
                  const currentFormat = {
                      numberOfChannels: 1,
                      originalNumberOfChannels: frame.numberOfChannels,
                      numberOfFrames: frame.numberOfFrames,
                      sampleRate: frame.sampleRate,
                      format: frame.format,
                      duration: frame.duration
                  };
  
                  // If format is different from last seen format, send update
                  if (!lastAudioFormat || 
                      JSON.stringify(currentFormat) !== JSON.stringify(lastAudioFormat)) {
                      lastAudioFormat = currentFormat;
                      ws.sendJson({
                          type: 'AudioFormatUpdate',
                          format: currentFormat
                      });
                  }
  
                  // If the audioData buffer is all zeros, then we don't want to send it
                  if (audioData.every(value => value === 0)) {
                      return;
                  }
  
                  if (!userIdForStreamId) {
                    userIdForStreamId = window.userManager?.getUserIdFromStreamId(firstStreamId);
                    if (userIdForStreamId) {
                        window.ws?.sendJson({
                                type: 'AudioTrackMappedToUserId',
                                trackId: event.track.id,
                                streamId: firstStreamId,
                                userId: userIdForStreamId
                        });
                    }
                    numAttemptsToMapToUserId++;
                    if (numAttemptsToMapToUserId === 1000 && !userIdForStreamId) {
                        window.ws?.sendJson({
                            type: 'AudioTrackMappedToUserIdTimedOut',
                            trackId: event.track.id,
                            streamId: firstStreamId,
                        });
                    }
                  }
                  if (userIdForStreamId)
                    ws.sendPerParticipantAudio(userIdForStreamId, audioData);
                      
                  // Pass through the original frame
                  controller.enqueue(frame);
              } catch (error) {
                  console.error('Error processing frame:', error);
                  frame.close();
              }
          },
          flush() {
              console.log('Transform stream flush called');
          }
      });
  
      // Create an abort controller for cleanup
      const abortController = new AbortController();
  
      try {
          // Connect the streams
          await readable
              .pipeThrough(transformStream)
              .pipeTo(writable, {
                  signal: abortController.signal
              })
              .catch(error => {
                  if (error.name !== 'AbortError') {
                      console.error('Pipeline error:', error);
                  }
              });
      } catch (error) {
          console.error('Stream pipeline error:', error);
          abortController.abort();
      }
  
    } catch (error) {
        console.error('Error setting up audio interceptor:', error);
    }
  };
  

class RTCInterceptor {
    constructor(callbacks) {
        // Store the original RTCPeerConnection
        const originalRTCPeerConnection = window.RTCPeerConnection;
        
        // Store callbacks
        const onPeerConnectionCreate = callbacks.onPeerConnectionCreate || (() => {});
        const onDataChannelCreate = callbacks.onDataChannelCreate || (() => {});
        
        // Override the RTCPeerConnection constructor
        window.RTCPeerConnection = function(...args) {
            // Create instance using the original constructor
            const peerConnection = Reflect.construct(
                originalRTCPeerConnection, 
                args
            );
            
            // Notify about the creation
            onPeerConnectionCreate(peerConnection);
            
            // Override createDataChannel
            const originalCreateDataChannel = peerConnection.createDataChannel.bind(peerConnection);
            peerConnection.createDataChannel = (label, options) => {
                const dataChannel = originalCreateDataChannel(label, options);
                onDataChannelCreate(dataChannel, peerConnection);
                return dataChannel;
            };
            
            return peerConnection;
        };
    }
}

new RTCInterceptor({
    onPeerConnectionCreate: (peerConnection) => {
        console.log('New RTCPeerConnection created:', peerConnection);

        peerConnection.addEventListener('track', (event) => {
            console.log('New track:', {
                trackId: event.track.id,
                trackKind: event.track.kind,
                streams: event.streams,
            });

            window.ws?.sendJson({
                type: 'WebRTCTrackStarted',
                trackId: event.track.id,
                trackKind: event.track.kind,
                streams: event.streams?.map(stream => stream?.id),
            });

            // We need to capture every audio track in the meeting,
            // but we don't need to do anything with the video tracks
            if (event.track.kind === 'audio') {
                window.mixedAudioStreamManager?.addAudioTrackFromTrackEvent(event);
                if (window.initialData.sendPerParticipantAudio) {
                    handleAudioTrack(event);
                }
            }
        });
    },
});

class MixedAudioStreamManager {
    constructor() {
        this.audioTracks = [];
        this.meetingAudioStream = null;
        this.audioTracksToBeAdded = [];
        this.audioContext = null;
        this.destination = null;
        this.seenTrackIds = new Set();
    }


    addAudioStream(audioStream) {
        const track = audioStream.getAudioTracks()[0];
        if (track) {
            this.addAudioTrack(track);
        }
    }

    addAudioTrackFromTrackEvent(trackEvent) {
        if (!trackEvent.track)
            return;
        const firstStreamId = trackEvent.streams[0]?.id;
        // streamId must contain +CS+ in it, which means it's from Zoom, not from a voice agent.
        if (!firstStreamId?.includes('+CS+')) {
            window.ws?.sendJson({
                type: 'AudioTrackNotAddedToMeetingAudioStream',
                trackId: trackEvent.track.id,
                streams: trackEvent.streams?.map(stream => stream?.id),
            });
            return;
        }
        window.ws?.sendJson({
            type: 'AudioTrackAddedToMeetingAudioStream',
            trackId: trackEvent.track.id,
            streams: trackEvent.streams?.map(stream => stream?.id),
        });
        this.addAudioTrack(trackEvent.track);
    }

    addAudioTrack(track) {
        if (!track || this.seenTrackIds.has(track.id)) {
            return;
        }

        // If start() already ran, patch the new track into the existing mix.
        if (this.audioContext && this.destination) {
            const mediaStream = new MediaStream([track]);
            const source = this.audioContext.createMediaStreamSource(mediaStream);
            source.connect(this.destination);
            this.seenTrackIds.add(track.id);
        }
        else {
            this.audioTracksToBeAdded.push(track);
        }
    }

    createStream() {
        if (this.meetingAudioStream)
            return;
        this.audioContext = new AudioContext({ sampleRate: 48000 });
        this.destination = this.audioContext.createMediaStreamDestination();

        this.audioTracksToBeAdded.forEach(track => this.addAudioTrack(track));

        this.meetingAudioStream = this.destination.stream;

        // Create a source from the destination's stream so that it actually plays
        this.audioContext.createMediaStreamSource(this.destination.stream);

        window.ws?.sendJson({
            type: 'MeetingAudioStreamCreated',
            message: 'Meeting audio stream created',
        });
    }

    getMeetingAudioStream() {
        this.createStream();
        return this.meetingAudioStream;
    }
}

// Style manager
class StyleManager {
    constructor() {
        this.started = false;
    }

    async start() {
        console.log('StyleManager start');

        this.started = true;

        if (window.zoomInitialData.modifyDomForVideoRecording) {
            this.onlyShowSubsetofZoomUI();
        }

        if (initialData.sendPerParticipantVideo) {
            window.perParticipantVideoCaptureManager.start();
        }
    }
    
    getMeetingAudioStream() {
        if (!this.started)
            return null;
        return window.mixedAudioStreamManager?.getMeetingAudioStream();
    }

    async stop() {
        console.log('StyleManager stop');
        if (window.zoomInitialData.modifyDomForVideoRecording) {
            this.showAllOfZoomUI();
        }
    }

    onlyShowSubsetofZoomUI() {
        try {
            // Find the main element that contains all the video elements
            this.mainElement = document.querySelector('#video-pip-container');
            if (!this.mainElement) {
                console.error('No #video-pip-container element found in the DOM');
                window.ws.sendJson({
                    type: 'Error',
                    message: 'No #video-pip-container element found in the DOM'
                });
                return;
            }

            const ancestors = [];
            let parent = this.mainElement.parentElement;
            while (parent) {
                ancestors.push(parent);
                parent = parent.parentElement;
            }
            
            // Hide all elements except main, its ancestors, and its descendants
            document.querySelectorAll('body *').forEach(element => {
                if (element !== this.mainElement && 
                    !ancestors.includes(element) && 
                    !this.mainElement.contains(element)) {
                    element.style.display = 'none';
                }
            });
        } catch (error) {
            console.error('Error in onlyShowSubsetofZoomUI:', error);
            window.ws.sendJson({
                type: 'Error',
                message: 'Error in onlyShowSubsetofZoomUI: ' + error.message
            });
        }
    }


    showAllOfZoomUI() {
        // Restore all elements that were hidden by onlyShowSubsetofZoomUI
        document.querySelectorAll('body *').forEach(element => {
            if (element.style.display === 'none') {
                // Only reset display property if we set it to 'none'
                // We can check if the element is a direct child of body or not in main/ancestors
                const isInMainTree = this.mainElement && 
                    (this.mainElement === element || 
                     this.mainElement.contains(element) || 
                     element.contains(this.mainElement));
                
                if (!isInMainTree) {
                    // Reset the display property to its default or empty string
                    // This will restore the element's original display value
                    element.style.display = '';
                }
            }
        });
        
        console.log('Restored all hidden elements to their original display values');
    }
}

// Websocket client
class WebSocketClient {
    // Message types
    static MESSAGE_TYPES = {
        JSON: 1,
        VIDEO: 2,
        AUDIO: 3,
        ENCODED_MP4_CHUNK: 4,
        PER_PARTICIPANT_AUDIO: 5,
        PER_PARTICIPANT_VIDEO: 6,
    };

    constructor() {
        const url = `ws://localhost:${window.initialData.websocketPort}`;
        console.log('WebSocketClient url', url);
        this.ws = new WebSocket(url);
        this.ws.binaryType = 'arraybuffer';
        
        this.ws.onopen = () => {
            console.log('WebSocket Connected');
        };
        
        this.ws.onmessage = (event) => {
            this.handleMessage(event.data);
        };
        
        this.ws.onerror = (error) => {
            console.error('WebSocket Error:', error);
        };
        
        this.ws.onclose = () => {
            console.log('WebSocket Disconnected');
        };

        this.mediaSendingEnabled = false;
    }

    async enableMediaSending() {
        this.mediaSendingEnabled = true;
        await window.styleManager.start();
    }

    async disableMediaSending() {
        window.styleManager.stop();
        // Give the media recorder a bit of time to send the final data
        await new Promise(resolve => setTimeout(resolve, 2000));
        this.mediaSendingEnabled = false;
    }

    handleMessage(data) {
        const view = new DataView(data);
        const messageType = view.getInt32(0, true); // true for little-endian
        
        // Handle different message types
        switch (messageType) {
            case WebSocketClient.MESSAGE_TYPES.JSON:
                const jsonData = new TextDecoder().decode(new Uint8Array(data, 4));
                console.log('Received JSON message:', JSON.parse(jsonData));
                break;
            // Add future message type handlers here
            default:
                console.warn('Unknown message type:', messageType);
        }
    }

    sendJson(data) {
        if (this.ws.readyState !== WebSocket.OPEN) {
            console.error('WebSocket is not connected');
            return;
        }

        try {
            // Convert JSON to string then to Uint8Array
            const jsonString = JSON.stringify(data);
            const jsonBytes = new TextEncoder().encode(jsonString);
            
            // Create final message: type (4 bytes) + json data
            const message = new Uint8Array(4 + jsonBytes.length);
            
            // Set message type (1 for JSON)
            new DataView(message.buffer).setInt32(0, WebSocketClient.MESSAGE_TYPES.JSON, true);
            
            // Copy JSON data after type
            message.set(jsonBytes, 4);
            
            // Send the binary message
            this.ws.send(message.buffer);
        } catch (error) {
            console.error('Error sending WebSocket message:', error);
            console.error('Message data:', data);
        }
    }

    sendClosedCaptionUpdate(item) {
        if (!this.mediaSendingEnabled)
            return;

        this.sendJson({
            type: 'CaptionUpdate',
            caption: item
        });
    }

    sendPerParticipantAudio(participantId, audioData) {
        if (this.ws.readyState !== WebSocket.OPEN) {
        console.error('WebSocket is not connected for per participant audio send', this.ws.readyState);
        return;
        }

        if (!this.mediaSendingEnabled) {
        return;
        }

        try {
            // Convert participantId to UTF-8 bytes
            const participantIdBytes = new TextEncoder().encode(participantId);
            
            // Create final message: type (4 bytes) + participantId length (1 byte) + 
            // participantId bytes + audio data
            const message = new Uint8Array(4 + 1 + participantIdBytes.length + audioData.buffer.byteLength);
            const dataView = new DataView(message.buffer);
            
            // Set message type (5 for PER_PARTICIPANT_AUDIO)
            dataView.setInt32(0, WebSocketClient.MESSAGE_TYPES.PER_PARTICIPANT_AUDIO, true);
            
            // Set participantId length as uint8 (1 byte)
            dataView.setUint8(4, participantIdBytes.length);
            
            // Copy participantId bytes
            message.set(participantIdBytes, 5);
            
            // Copy audio data after type, length and participantId
            message.set(new Uint8Array(audioData.buffer), 5 + participantIdBytes.length);
            
            // Send the binary message
            this.ws.send(message.buffer);
        } catch (error) {
            console.error('Error sending WebSocket audio message:', error);
        }
    }

    sendPerParticipantVideo(participantId, isScreenShare, videoData) {
        if (this.ws.readyState !== WebSocket.OPEN) {
          console.error('WebSocket is not connected for per participant video send', this.ws.readyState);
          return;
        }
    
        if (!this.mediaSendingEnabled) {
          return;
        }
    
        try {
            // Convert participantId to UTF-8 bytes
            const participantIdBytes = new TextEncoder().encode(participantId);
            
            // Convert videoData string to UTF-8 bytes
            const videoDataBytes = new TextEncoder().encode(videoData);
            
            // Create final message: type (4 bytes) + participantId length (1 byte) + 
            // participantId bytes + isScreenShare (1 byte) + video data
            const message = new Uint8Array(4 + 1 + participantIdBytes.length + 1 + videoDataBytes.length);
            const dataView = new DataView(message.buffer);
            
            // Set message type (6 for PER_PARTICIPANT_VIDEO)
            dataView.setInt32(0, WebSocketClient.MESSAGE_TYPES.PER_PARTICIPANT_VIDEO, true);
            
            // Set participantId length as uint8 (1 byte)
            dataView.setUint8(4, participantIdBytes.length);
            
            // Copy participantId bytes
            message.set(participantIdBytes, 5);
            
            // Set isScreenShare byte (0 = webcam, 1 = screenshare)
            dataView.setUint8(5 + participantIdBytes.length, isScreenShare ? 1 : 0);
            
            // Copy video data after type, length, participantId, and isScreenShare
            message.set(videoDataBytes, 5 + participantIdBytes.length + 1);
            
            // Send the binary message
            this.ws.send(message.buffer);
        } catch (error) {
            console.error('Error sending WebSocket video message:', error);
        }
      }
  

    sendMixedAudio(timestamp, audioData) {
        if (this.ws.readyState !== WebSocket.OPEN) {
            console.error('WebSocket is not connected for audio send', this.ws.readyState);
            return;
        }

        if (!this.mediaSendingEnabled) {
            return;
        }

        try {
            // Create final message: type (4 bytes) + audio data
            const message = new Uint8Array(4 + audioData.buffer.byteLength);
            const dataView = new DataView(message.buffer);
            
            // Set message type (3 for AUDIO)
            dataView.setInt32(0, WebSocketClient.MESSAGE_TYPES.AUDIO, true);
            
            // Copy audio data after type
            message.set(new Uint8Array(audioData.buffer), 4);
            
            // Send the binary message
            this.ws.send(message.buffer);
        } catch (error) {
            console.error('Error sending WebSocket audio message:', error);
        }
    }
}

class UserManager {
    constructor(ws) {
        this.allUsersMap = new Map();
        this.currentUsersMap = new Map();
        this.deviceOutputMap = new Map();

        this.ws = ws;
    }

    getUserIdFromStreamId(streamId) {
        const decoded = decodeURIComponent(streamId);
        const match = decoded.match(/^(\d+)\+/);
        if (match) {
            const rawId = Number(match[1]);
            const participantId = rawId >> 10 << 10;
            // Check if this exists in the current users map
            if (this.currentUsersMap.has(participantId.toString())) {
                return participantId.toString();
            }
            return null;
        }
        return null;
    }

    getUserByDeviceId(deviceId) {
        return this.allUsersMap.get(deviceId);
    }

    // constants for meeting status
    MEETING_STATUS = {
        IN_MEETING: 1,
        NOT_IN_MEETING: 6
    }

    getCurrentUsersInMeeting() {
        return Array.from(this.currentUsersMap.values()).filter(user => user.status === this.MEETING_STATUS.IN_MEETING);
    }

    getCurrentUsersInMeetingWhoAreScreenSharing() {
        return this.getCurrentUsersInMeeting().filter(user => user.parentDeviceId);
    }

    convertUser(zoomUser) {
        return {
            deviceId: zoomUser.userId.toString(),
            displayName: zoomUser.userName,
            fullName: zoomUser.userName,
            profile: '',
            status: zoomUser.state,
            isHost: zoomUser.isHost,
            humanized_status: zoomUser.state === "active" ? "in_meeting" : "not_in_meeting",
            isCurrentUser: zoomUser.self
        };
    }

    singleUserSynced(user) {
      const convertedUser = this.convertUser(user);
      console.log('singleUserSynced called w', convertedUser);
      // Create array with new user and existing users, then filter for unique deviceIds
      // keeping the first occurrence (new user takes precedence)
      const allUsers = [...this.currentUsersMap.values(), convertedUser];
      console.log('allUsers', allUsers);
      const uniqueUsers = Array.from(
        new Map(allUsers.map(singleUser => [singleUser.deviceId, singleUser])).values()
      );
      this.newUsersListSynced(uniqueUsers);
    }

    newUsersListSynced(newUsersList) {
        console.log('newUsersListSynced called w', newUsersList);
        // Get the current user IDs before updating
        const previousUserIds = new Set(this.currentUsersMap.keys());
        const newUserIds = new Set(newUsersList.map(user => user.deviceId));
        const updatedUserIds = new Set([])

        // Update all users map
        for (const user of newUsersList) {
            if (previousUserIds.has(user.deviceId) && JSON.stringify(this.currentUsersMap.get(user.deviceId)) !== JSON.stringify(user)) {
                updatedUserIds.add(user.deviceId);
            }

            this.allUsersMap.set(user.deviceId, {
                deviceId: user.deviceId,
                displayName: user.displayName,
                fullName: user.fullName,
                profile: user.profile,
                status: user.status,
                humanized_status: user.humanized_status,
                parentDeviceId: user.parentDeviceId,
                isCurrentUser: user.isCurrentUser,
                isHost: user.isHost
            });
        }

        // Calculate new, removed, and updated users
        const newUsers = newUsersList.filter(user => !previousUserIds.has(user.deviceId));
        const removedUsers = Array.from(previousUserIds)
            .filter(id => !newUserIds.has(id))
            .map(id => this.currentUsersMap.get(id));

        if (removedUsers.length > 0) {
            console.log('removedUsers', removedUsers);
        }

        // Clear current users map and update with new list
        this.currentUsersMap.clear();
        for (const user of newUsersList) {
            this.currentUsersMap.set(user.deviceId, {
                deviceId: user.deviceId,
                displayName: user.displayName,
                fullName: user.fullName,
                profilePicture: user.profilePicture,
                status: user.status,
                humanized_status: user.humanized_status,
                parentDeviceId: user.parentDeviceId,
                isCurrentUser: user.isCurrentUser,
                isHost: user.isHost
            });
        }

        const updatedUsers = Array.from(updatedUserIds).map(id => this.currentUsersMap.get(id));

        if (newUsers.length > 0 || removedUsers.length > 0 || updatedUsers.length > 0) {
            this.ws.sendJson({
                type: 'UsersUpdate',
                newUsers: newUsers,
                removedUsers: removedUsers,
                updatedUsers: updatedUsers
            });
        }
    }
}

// This code intercepts the connect method on the AudioNode class
// When something is connected to the speaker the underlying track is added to our styleManager
// so that it can be aggregated into a stream representing the meeting audio
(() => {
    const origConnect = AudioNode.prototype.connect;
  
    AudioNode.prototype.connect = function(target, ...rest) {

      // Only intercept connections directly to the speakers. The target !== window.botOutputManager?.getAudioContextDestination() condition is to avoid capturing the bots output 
      if (target instanceof AudioDestinationNode && target !== window.botOutputManager?.getAudioContextDestination()) {
        const ctx = this.context;
        // Create a single tee per context
        if (!ctx.__captureTee) {
        try{
          const tee = ctx.createGain();
          const tap = ctx.createMediaStreamDestination();
          origConnect.call(tee, ctx.destination); // keep normal playback
          origConnect.call(tee, tap);             // capture
          ctx.__captureTee = { tee, tap };
          const capturedStream = tap.stream;
          if (capturedStream)
            window.mixedAudioStreamManager.addAudioStream(capturedStream);
        }
        catch (error) {
            console.error('Error in AudioNodeInterceptor:', error);
        }
        }
  
        // Reroute to the tee instead of the destination
        return origConnect.call(this, ctx.__captureTee.tee, ...rest);
      }
  
      return origConnect.call(this, target, ...rest);
    };
  })();

const ws = new WebSocketClient();
window.ws = ws;
const dominantSpeakerManager = new DominantSpeakerManager();
window.dominantSpeakerManager = dominantSpeakerManager;
const styleManager = new StyleManager();
window.styleManager = styleManager;
const userManager = new UserManager(ws);
window.userManager = userManager;
const participantSpeechStartStopManager = new ParticipantSpeechStartStopManager();
window.participantSpeechStartStopManager = participantSpeechStartStopManager;
const mixedAudioStreamManager = new MixedAudioStreamManager();
window.mixedAudioStreamManager = mixedAudioStreamManager;
const perParticipantVideoCaptureManager = new PerParticipantVideoCaptureManager();
window.perParticipantVideoCaptureManager = perParticipantVideoCaptureManager;

const turnOnCameraArialLabel = "start my video"
const turnOffCameraArialLabel = "stop my video"
const turnOnMicArialLabel = "unmute my microphone"
const turnOffMicArialLabel = "mute my microphone"
const turnOnScreenshareArialLabel = "Share Screen"
const turnOffScreenshareClass = "sharer-button--stop"

async function turnOnCamera() {
    // Click camera button to turn it on
    let cameraButton = null;
    const numAttempts = 30;
    for (let i = 0; i < numAttempts; i++) {
        cameraButton = document.querySelector(`button[aria-label="${turnOnCameraArialLabel}"]`) || document.querySelector(`div[aria-label="${turnOnCameraArialLabel}"]`);
        if (cameraButton) {
            break;
        }
        window.ws.sendJson({
            type: 'Error',
            message: 'Camera button not found in turnOnCamera, but will try again'
        });
        await new Promise(resolve => setTimeout(resolve, 100));
    }
    
    if (cameraButton) {
        console.log("Clicking the camera button to turn it on");
        cameraButton.click();
    } else {
        console.log("Camera button not found");
        window.ws.sendJson({
            type: 'Error',
            message: 'Camera button not found in turnOnCamera'
        });
    }
}

function turnOnMic() {
    // Click microphone button to turn it on
    const microphoneButton = document.querySelector(`button[aria-label="${turnOnMicArialLabel}"]`) || document.querySelector(`div[aria-label="${turnOnMicArialLabel}"]`);
    if (microphoneButton) {
        console.log("Clicking the microphone button to turn it on");
        microphoneButton.click();
    } else {
        console.log("Microphone button not found");
    }
}

function turnOffMic() {
    // Click microphone button to turn it off
    const microphoneButton = document.querySelector(`button[aria-label="${turnOffMicArialLabel}"]`) || document.querySelector(`div[aria-label="${turnOffMicArialLabel}"]`);
    if (microphoneButton) {
        console.log("Clicking the microphone button to turn it off");
        microphoneButton.click();
    } else {
        console.log("Microphone off button not found");
    }
}

function turnOnMicAndCamera() {
    // Click microphone button to turn it on
    const microphoneButton = document.querySelector(`button[aria-label="${turnOnMicArialLabel}"]`) || document.querySelector(`div[aria-label="${turnOnMicArialLabel}"]`);
    if (microphoneButton) {
        console.log("Clicking the microphone button to turn it on");
        microphoneButton.click();
    } else {
        console.log("Microphone button not found");
    }

    // Click camera button to turn it on
    const cameraButton = document.querySelector(`button[aria-label="${turnOnCameraArialLabel}"]`) || document.querySelector(`div[aria-label="${turnOnCameraArialLabel}"]`);
    if (cameraButton) {
        console.log("Clicking the camera button to turn it on");
        cameraButton.click();
    } else {
        console.log("Camera button not found");
    }
}

function turnOffMicAndCamera() {
    // Click microphone button to turn it off
    const microphoneButton = document.querySelector(`button[aria-label="${turnOffMicArialLabel}"]`) || document.querySelector(`div[aria-label="${turnOffMicArialLabel}"]`);
    if (microphoneButton) {
        console.log("Clicking the microphone button to turn it off");
        microphoneButton.click();
    } else {
        console.log("Microphone off button not found");
    }

    // Click camera button to turn it off
    const cameraButton = document.querySelector(`button[aria-label="${turnOffCameraArialLabel}"]`) || document.querySelector(`div[aria-label="${turnOffCameraArialLabel}"]`);
    if (cameraButton) {
        console.log("Clicking the camera button to turn it off");
        cameraButton.click();
    } else {
        console.log("Camera off button not found");
    }
}

function turnOnMicAndScreenshare() {
    // Click microphone button to turn it on
    const microphoneButton = document.querySelector(`button[aria-label="${turnOnMicArialLabel}"]`) || document.querySelector(`div[aria-label="${turnOnMicArialLabel}"]`);
    if (microphoneButton) {
        console.log("Clicking the microphone button to turn it on");
        microphoneButton.click();
    } else {
        console.log("Microphone button not found");
    }

    // Click screenshare button to turn it on
    const screenshareButton = document.querySelector(`button[aria-label="${turnOnScreenshareArialLabel}"]`) || document.querySelector(`div[aria-label="${turnOnScreenshareArialLabel}"]`);
    if (screenshareButton) {
        console.log("Clicking the screenshare button to turn it on");
        screenshareButton.click();
    } else {
        console.log("Screenshare button not found");
    }
}

function turnOffMicAndScreenshare() {
    // Click microphone button to turn it off
    const microphoneButton = document.querySelector(`button[aria-label="${turnOffMicArialLabel}"]`) || document.querySelector(`div[aria-label="${turnOffMicArialLabel}"]`);
    if (microphoneButton) {
        console.log("Clicking the microphone button to turn it off");
        microphoneButton.click();
    } else {
        console.log("Microphone off button not found");
    }

    // Click screenshare button to turn it off
    const screenshareButton = document.querySelector(`.${turnOffScreenshareClass}`);
    if (screenshareButton) {
        console.log("Clicking the screenshare button to turn it off");
        screenshareButton.click();
    } else {
        console.log("Screenshare off button not found");
    }
}

function turnOnScreenshare() {
    // Click screenshare button to turn it on
    const screenshareButton = document.querySelector(`button[aria-label="${turnOnScreenshareArialLabel}"]`) || document.querySelector(`div[aria-label="${turnOnScreenshareArialLabel}"]`);
    if (screenshareButton) {
        console.log("Clicking the screenshare button to turn it on");
        screenshareButton.click();
    } else {
        console.log("Screenshare button not found");
    }
}

function turnOffScreenshare() {
    // Click screenshare button to turn it off
    const screenshareButton = document.querySelector(`.${turnOffScreenshareClass}`);
    if (screenshareButton) {
        console.log("Clicking the screenshare button to turn it off");
        screenshareButton.click();
    } else {
        console.log("Screenshare off button not found");
    }
}

// BotOutputManager is defined in shared_chromedriver_payload.js

botOutputManager = new BotOutputManager({
    turnOnWebcam: turnOnCamera,
    turnOffWebcam: () => {
        console.log("Turning off webcam");
    },
    turnOnScreenshare: turnOnScreenshare,
    turnOffScreenshare: turnOffScreenshare,
    turnOnMic: turnOnMic,
    turnOffMic: turnOffMic,
    callOriginalGetUserMedia: true,
});

window.botOutputManager = botOutputManager;
