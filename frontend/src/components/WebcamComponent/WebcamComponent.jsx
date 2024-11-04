/* eslint-disable react/prop-types */
/* eslint-disable no-unused-vars */
import { useRef, useEffect, useState } from 'react';
import io from 'socket.io-client';

// Initialize Socket.io client to communicate with the Express server
const socket = io('http://localhost:5000', {
  transports: ['websocket'],
});

const WebcamComponent = ({selectedPose}) => {
  console.log(selectedPose)
  const videoRef = useRef(null);
  const [isStreaming, setIsStreaming] = useState(false);
  const intervalRef = useRef(null);
  const [poseResults, setPoseResults] = useState([]); // State to hold pose estimation results

  const startWebcam = () => {
    if (navigator.mediaDevices.getUserMedia) {
      navigator.mediaDevices.getUserMedia({ video: true })
        .then((stream) => {
          videoRef.current.srcObject = stream;
          setIsStreaming(true);
          startSendingFrames();
        })
        .catch((err) => {
          console.error('Error accessing webcam:', err);
        });
    }
  };

  const startSendingFrames = () => {
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
    }

    // Capture and send the first frame immediately
    if (videoRef.current && videoRef.current.srcObject) {
      console.log("Sending the first frame immediately");
      captureAndSendFrame();
    }

    intervalRef.current = setInterval(() => {
      if (videoRef.current && videoRef.current.srcObject) {
        console.log("Sending frame after 7 seconds");
        captureAndSendFrame();
      } else {
        console.log("Webcam not streaming, stopping frame capture");
        clearInterval(intervalRef.current);
      }
    }, 7000);
  };

  const captureAndSendFrame = () => {
    const videoElement = videoRef.current;
    const canvas = document.createElement('canvas');
    canvas.width = videoElement.videoWidth;
    canvas.height = videoElement.videoHeight;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(videoElement, 0, 0, canvas.width, canvas.height);
    const frame = canvas.toDataURL('image/jpeg');

    // Emit the frame to Express server via Socket.io
    socket.emit('frame', { image: frame, poseType: selectedPose });
  };

  const closeWebcam = () => {
    if (videoRef.current && videoRef.current.srcObject) {
      const stream = videoRef.current.srcObject;
      const tracks = stream.getTracks();
      tracks.forEach(track => track.stop());
      videoRef.current.srcObject = null;
      setIsStreaming(false);
  
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
        intervalRef.current = null;
      }
  
      // Clear the pose results when closing the webcam
      setPoseResults([]); // Reset the pose results
    }
  };

  useEffect(() => {
    socket.on('connection-status', (data) => {
      console.log(data)
      console.log('Connection status:', data.message);
    });

    socket.on('pose-estimation-result', (data) => {
      console.log('Pose Estimation Result:', data.prompts); // Log the received prompts
      setPoseResults(data.prompts); // Store the results in state for rendering
    });

    socket.on('connect', () => {
      console.log('Socket connected');
    });

    socket.on('disconnect', () => {
      console.log('Socket disconnected');
    });

    return () => closeWebcam();
  }, []);

  return (
    <div className="flex items-center justify-center h-screen w-screen">
      <div className="bg-gray-100 p-6 rounded-lg shadow-lg" style={{ width: '940px', height: '600px' }}>
        <video
          ref={videoRef}
          className="rounded-md border w-full h-[500px]"
          autoPlay
          playsInline
          style={{ maxWidth: '940px', maxHeight: '680px' }}
        />
        <div className="mt-4 flex justify-between">
          <button
            onClick={startWebcam}
            className="px-4 py-2 bg-green-500 text-white rounded hover:bg-green-600"
          >
            Start Webcam
          </button>
          <button
            onClick={closeWebcam}
            className="px-4 py-2 bg-red-500 text-white rounded hover:bg-red-600"
          >
            Close Webcam
          </button>
        </div>
        {/* Display pose results here */}
        <div className="mt-4 pt-10 flex flex-col justify-center items-center">
          <h3 className="text-lg font-bold">Pose Estimation Results:</h3>
          {poseResults && Array.isArray(poseResults) && poseResults.length > 0 ? (
            poseResults.map((prompt, index) => (
              <div key={index}>{prompt}</div>
            ))
          ) : (
            <p>No pose results available.</p> // Provide feedback if there are no results
          )}
        </div>
      </div>
    </div>
  );
};

export default WebcamComponent;
