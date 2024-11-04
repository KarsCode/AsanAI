const express = require('express');
const http = require('http');
const socketIo = require('socket.io');
const axios = require('axios');
const FormData = require('form-data'); // Import FormData to send form data

const app = express();
const server = http.createServer(app);
const io = socketIo(server);

io.on('connection', (socket) => {
  console.log('Client connected');

  // Listen for frames from the frontend
  socket.on('frame', async (data) => {
    try {
      const formData = new FormData();
      // Assuming data.image contains the base64 string of the image
      const buffer = Buffer.from(data.image.split(',')[1], 'base64'); // Convert base64 to buffer
      formData.append('file', buffer, {
        filename: 'image.jpg', // Set a filename for the image
        contentType: 'image/jpeg', // Set the content type
      });
      formData.append('poseType', data.poseType);

      console.log(data.poseType)

      // Forward the frame to FastAPI for processing
      const response = await axios.post('http://localhost:8000/pose-estimation/', formData, {
        headers: {
          ...formData.getHeaders(), // Include headers required for form-data
        },
      });

      console.log('FastAPI response:', response.data.prompts); // Adjust based on your FastAPI response structure
       socket.emit('pose-estimation-result', { prompts: response.data.prompts })

      // Send back status to frontend
      socket.emit('connection-status', { prompts: response.data.prompts });
    } catch (error) {
      console.error('Error connecting to FastAPI:', error.response ? error.response.data : error.message);
      socket.emit('connection-status', { error: 'Failed to process the frame.' });
    }
  });

  socket.on('disconnect', () => {
    console.log('Client disconnected');
  });
});

server.listen(5000, () => {
  console.log('Express server running on port 5000');
});
