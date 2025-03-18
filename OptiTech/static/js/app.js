// DOM elements
const videoFeed = document.getElementById('video-feed');
const fatigueAlert = document.getElementById('fatigue-alert');
const earValue = document.getElementById('ear-value');
const fatigueDuration = document.getElementById('fatigue-duration');
const startBtn = document.getElementById('start-btn');
const stopBtn = document.getElementById('stop-btn');

// Configuration
const BLINK_THRESHOLD = 0.2;         // EAR threshold to detect blink
const FATIGUE_THRESHOLD = 2.0;       // Seconds of eye closure for fatigue detection
let stream = null;
let isDetecting = false;
let closureStartTime = null;
let currentEAR = 0;
let currentFatigueDuration = 0;
let isEyeClosed = false;
let frameInterval = null;

// Start webcam and detection
startBtn.addEventListener('click', async () => {
    try {
        stream = await navigator.mediaDevices.getUserMedia({ video: true });
        videoFeed.srcObject = stream;
        
        startBtn.classList.add('hidden');
        stopBtn.classList.remove('hidden');
        isDetecting = true;
        
        // Start sending frames to backend for analysis
        frameInterval = setInterval(analyzeFrame, 100);  // 10 frames per second
    } catch (error) {
        console.error('Error accessing webcam:', error);
        alert('Unable to access webcam. Please make sure your camera is connected and permissions are granted.');
    }
});

// Stop detection
stopBtn.addEventListener('click', () => {
    stopDetection();
});

function stopDetection() {
    if (stream) {
        stream.getTracks().forEach(track => track.stop());
        videoFeed.srcObject = null;
    }
    
    if (frameInterval) {
        clearInterval(frameInterval);
    }
    
    startBtn.classList.remove('hidden');
    stopBtn.classList.add('hidden');
    isDetecting = false;
    fatigueAlert.style.display = 'none';
    
    // Reset values
    earValue.textContent = '0.00';
    fatigueDuration.textContent = '0s';
    closureStartTime = null;
    currentEAR = 0;
    currentFatigueDuration = 0;
    isEyeClosed = false;
}

async function analyzeFrame() {
    if (!isDetecting || !videoFeed.srcObject) return;

    try {
        // Create a canvas and draw the current video frame
        const canvas = document.createElement('canvas');
        canvas.width = videoFeed.videoWidth;
        canvas.height = videoFeed.videoHeight;
        const ctx = canvas.getContext('2d');
        ctx.drawImage(videoFeed, 0, 0, canvas.width, canvas.height);
        
        // Convert canvas to blob
        const blob = await new Promise(resolve => {
            canvas.toBlob(resolve, 'image/jpeg');
        });
        
        // Create form data to send to backend
        const formData = new FormData();
        formData.append('frame', blob, 'frame.jpg');
        
        // Send to backend API
        const response = await fetch('/analyze_frame', {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error ${response.status}`);
        }
        
        const data = await response.json();
        
        // Update UI with detection results
        updateDetectionResults(data);
        
    } catch (error) {
        console.error('Error analyzing frame:', error);
    }
}

function updateDetectionResults(data) {
    if (data.error) {
        console.error('Backend error:', data.error);
        return;
    }
    
    // Update EAR value
    currentEAR = data.avg_ear;
    earValue.textContent = currentEAR.toFixed(2);
    
    // Check if eyes are closed based on EAR
    isEyeClosed = currentEAR < BLINK_THRESHOLD;
    
    // Track eye closure time
    const currentTime = Date.now() / 1000;  // Current time in seconds
    
    if (isEyeClosed) {
        if (closureStartTime === null) {
            closureStartTime = currentTime;
        }
        
        currentFatigueDuration = currentTime - closureStartTime;
        fatigueDuration.textContent = currentFatigueDuration.toFixed(1) + 's';
        
        // Check for fatigue
        if (currentFatigueDuration > FATIGUE_THRESHOLD) {
            showFatigueAlert();
        }
    } else {
        closureStartTime = null;
        currentFatigueDuration = 0;
        fatigueDuration.textContent = '0s';
        fatigueAlert.style.display = 'none';
    }
}

function showFatigueAlert() {
    // Display fatigue alert
    if (fatigueAlert.style.display !== 'block') {
        fatigueAlert.style.display = 'block';
        
        // Optionally add sound alert
        const audio = new Audio('/static/alert.mp3');
        audio.play().catch(e => console.error('Could not play alert sound:', e));
        
        // Vibration if supported
        if ('vibrate' in navigator) {
            navigator.vibrate([200, 100, 200]);
        }
    }
}

// Handle page unload
window.addEventListener('beforeunload', stopDetection);