class EyeFatigueDetector {
    constructor() {
        this.video = document.getElementById('video-feed');
        this.startBtn = document.getElementById('start-btn');
        this.stopBtn = document.getElementById('stop-btn');
        this.earValue = document.getElementById('ear-value');
        this.fatigueDuration = document.getElementById('fatigue-duration');
        this.fatigueAlert = document.getElementById('fatigue-alert');
        
        this.stream = null;
        this.isDetecting = false;
        this.canvas = document.createElement('canvas');
        this.context = this.canvas.getContext('2d');
        this.analysisInterval = null;
        
        // Fatigue tracking
        this.fatigueStartTime = null;
        this.currentFatigueDuration = 0;
        this.fatigueThreshold = 0.25;
        this.maxFatigueDuration = 0;
        
        this.initializeEventListeners();
        this.checkSystemStatus();
    }

    initializeEventListeners() {
        this.startBtn.addEventListener('click', () => this.startDetection());
        this.stopBtn.addEventListener('click', () => this.stopDetection());
        
        // Handle page visibility changes
        document.addEventListener('visibilitychange', () => {
            if (document.hidden && this.isDetecting) {
                this.pauseDetection();
            } else if (!document.hidden && this.isDetecting) {
                this.resumeDetection();
            }
        });
    }

    async checkSystemStatus() {
        try {
            const response = await fetch('/fatigue/status');
            const status = await response.json();
            
            if (!status.model_loaded || !status.face_detection_ready) {
                this.showError('System not ready. Please check if all models are loaded.');
                this.startBtn.disabled = true;
            }
            
            console.log('System status:', status);
        } catch (error) {
            console.error('Error checking system status:', error);
            this.showError('Could not connect to detection service.');
        }
    }

    async startDetection() {
        try {
            // Request camera access
            this.stream = await navigator.mediaDevices.getUserMedia({
                video: {
                    width: { ideal: 1280 },
                    height: { ideal: 720 },
                    facingMode: 'user'
                },
                audio: false
            });

            this.video.srcObject = this.stream;
            this.video.play();

            // Wait for video to be ready
            await new Promise((resolve) => {
                this.video.onloadedmetadata = () => {
                    this.canvas.width = this.video.videoWidth;
                    this.canvas.height = this.video.videoHeight;
                    resolve();
                };
            });

            this.isDetecting = true;
            this.startBtn.classList.add('hidden');
            this.stopBtn.classList.remove('hidden');

            // Start analysis loop
            this.startAnalysisLoop();
            
            this.showSuccess('Detection started successfully');
            
        } catch (error) {
            console.error('Error starting detection:', error);
            this.showError('Could not access camera. Please check permissions.');
        }
    }

    stopDetection() {
        this.isDetecting = false;
        
        // Stop analysis loop
        if (this.analysisInterval) {
            clearInterval(this.analysisInterval);
            this.analysisInterval = null;
        }

        // Stop camera stream
        if (this.stream) {
            this.stream.getTracks().forEach(track => track.stop());
            this.stream = null;
        }

        // Reset UI
        this.video.srcObject = null;
        this.startBtn.classList.remove('hidden');
        this.stopBtn.classList.add('hidden');
        this.hideFatigueAlert();
        
        // Reset stats
        this.earValue.textContent = '0.00';
        this.fatigueDuration.textContent = '0s';
        this.fatigueStartTime = null;
        this.currentFatigueDuration = 0;
    }

    startAnalysisLoop() {
        // Analyze frame every 500ms to balance accuracy and performance
        this.analysisInterval = setInterval(async () => {
            if (this.isDetecting && this.video.readyState === 4) {
                await this.analyzeCurrentFrame();
            }
        }, 500);
    }

    async analyzeCurrentFrame() {
        try {
            // Capture current frame
            this.context.drawImage(this.video, 0, 0, this.canvas.width, this.canvas.height);
            
            // Convert canvas to blob
            const blob = await new Promise(resolve => {
                this.canvas.toBlob(resolve, 'image/jpeg', 0.8);
            });

            // Create FormData for upload
            const formData = new FormData();
            formData.append('frame', blob, 'frame.jpg');

            // Send to analysis endpoint
            const response = await fetch('/analyze_frame', {
                method: 'POST',
                body: formData
            });

            if (response.ok) {
                const result = await response.json();
                this.updateUI(result);
            } else {
                const error = await response.json();
                console.error('Analysis error:', error.detail);
                
                // Don't show error for no face detected (common when user moves)
                if (!error.detail.includes('No face detected')) {
                    this.showError(`Analysis error: ${error.detail}`);
                }
            }

        } catch (error) {
            console.error('Error analyzing frame:', error);
            // Only show error if it's persistent
            if (this.consecutiveErrors > 3) {
                this.showError('Connection lost. Retrying...');
            }
        }
    }

    updateUI(result) {
        // Update EAR display
        this.earValue.textContent = result.avg_ear.toFixed(3);
        
        // Handle fatigue detection
        if (result.is_fatigued) {
            if (!this.fatigueStartTime) {
                this.fatigueStartTime = Date.now();
            }
            
            // Calculate current fatigue duration
            this.currentFatigueDuration = (Date.now() - this.fatigueStartTime) / 1000;
            this.fatigueDuration.textContent = `${this.currentFatigueDuration.toFixed(1)}s`;
            
            // Update max fatigue duration
            this.maxFatigueDuration = Math.max(this.maxFatigueDuration, this.currentFatigueDuration);
            
            // Show alert if fatigued for more than 2 seconds
            if (this.currentFatigueDuration > 2) {
                this.showFatigueAlert();
            }
            
        } else {
            // Reset fatigue tracking
            this.fatigueStartTime = null;
            this.currentFatigueDuration = 0;
            this.fatigueDuration.textContent = '0s';
            this.hideFatigueAlert();
        }

        // Update stats with additional info
        this.updateStats(result);
    }

    updateStats(result) {
        // You can add more detailed statistics here
        const confidencePercent = (result.confidence * 100).toFixed(1);
        
        // Update title attributes for additional info
        this.earValue.title = `Left: ${result.left_ear.toFixed(3)}, Right: ${result.right_ear.toFixed(3)}`;
        
        if (result.is_fatigued) {
            this.fatigueDuration.title = `Confidence: ${confidencePercent}% | Max: ${this.maxFatigueDuration.toFixed(1)}s`;
        }
    }

    showFatigueAlert() {
        this.fatigueAlert.style.display = 'block';
        
        // Play alert sound if available
        this.playAlertSound();
        
        // Vibrate if supported
        if (navigator.vibrate) {
            navigator.vibrate([200, 100, 200]);
        }
    }

    hideFatigueAlert() {
        this.fatigueAlert.style.display = 'none';
    }

    playAlertSound() {
        // Try to play alert sound
        try {
            const audio = new Audio('/static/alert.mp3');
            audio.volume = 0.3;
            audio.play().catch(e => {
                // Ignore audio play errors (user interaction required)
                console.log('Audio play blocked:', e.message);
            });
        } catch (error) {
            console.log('Audio not available');
        }
    }

    pauseDetection() {
        if (this.analysisInterval) {
            clearInterval(this.analysisInterval);
            this.analysisInterval = null;
        }
    }

    resumeDetection() {
        if (this.isDetecting) {
            this.startAnalysisLoop();
        }
    }

    showError(message) {
        console.error(message);
        // You can implement a proper error display UI here
        // For now, we'll use console.error
    }

    showSuccess(message) {
        console.log(message);
        // You can implement a proper success display UI here
    }
}

// Initialize the detector when the page loads
document.addEventListener('DOMContentLoaded', () => {
    const detector = new EyeFatigueDetector();
    
    // Add global error handler
    window.addEventListener('error', (event) => {
        console.error('Global error:', event.error);
    });
    
    // Add unhandled promise rejection handler
    window.addEventListener('unhandledrejection', (event) => {
        console.error('Unhandled promise rejection:', event.reason);
    });
});

// Service Worker registration for offline capabilities (optional)
if ('serviceWorker' in navigator) {
    window.addEventListener('load', () => {
        navigator.serviceWorker.register('/static/js/sw.js')
        .then(registration => {
            console.log('SW registered: ', registration);
        })
        .catch(registrationError => {
            console.log('SW registration failed: ', registrationError);
        });
    });
}