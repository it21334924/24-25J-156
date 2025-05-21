document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements
    const startTestBtn = document.getElementById('start-test-btn');
    const nextStepBtn = document.getElementById('next-step-btn');
    const retryTestBtn = document.getElementById('retry-test-btn');
    const testContainer = document.getElementById('test-container');
    const resultsContainer = document.getElementById('results-container');
    const colorGrid = document.getElementById('color-grid');
    const targetColorEl = document.getElementById('target-color');
    const currentStepEl = document.getElementById('current-step');
    const totalStepsEl = document.getElementById('total-steps');
    const progressBar = document.getElementById('progress-bar');
    const resultMessage = document.getElementById('result-message');
    const resultDetails = document.getElementById('result-details');

    // Test Configuration
    const TOTAL_STEPS = 8;
    let currentStep = 0;
    let userSelections = [];
    let startTime; // Track when each test step starts
    let testResults = {}; // Store comprehensive test results
    
    // Color configurations for the test
    const colors = [
        { name: 'Red', hex: '#FF0000', rgb: [255, 0, 0] },
        { name: 'Green', hex: '#00FF00', rgb: [0, 255, 0] },
        { name: 'Blue', hex: '#0000FF', rgb: [0, 0, 255] },
        { name: 'Yellow', hex: '#FFFF00', rgb: [255, 255, 0] },
        { name: 'Cyan', hex: '#00FFFF', rgb: [0, 255, 255] },
        { name: 'Magenta', hex: '#FF00FF', rgb: [255, 0, 255] },
        { name: 'White', hex: '#FFFFFF', rgb: [255, 255, 255] },
        { name: 'Black', hex: '#000000', rgb: [0, 0, 0] },
        // Additional colors for more nuanced testing
        { name: 'Orange', hex: '#FFA500', rgb: [255, 165, 0] },
        { name: 'Purple', hex: '#800080', rgb: [128, 0, 128] },
        { name: 'Brown', hex: '#A52A2A', rgb: [165, 42, 42] },
        { name: 'Pink', hex: '#FFC0CB', rgb: [255, 192, 203] },
        { name: 'Light Green', hex: '#90EE90', rgb: [144, 238, 144] },
        { name: 'Light Blue', hex: '#ADD8E6', rgb: [173, 216, 230] }
    ];
    
    // Enhanced test sequence - each type tests a specific color blindness type
    const testSequence = [
        // Protanopia (red deficiency) test
        { 
            targetColor: 'Red', 
            confusionColors: ['Brown', 'Green'],
            type: 'protanopia'
        },
        // Deuteranopia (green deficiency) test
        {
            targetColor: 'Green',
            confusionColors: ['Brown', 'Red'],
            type: 'deuteranopia'
        },
        // Tritanopia (blue-yellow deficiency) test
        {
            targetColor: 'Blue',
            confusionColors: ['Purple', 'Green'],
            type: 'tritanopia'
        },
        {
            targetColor: 'Yellow',
            confusionColors: ['Pink', 'White'],
            type: 'tritanopia'
        },
        // Color shade discrimination test
        {
            targetColor: 'Light Green',
            confusionColors: ['Green', 'Light Blue'],
            type: 'shade_discrimination'
        },
        // More protanopia tests
        {
            targetColor: 'Orange',
            confusionColors: ['Green', 'Brown'],
            type: 'protanopia'
        },
        // More deuteranopia tests
        {
            targetColor: 'Purple',
            confusionColors: ['Blue', 'Pink'],
            type: 'deuteranopia'
        },
        // General color identification test
        {
            targetColor: 'Cyan',
            confusionColors: ['Light Blue', 'White'],
            type: 'general'
        }
    ];
    
    // Threshold values for response time (in milliseconds)
    const RESPONSE_TIME = {
        FAST: 1500,     // Less than 1.5 seconds
        MEDIUM: 3000,   // Between 1.5 and 3 seconds
        SLOW: 5000      // More than 3 seconds is considered slow
    };

    // Event Listeners
    startTestBtn.addEventListener('click', startTest);
    nextStepBtn.addEventListener('click', nextStep);
    retryTestBtn.addEventListener('click', resetTest);
    
    // Start the color blindness test
    function startTest() {
        startTestBtn.classList.add('hidden');
        currentStep = 0;
        userSelections = [];
        testResults = {
            protanopia: { correct: 0, incorrect: 0, avgResponseTime: 0, totalTests: 0 },
            deuteranopia: { correct: 0, incorrect: 0, avgResponseTime: 0, totalTests: 0 },
            tritanopia: { correct: 0, incorrect: 0, avgResponseTime: 0, totalTests: 0 },
            shade_discrimination: { correct: 0, incorrect: 0, avgResponseTime: 0, totalTests: 0 },
            general: { correct: 0, incorrect: 0, avgResponseTime: 0, totalTests: 0 }
        };
        updateProgressBar();
        displayColorTest();
    }
    
    // Function to display the current color test step
    function displayColorTest() {
        // Clear previous colors
        colorGrid.innerHTML = '';
        
        // Update step counter
        currentStepEl.textContent = currentStep + 1;
        totalStepsEl.textContent = TOTAL_STEPS;
        
        // Set the target color name
        const targetColor = testSequence[currentStep].targetColor;
        targetColorEl.textContent = targetColor;
        
        // Create array with target color and confusion colors
        let displayColors = [
            colors.find(c => c.name === targetColor)
        ];
        
        // Add confusion colors
        testSequence[currentStep].confusionColors.forEach(confusionColor => {
            const colorObj = colors.find(c => c.name === confusionColor);
            if (colorObj) displayColors.push(colorObj);
        });
        
        // Add some random colors to fill the grid
        const remainingColors = colors.filter(c => 
            c.name !== targetColor && 
            !testSequence[currentStep].confusionColors.includes(c.name)
        );
        
        // Shuffle and take some random colors
        const randomColors = shuffleArray([...remainingColors]).slice(0, 6 - displayColors.length);
        displayColors = displayColors.concat(randomColors);
        
        // Shuffle the display colors
        displayColors = shuffleArray(displayColors);
        
        // Start timing
        startTime = performance.now();
        
        // Create and append color boxes
        displayColors.forEach(color => {
            const colorBox = document.createElement('div');
            colorBox.className = 'color-box';
            colorBox.style.backgroundColor = color.hex;
            
            // For white, add a thin border to make it visible
            if (color.name === 'White') {
                colorBox.style.border = '1px solid #ddd';
            }
            
            colorBox.setAttribute('data-color', color.name);
            colorBox.addEventListener('click', () => selectColor(colorBox, color.name));
            
            colorGrid.appendChild(colorBox);
        });
        
        // Show the color grid
        nextStepBtn.classList.add('hidden');
    }
    
    // Function to handle color selection
    function selectColor(element, colorName) {
        // Calculate response time
        const responseTime = performance.now() - startTime;
        
        // Remove previous selection
        const previousSelected = document.querySelector('.color-box.selected');
        if (previousSelected) {
            previousSelected.classList.remove('selected');
        }
        
        // Mark this color as selected
        element.classList.add('selected');
        
        // Record the selection with response time
        const targetColor = testSequence[currentStep].targetColor;
        const testType = testSequence[currentStep].type;
        const isCorrect = targetColor === colorName;
        
        userSelections[currentStep] = {
            targetColor: targetColor,
            selectedColor: colorName,
            responseTime: responseTime,
            isCorrect: isCorrect,
            testType: testType
        };
        
        // Update test results
        if (testResults[testType]) {
            testResults[testType].totalTests++;
            if (isCorrect) {
                testResults[testType].correct++;
            } else {
                testResults[testType].incorrect++;
            }
            // Update running average of response time
            const oldAvg = testResults[testType].avgResponseTime;
            const oldCount = testResults[testType].totalTests - 1;
            const newAvg = oldCount > 0 
                ? (oldAvg * oldCount + responseTime) / testResults[testType].totalTests 
                : responseTime;
            testResults[testType].avgResponseTime = newAvg;
        }
        
        // Show the next button
        nextStepBtn.classList.remove('hidden');
    }
    
    // Move to the next test step
    function nextStep() {
        currentStep++;
        updateProgressBar();
        
        if (currentStep < TOTAL_STEPS) {
            displayColorTest();
        } else {
            showResults();
        }
    }
    
    // Update the progress bar
    function updateProgressBar() {
        const progress = (currentStep / TOTAL_STEPS) * 100;
        progressBar.style.width = `${progress}%`;
    }
    
    // Show test results
    function showResults() {
        testContainer.classList.add('hidden');
        resultsContainer.classList.remove('hidden');
        
        // Analyze the results
        const results = analyzeResults();
        
        // Update the UI with results
        resultMessage.className = 'result-message';
        resultMessage.classList.add(`result-${results.type.toLowerCase().replace('/', '-')}`);
        resultMessage.textContent = results.message;
        
        // Format average response time
        const avgResponseTime = (userSelections.reduce((sum, s) => sum + s.responseTime, 0) / userSelections.length / 1000).toFixed(2);
        
        // Show detailed results
        resultDetails.innerHTML = '';
        const detailsHtml = `
            <p>You correctly identified <strong>${results.correctAnswers} out of ${TOTAL_STEPS}</strong> colors.</p>
            <p>Your average response time was <strong>${avgResponseTime} seconds</strong>.</p>
            <p>Your color identification pattern suggests you may have: <strong>${results.type}</strong></p>
            <p>${results.explanation}</p>
            <div class="response-detail">
                <h4>Response Time Analysis:</h4>
                <p>${results.responseTimeAnalysis}</p>
            </div>
            <div class="confusion-detail">
                <h4>Confusion Pattern Analysis:</h4>
                <p>${results.confusionAnalysis}</p>
            </div>
        `;
        resultDetails.innerHTML = detailsHtml;
    }
    
    // Analyze the test results
    function analyzeResults() {
        // Count correct answers
        const correctAnswers = userSelections.filter(selection => selection.isCorrect).length;
        
        // Calculate average response time for all selections
        const totalResponseTime = userSelections.reduce((sum, s) => sum + s.responseTime, 0);
        const avgResponseTime = totalResponseTime / userSelections.length;
        
        // Analyze response time patterns
        const slowResponses = userSelections.filter(s => s.responseTime > RESPONSE_TIME.SLOW);
        const fastResponses = userSelections.filter(s => s.responseTime < RESPONSE_TIME.FAST);
        
        // Response time analysis for correct vs incorrect answers
        const correctResponseTimes = userSelections.filter(s => s.isCorrect).map(s => s.responseTime);
        const incorrectResponseTimes = userSelections.filter(s => !s.isCorrect).map(s => s.responseTime);
        
        const avgCorrectTime = correctResponseTimes.length > 0 
            ? correctResponseTimes.reduce((sum, t) => sum + t, 0) / correctResponseTimes.length 
            : 0;
            
        const avgIncorrectTime = incorrectResponseTimes.length > 0 
            ? incorrectResponseTimes.reduce((sum, t) => sum + t, 0) / incorrectResponseTimes.length 
            : 0;
        
        // Create a response time analysis message
        let responseTimeAnalysis = "";
        if (avgCorrectTime > 0 && avgIncorrectTime > 0) {
            const timeComparisonRatio = avgIncorrectTime / avgCorrectTime;
            if (timeComparisonRatio > 1.5) {
                responseTimeAnalysis = `You took significantly longer (${(timeComparisonRatio).toFixed(1)}x) to select colors you had difficulty with, indicating conscious color discrimination effort.`;
            } else if (timeComparisonRatio < 1.2) {
                responseTimeAnalysis = `Your response times were similar for both correct and incorrect selections, which can indicate genuine color blindness rather than uncertainty.`;
            }
        }
        
        if (responseTimeAnalysis === "") {
            if (avgResponseTime > RESPONSE_TIME.SLOW) {
                responseTimeAnalysis = `Your overall slow response time (${(avgResponseTime/1000).toFixed(2)}s) suggests you may have difficulty with color discrimination across the spectrum.`;
            } else if (avgResponseTime < RESPONSE_TIME.FAST) {
                responseTimeAnalysis = `Your quick response time (${(avgResponseTime/1000).toFixed(2)}s) suggests confident color identification.`;
            } else {
                responseTimeAnalysis = `Your response time (${(avgResponseTime/1000).toFixed(2)}s) is within the average range.`;
            }
        }
        
        // Specific color blindness type analysis
        const protanopiaScore = testResults.protanopia.correct / testResults.protanopia.totalTests;
        const deuteranopiaScore = testResults.deuteranopia.correct / testResults.deuteranopia.totalTests;
        const tritanopiaScore = testResults.tritanopia.correct / testResults.tritanopia.totalTests;
        const shadeScore = testResults.shade_discrimination.correct / testResults.shade_discrimination.totalTests;
        
        // Confusion pattern analysis
        let confusionAnalysis = "";
        
        // Analyze specific confusion patterns
        const redGreenConfusions = userSelections.filter(s => 
            (s.targetColor === 'Red' && (s.selectedColor === 'Green' || s.selectedColor === 'Brown')) ||
            (s.targetColor === 'Green' && (s.selectedColor === 'Red' || s.selectedColor === 'Brown'))
        ).length;
        
        const blueYellowConfusions = userSelections.filter(s => 
            (s.targetColor === 'Blue' && (s.selectedColor === 'Purple' || s.selectedColor === 'Green')) ||
            (s.targetColor === 'Yellow' && (s.selectedColor === 'Pink' || s.selectedColor === 'White'))
        ).length;
        
        // Generate detailed confusion analysis
        if (redGreenConfusions > 0) {
            confusionAnalysis += `You confused red and green (or similar colors) ${redGreenConfusions} times. `;
        }
        
        if (blueYellowConfusions > 0) {
            confusionAnalysis += `You confused blue and yellow (or similar colors) ${blueYellowConfusions} times. `;
        }
        
        if (confusionAnalysis === "") {
            confusionAnalysis = "No specific color confusion patterns were detected.";
        }
        
        // Determine the type of color blindness
        let type, message, explanation;
        
        if (correctAnswers >= 7 && avgResponseTime < RESPONSE_TIME.MEDIUM) {
            type = 'Normal';
            message = 'Your color vision appears to be normal!';
            explanation = 'You correctly identified most colors with quick response times, indicating normal color vision.';
        } else if (protanopiaScore <= 0.5 && deuteranopiaScore > 0.5) {
            type = 'Protanopia';
            message = 'You may have red-deficient color blindness (Protanopia).';
            explanation = 'You showed specific difficulty with red and similar colors, which is typical of protanopia.';
        } else if (deuteranopiaScore <= 0.5 && protanopiaScore > 0.5) {
            type = 'Deuteranopia';
            message = 'You may have green-deficient color blindness (Deuteranopia).';
            explanation = 'You showed specific difficulty with green and similar colors, which is typical of deuteranopia.';
        } else if (protanopiaScore <= 0.5 && deuteranopiaScore <= 0.5) {
            type = 'Deuteranopia/Protanopia';
            message = 'You may have red-green color blindness.';
            explanation = 'You showed difficulty distinguishing between red and green colors, which is typical of red-green color blindness.';
        } else if (tritanopiaScore <= 0.5) {
            type = 'Tritanopia';
            message = 'You may have blue-yellow color blindness.';
            explanation = 'You showed difficulty distinguishing between blue and yellow colors, which is typical of blue-yellow color blindness (tritanopia).';
        } else if (correctAnswers <= 3 || (avgResponseTime > RESPONSE_TIME.SLOW && correctAnswers < 6)) {
            type = 'Achromatopsia';
            message = 'You may have complete color blindness.';
            explanation = 'You had significant difficulty identifying most colors with slow response times, which may indicate complete color blindness (achromatopsia) or severe color vision deficiency.';
        } else {
            type = 'Mild Deficiency';
            message = 'You may have a mild color vision deficiency.';
            explanation = 'Your results show some difficulty with color identification but don\'t strongly indicate a specific type of color blindness.';
        }
        
        return {
            correctAnswers,
            type,
            message,
            explanation,
            responseTimeAnalysis,
            confusionAnalysis
        };
    }
    
    // Reset the test
    function resetTest() {
        resultsContainer.classList.add('hidden');
        testContainer.classList.remove('hidden');
        startTest();
    }
    
    // Utility function to shuffle an array
    function shuffleArray(array) {
        for (let i = array.length - 1; i > 0; i--) {
            const j = Math.floor(Math.random() * (i + 1));
            [array[i], array[j]] = [array[j], array[i]];
        }
        return array;
    }
});