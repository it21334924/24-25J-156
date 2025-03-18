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
    
    // Color configurations for the test
    const colors = [
        { name: 'Red', hex: '#FF0000' },
        { name: 'Green', hex: '#00FF00' },
        { name: 'Blue', hex: '#0000FF' },
        { name: 'Yellow', hex: '#FFFF00' },
        { name: 'Cyan', hex: '#00FFFF' },
        { name: 'Magenta', hex: '#FF00FF' },
        { name: 'White', hex: '#FFFFFF' },
        { name: 'Black', hex: '#000000' }
    ];
    
    // Test sequence - which colors to test in which order
    // This will test for different color blindness types
    const testSequence = [
        { targetColor: 'Red', confusionColors: ['Green'] },
        { targetColor: 'Green', confusionColors: ['Red'] },
        { targetColor: 'Blue', confusionColors: ['Yellow'] },
        { targetColor: 'Yellow', confusionColors: ['Blue'] },
        { targetColor: 'Cyan', confusionColors: ['Magenta'] },
        { targetColor: 'Magenta', confusionColors: ['Cyan'] },
        { targetColor: 'White', confusionColors: [] },
        { targetColor: 'Black', confusionColors: [] }
    ];
    
    // Event Listeners
    startTestBtn.addEventListener('click', startTest);
    nextStepBtn.addEventListener('click', nextStep);
    retryTestBtn.addEventListener('click', resetTest);
    
    // Start the color blindness test
    function startTest() {
        startTestBtn.classList.add('hidden');
        currentStep = 0;
        userSelections = [];
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
        
        // Shuffle the colors for display
        const shuffledColors = shuffleArray([...colors]);
        
        // Create and append color boxes
        shuffledColors.forEach(color => {
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
        // Remove previous selection
        const previousSelected = document.querySelector('.color-box.selected');
        if (previousSelected) {
            previousSelected.classList.remove('selected');
        }
        
        // Mark this color as selected
        element.classList.add('selected');
        
        // Record the selection
        userSelections[currentStep] = {
            targetColor: testSequence[currentStep].targetColor,
            selectedColor: colorName
        };
        
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
        resultMessage.classList.add(`result-${results.type.toLowerCase()}`);
        resultMessage.textContent = results.message;
        
        // Show detailed results
        resultDetails.innerHTML = '';
        const detailsHtml = `
            <p>You correctly identified <strong>${results.correctAnswers} out of ${TOTAL_STEPS}</strong> colors.</p>
            <p>Your color identification pattern suggests you may have: <strong>${results.type}</strong></p>
            <p>${results.explanation}</p>
        `;
        resultDetails.innerHTML = detailsHtml;
    }
    
    // Analyze the test results
    function analyzeResults() {
        // Count correct answers
        const correctAnswers = userSelections.filter(
            selection => selection.targetColor === selection.selectedColor
        ).length;
        
        // Count specific confusion patterns
        const redGreenConfusions = userSelections.filter(selection => 
            (selection.targetColor === 'Red' && selection.selectedColor === 'Green') ||
            (selection.targetColor === 'Green' && selection.selectedColor === 'Red')
        ).length;
        
        const blueYellowConfusions = userSelections.filter(selection => 
            (selection.targetColor === 'Blue' && selection.selectedColor === 'Yellow') ||
            (selection.targetColor === 'Yellow' && selection.selectedColor === 'Blue')
        ).length;
        
        // Determine the type of color blindness
        let type, message, explanation;
        
        if (correctAnswers >= 7) {
            type = 'Normal';
            message = 'Your color vision appears to be normal!';
            explanation = 'You correctly identified most colors, indicating normal color vision.';
        } else if (redGreenConfusions >= 2 && blueYellowConfusions < 2) {
            type = 'Deuteranopia/Protanopia';
            message = 'You may have red-green color blindness.';
            explanation = 'You showed difficulty distinguishing between red and green colors, which is typical of red-green color blindness (deuteranopia or protanopia).';
        } else if (blueYellowConfusions >= 2 && redGreenConfusions < 2) {
            type = 'Tritanopia';
            message = 'You may have blue-yellow color blindness.';
            explanation = 'You showed difficulty distinguishing between blue and yellow colors, which is typical of blue-yellow color blindness (tritanopia).';
        } else if (correctAnswers <= 3) {
            type = 'Achromatopsia';
            message = 'You may have complete color blindness.';
            explanation = 'You had difficulty identifying most colors, which may indicate complete color blindness (achromatopsia) or severe color vision deficiency.';
        } else {
            type = 'Mild Deficiency';
            message = 'You may have a mild color vision deficiency.';
            explanation = 'Your results show some difficulty with color identification but don\'t strongly indicate a specific type of color blindness.';
        }
        
        return {
            correctAnswers,
            type,
            message,
            explanation
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