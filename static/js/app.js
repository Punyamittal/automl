// Global state
let statusInterval = null;
let currentResults = null;

// Start pipeline
async function startPipeline() {
    try {
        const response = await fetch('/api/start', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            }
        });

        if (response.ok) {
            // Switch to timeline screen
            document.getElementById('start-screen').classList.remove('active');
            document.getElementById('timeline-screen').classList.add('active');
            
            // Start polling for status
            startStatusPolling();
        } else {
            alert('Failed to start pipeline');
        }
    } catch (error) {
        console.error('Error starting pipeline:', error);
        alert('Error starting pipeline: ' + error.message);
    }
}

// Poll for status updates
function startStatusPolling() {
    statusInterval = setInterval(async () => {
        try {
            const response = await fetch('/api/status');
            const status = await response.json();
            
            updateTimeline(status);
            updateLogs(status.logs || []);
            
            // Check if completed
            if (status.status === 'completed' || status.status === 'error') {
                clearInterval(statusInterval);
                
                if (status.status === 'completed') {
                    // Load results and show dashboard
                    await loadResults();
                    showDashboard();
                } else {
                    alert('Pipeline failed: ' + (status.error || 'Unknown error'));
                }
            }
        } catch (error) {
            console.error('Error polling status:', error);
        }
    }, 1000); // Poll every second
}

// Update timeline
function updateTimeline(status) {
    const timeline = document.getElementById('timeline');
    const stages = [
        { key: 'problem_mining', name: 'Problem Mining', icon: '🔍' },
        { key: 'feasibility', name: 'Feasibility Classification', icon: '✅' },
        { key: 'dataset_discovery', name: 'Dataset Discovery', icon: '📊' },
        { key: 'dataset_matching', name: 'Dataset Matching', icon: '🔗' },
        { key: 'training', name: 'Model Training', icon: '🤖' },
        { key: 'testing', name: 'Model Testing', icon: '🧪' },
        { key: 'code_generation', name: 'Code Generation', icon: '💻' },
        { key: 'github_publishing', name: 'GitHub Publishing', icon: '🚀' }
    ];
    
    timeline.innerHTML = '';
    
    stages.forEach(stage => {
        const progress = status.progress[stage.key] || {};
        const item = document.createElement('div');
        item.className = 'timeline-item';
        
        let statusClass = '';
        let statusText = 'Pending';
        
        if (progress.status === 'running') {
            statusClass = 'running';
            statusText = 'Running...';
        } else if (progress.status === 'completed') {
            statusClass = 'completed';
            statusText = 'Completed';
        } else if (progress.status === 'error') {
            statusClass = 'error';
            statusText = 'Error';
        }
        
        item.classList.add(statusClass);
        item.innerHTML = `
            <div class="stage-icon">${stage.icon}</div>
            <div class="stage-info">
                <div class="stage-name">${stage.name}</div>
                <div class="stage-status">${statusText}</div>
            </div>
        `;
        
        timeline.appendChild(item);
    });
}

// Update logs
function updateLogs(logs) {
    const logsContainer = document.getElementById('logs');
    logsContainer.innerHTML = '';
    
    logs.slice(-20).forEach(log => {
        const entry = document.createElement('div');
        entry.className = 'log-entry';
        entry.innerHTML = `<span class="timestamp">[${log.timestamp}]</span>${log.message}`;
        logsContainer.appendChild(entry);
    });
    
    // Auto-scroll to bottom
    logsContainer.scrollTop = logsContainer.scrollHeight;
}

// Load results
async function loadResults() {
    try {
        const response = await fetch('/api/results');
        if (response.ok) {
            currentResults = await response.json();
        }
    } catch (error) {
        console.error('Error loading results:', error);
    }
}

// Show dashboard
function showDashboard() {
    document.getElementById('timeline-screen').classList.remove('active');
    document.getElementById('dashboard-screen').classList.add('active');
    
    if (currentResults) {
        displayProblemInfo(currentResults);
        displayModelPerformance(currentResults);
        displayModelTester(currentResults);
        displayGeneratedFiles(currentResults);
        displayGitHubInfo(currentResults);
    }
}

// Display problem information
function displayProblemInfo(results) {
    const problemInfo = document.getElementById('problem-info');
    const stages = results.stages || {};
    
    // Try to get problem from results
    let problemTitle = 'Unknown Problem';
    let taskType = 'classification';
    
    // Extract from results if available
    if (stages.feasibility && stages.feasibility.feasible_problems > 0) {
        problemTitle = 'ML Feasible Problem Detected';
    }
    
    if (stages.training && stages.training.metrics) {
        taskType = stages.training.metrics.task_type || 'classification';
    }
    
    problemInfo.innerHTML = `
        <h4>${problemTitle}</h4>
        <p><strong>Task Type:</strong> ${taskType}</p>
        <p>The ML pipeline has identified and solved a machine learning problem using automated techniques.</p>
    `;
}

// Display model performance
function displayModelPerformance(results) {
    const performanceDiv = document.getElementById('model-performance');
    const stages = results.stages || {};
    const training = stages.training || {};
    const testing = stages.testing || {};
    const metrics = training.metrics || {};
    const testMetrics = testing.test_metrics || {};
    
    let html = '<div class="metrics-grid">';
    
    if (metrics.task_type === 'classification') {
        html += `
            <div class="metric-card">
                <div class="metric-label">Training Accuracy</div>
                <div class="metric-value">${(metrics.accuracy * 100).toFixed(1)}%</div>
            </div>
        `;
        
        if (testMetrics.accuracy) {
            html += `
                <div class="metric-card">
                    <div class="metric-label">Test Accuracy</div>
                    <div class="metric-value">${(testMetrics.accuracy * 100).toFixed(1)}%</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Precision</div>
                    <div class="metric-value">${(testMetrics.precision * 100).toFixed(1)}%</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">F1 Score</div>
                    <div class="metric-value">${(testMetrics.f1_score * 100).toFixed(1)}%</div>
                </div>
            `;
        }
    } else {
        html += `
            <div class="metric-card">
                <div class="metric-label">R² Score</div>
                <div class="metric-value">${(metrics.r2 || testMetrics.r2_score || 0).toFixed(3)}</div>
            </div>
        `;
        
        if (testMetrics.rmse) {
            html += `
                <div class="metric-card">
                    <div class="metric-label">RMSE</div>
                    <div class="metric-value">${testMetrics.rmse.toFixed(3)}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">MAE</div>
                    <div class="metric-value">${testMetrics.mae.toFixed(3)}</div>
                </div>
            `;
        }
    }
    
    html += '</div>';
    
    if (metrics.best_model) {
        html += `<p style="margin-top: 15px;"><strong>Best Model:</strong> ${metrics.best_model}</p>`;
    }
    
    performanceDiv.innerHTML = html;
}

// Display model tester
function displayModelTester(results) {
    const testerDiv = document.getElementById('model-tester');
    const inputsDiv = document.getElementById('feature-inputs');
    
    // Create feature inputs (default 10 features)
    let html = '<p>Enter feature values to test the model:</p>';
    for (let i = 1; i <= 10; i++) {
        html += `
            <div class="feature-input-group">
                <label>Feature ${i}:</label>
                <input type="number" id="feature_${i}" step="0.01" value="0" placeholder="Enter value">
            </div>
        `;
    }
    
    inputsDiv.innerHTML = html;
    document.getElementById('predict-btn').style.display = 'block';
}

// Make prediction
async function makePrediction() {
    const features = {};
    for (let i = 1; i <= 10; i++) {
        const value = document.getElementById(`feature_${i}`).value;
        features[`feature_${i}`] = parseFloat(value) || 0;
    }
    
    try {
        const response = await fetch('/api/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ features })
        });
        
        if (response.ok) {
            const result = await response.json();
            displayPrediction(result);
        } else {
            const error = await response.json();
            alert('Prediction failed: ' + (error.error || 'Unknown error'));
        }
    } catch (error) {
        console.error('Error making prediction:', error);
        alert('Error making prediction: ' + error.message);
    }
}

// Display prediction result
function displayPrediction(result) {
    const resultDiv = document.getElementById('prediction-result');
    let html = '<div class="prediction-result">';
    html += `<h4>Prediction Result</h4>`;
    html += `<div class="prediction-value">${result.prediction}</div>`;
    
    if (result.confidence) {
        html += `<div class="confidence">Confidence: ${(result.confidence * 100).toFixed(1)}%</div>`;
    }
    
    if (result.probabilities) {
        html += '<div style="margin-top: 15px;"><strong>Class Probabilities:</strong><ul>';
        for (const [cls, prob] of Object.entries(result.probabilities)) {
            html += `<li>${cls}: ${(prob * 100).toFixed(1)}%</li>`;
        }
        html += '</ul></div>';
    }
    
    html += '</div>';
    resultDiv.innerHTML = html;
    resultDiv.style.display = 'block';
}

// Display generated files
function displayGeneratedFiles(results) {
    const filesDiv = document.getElementById('generated-files');
    const codeGen = results.stages?.code_generation || {};
    
    if (codeGen.success && codeGen.files) {
        let html = '<ul class="file-list">';
        codeGen.files.forEach(file => {
            html += `<li>${file}</li>`;
        });
        html += '</ul>';
        
        if (codeGen.project_dir) {
            html += `<p><strong>Project Directory:</strong> ${codeGen.project_dir}</p>`;
        }
        
        filesDiv.innerHTML = html;
    } else {
        filesDiv.innerHTML = '<p>No files generated yet.</p>';
    }
}

// Display GitHub info
function displayGitHubInfo(results) {
    const githubDiv = document.getElementById('github-info');
    const github = results.stages?.github_publishing || {};
    
    if (github.success && github.repo_url) {
        githubDiv.innerHTML = `
            <p>Repository has been published to GitHub!</p>
            <a href="${github.repo_url}" target="_blank" class="github-link">
                View Repository →
            </a>
        `;
    } else if (github.skipped) {
        githubDiv.innerHTML = '<p>GitHub publishing was skipped (token not configured).</p>';
    } else {
        githubDiv.innerHTML = '<p>Repository not published yet.</p>';
    }
}
