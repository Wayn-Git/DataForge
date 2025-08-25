// --- CONFIG & GLOBAL VARS ---
const API_BASE_URL = 'http://localhost:8000';
let currentFile = null;

const OPERATIONS = {
    data_type_conversion: {
        title: "Optimize Data Types",
        configHTML: `
            <div class="form-group form-group-checkbox">
                <input type="checkbox" id="data-auto-detect" checked> <label for="data-auto-detect">Auto-detect data types</label>
            </div>
            <p class="op-description">Automatically convert columns to numeric, datetime, or other appropriate types.</p>
        `
    },
    text_cleaning: {
        title: "Clean Text Columns",
        configHTML: `
            <div class="form-group form-group-checkbox">
                <input type="checkbox" id="text-lower" checked> <label for="text-lower">Convert to lowercase</label>
            </div>
            <div class="form-group form-group-checkbox">
                <input type="checkbox" id="text-punct" checked> <label for="text-punct">Remove punctuation</label>
            </div>
            <div class="form-group form-group-checkbox">
                <input type="checkbox" id="text-whitespace" checked> <label for="text-whitespace">Strip whitespace</label>
            </div>
            <div class="form-group form-group-checkbox">
                <input type="checkbox" id="text-numbers"> <label for="text-numbers">Remove numbers</label>
            </div>
        `
    },
    datetime_parsing: {
        title: "Parse DateTime Columns",
        configHTML: `
            <div class="form-group">
                <label>Date Format (optional)</label>
                <input type="text" id="date-format" placeholder="e.g., %Y-%m-%d">
                <p class="op-description">Leave blank to auto-detect format.</p>
            </div>
            <div class="form-group form-group-checkbox">
                <input type="checkbox" id="datetime-extract-features"> <label for="datetime-extract-features">Extract date features (year, month, day)</label>
            </div>
        `
    },
    missing_values: {
        title: "Handle Missing Values",
        configHTML: `
            <div class="form-group">
                <label>Strategy</label>
                <select id="mv-strategy">
                    <option value="drop_rows">Drop rows with missing values</option>
                    <option value="drop_rows_threshold">Drop rows above threshold</option>
                    <option value="drop_columns">Drop columns with missing values</option>
                    <option value="drop_columns_threshold">Drop columns above threshold</option>
                    <option value="fill_mean">Fill with mean/mode</option>
                    <option value="fill_median">Fill with median/mode</option>
                    <option value="fill_mode">Fill with mode</option>
                    <option value="forward_fill">Forward fill</option>
                    <option value="backward_fill">Backward fill</option>
                </select>
            </div>
            <div class="form-group">
                <label>Threshold (0.0-1.0)</label>
                <input type="number" id="mv-threshold" value="0.5" min="0" max="1" step="0.1">
            </div>
        `
    },
    duplicates: {
        title: "Remove Duplicates",
        configHTML: `<p class="op-description">Removes all rows that are exact copies of another.</p>`
    },
    outliers: {
        title: "Handle Outliers",
        configHTML: `
            <div class="form-group">
                <label>Method</label>
                <select id="outlier-method">
                    <option value="iqr">IQR (Interquartile Range)</option>
                    <option value="zscore">Z-Score</option>
                    <option value="modified_zscore">Modified Z-Score</option>
                    <option value="isolation_forest">Isolation Forest</option>
                </select>
            </div>
            <div class="form-group">
                <label>Action</label>
                <select id="outlier-action">
                    <option value="remove">Remove outliers</option>
                    <option value="cap">Cap outliers</option>
                    <option value="transform">Transform outliers</option>
                </select>
            </div>
            <div class="form-group">
                <label>Threshold</label>
                <input type="number" id="outlier-threshold" value="3" min="0" step="0.1">
                <p class="op-description">Multiplier for IQR, or standard deviations for Z-score.</p>
            </div>
        `
    },
    typo_fix: {
        title: "Fix Typos",
        configHTML: `
            <div class="form-group">
                <label>Method</label>
                <select id="typo-method">
                    <option value="common_typos">Common Typos</option>
                    <option value="fuzzy_match">Fuzzy Match</option>
                    <option value="spell_check">Spell Check</option>
                </select>
            </div>
            <div class="form-group">
                <label>Similarity Threshold (0-100)</label>
                <input type="number" id="typo-threshold" value="80" min="0" max="100" step="1">
            </div>
        `
    },
    encoding: {
        title: "Categorical Encoding",
        configHTML: `
            <div class="form-group">
                <label>Method</label>
                <select id="encoding-method">
                    <option value="label">Label Encoding</option>
                    <option value="onehot">One-Hot Encoding</option>
                    <option value="target">Target Encoding</option>
                </select>
            </div>
            <div class="form-group form-group-checkbox">
                <input type="checkbox" id="encoding-drop-first"> <label for="encoding-drop-first">Drop first category (for one-hot)</label>
            </div>
        `
    },
    normalization: {
        title: "Normalize Data",
        configHTML: `
            <div class="form-group">
                <label>Method</label>
                <select id="normalization-method">
                    <option value="standard">Standard Scaling</option>
                    <option value="minmax">Min-Max Scaling</option>
                    <option value="robust">Robust Scaling</option>
                    <option value="normalize">Normalize</option>
                </select>
            </div>
        `
    }
};

// --- INITIALIZATION ---
document.addEventListener('DOMContentLoaded', () => {
    setTimeout(() => {
        initializeApp();
    }, 1500);
});

async function initializeApp() {
    setupEventListeners();
    populateOperations();
    await checkBackendConnection();
    
    const splash = document.getElementById('splash-screen');
    const app = document.getElementById('app-container');
    splash.style.opacity = '0';
    setTimeout(() => {
        splash.style.display = 'none';
        app.style.display = 'block';
    }, 500);
}

function setupEventListeners() {
    const uploadArea = document.getElementById('file-upload-area');
    const fileInput = document.getElementById('file-input');
    uploadArea.addEventListener('click', () => fileInput.click());
    uploadArea.addEventListener('dragover', e => { e.preventDefault(); uploadArea.classList.add('dragover'); });
    uploadArea.addEventListener('dragleave', () => uploadArea.classList.remove('dragover'));
    uploadArea.addEventListener('drop', handleFileDrop);
    fileInput.addEventListener('change', handleFileSelect);
    document.getElementById('run-pipeline-btn').addEventListener('click', runPipeline);
}

// --- BACKEND CONNECTION & UI HELPERS ---
async function checkBackendConnection() {
    try {
        const response = await fetch(`${API_BASE_URL}/health`);
        if (!response.ok) throw new Error();
        const data = await response.json();
        updateConnectionStatus(data.status === 'healthy');
    } catch {
        updateConnectionStatus(false);
    }
}

function updateConnectionStatus(isHealthy) {
    const statusText = document.getElementById('connection-status');
    const light = document.querySelector('#status-indicator .light');
    statusText.textContent = isHealthy ? 'Connected' : 'Disconnected';
    light.classList.toggle('success', isHealthy);
    light.classList.toggle('error', !isHealthy);
}

function handleFileDrop(e) {
    e.preventDefault();
    document.getElementById('file-upload-area').classList.remove('dragover');
    if (e.dataTransfer.files.length) processFile(e.dataTransfer.files[0]);
}

function handleFileSelect(e) {
    if (e.target.files.length) processFile(e.target.files[0]);
}

async function processFile(file) {
    if (!file || !file.name.endsWith('.csv')) return;
    document.getElementById('file-name').textContent = file.name;
    document.getElementById('file-size').textContent = `${(file.size / 1024).toFixed(1)} KB`;
    document.getElementById('file-info-bar').classList.remove('hidden');

    const formData = new FormData();
    formData.append('file', file);

    try {
        const response = await fetch(`${API_BASE_URL}/upload`, { method: 'POST', body: formData });
        const data = await response.json();
        if (data.status !== 'success') throw new Error(data.error);
        
        currentFile = data.file_path;
        showDatasetInfo(data.dataset_info);
        document.getElementById('config-section').classList.add('active');
        document.getElementById('run-pipeline-btn').disabled = false;
    } catch (err) { 
        console.error('Upload error:', err); 
        alert('Upload failed: ' + err.message);
    }
}

function showDatasetInfo(info) {
    const container = document.getElementById('dataset-info-grid');
    
    // Calculate total missing values more accurately
    let totalMissing = 0;
    if (info.missing_values && typeof info.missing_values === 'object') {
        totalMissing = Object.values(info.missing_values).reduce((sum, count) => {
            const numCount = parseInt(count) || 0;
            return sum + numCount;
        }, 0);
    } else if (info.total_missing) {
        totalMissing = parseInt(info.total_missing) || 0;
    }
    
    // Also calculate missing percentage
    const totalCells = info.shape[0] * info.shape[1];
    const missingPercentage = totalCells > 0 ? ((totalMissing / totalCells) * 100).toFixed(1) : '0.0';
    
    container.innerHTML = `
        <div class="info-card">
            <div class="value">${info.shape[0].toLocaleString()}</div>
            <div class="label">Rows</div>
        </div>
        <div class="info-card">
            <div class="value">${info.shape[1]}</div>
            <div class="label">Columns</div>
        </div>
        <div class="info-card ${totalMissing > 0 ? 'warning' : ''}">
            <div class="value">${totalMissing.toLocaleString()}</div>
            <div class="label">Missing (${missingPercentage}%)</div>
        </div>
        <div class="info-card ${info.duplicates > 0 ? 'warning' : ''}">
            <div class="value">${info.duplicates}</div>
            <div class="label">Duplicates</div>
        </div>
    `;
    
    // Show detailed missing values breakdown if available
    if (info.missing_values && typeof info.missing_values === 'object' && totalMissing > 0) {
        const missingColumns = Object.entries(info.missing_values)
            .filter(([col, count]) => parseInt(count) > 0)
            .sort(([,a], [,b]) => parseInt(b) - parseInt(a))
            .slice(0, 5); // Show top 5 columns with missing values
        
        if (missingColumns.length > 0) {
            const missingBreakdown = missingColumns
                .map(([col, count]) => `${col}: ${count}`)
                .join(', ');
            
            const breakdownElement = document.createElement('div');
            breakdownElement.className = 'missing-breakdown';
            breakdownElement.innerHTML = `
                <p><strong>Top columns with missing values:</strong></p>
                <p class="missing-details">${missingBreakdown}${missingColumns.length === 5 ? '...' : ''}</p>
            `;
            container.appendChild(breakdownElement);
        }
    }
    
    console.log('Dataset info processed:', {
        totalMissing,
        missingPercentage,
        missingValues: info.missing_values
    });
}

function populateOperations() {
    const container = document.getElementById('operations-container');
    container.innerHTML = Object.entries(OPERATIONS).map(([id, op]) => `
        <div class="op-item" id="op-item-${id}">
            <div class="op-header">
                <span class="op-title">${op.title}</span>
                <label class="switch"><input type="checkbox" id="toggle-${id}" data-op-id="${id}" checked><span class="slider"></span></label>
            </div>
            <div class="op-config">${op.configHTML}</div>
        </div>`).join('');

    container.querySelectorAll('.switch input').forEach(toggle => {
        toggle.addEventListener('change', () => {
            document.getElementById(`op-item-${toggle.dataset.opId}`).classList.toggle('active', toggle.checked);
        });
        document.getElementById(`op-item-${toggle.dataset.opId}`).classList.toggle('active', toggle.checked);
    });
}

// --- PIPELINE EXECUTION ---
function buildOperationsConfig() {
    const config = {};
    document.querySelectorAll('.operations-container .switch input:checked').forEach(toggle => {
        const opId = toggle.dataset.opId;
        config[opId] = { enabled: true };
        
        // Handle missing values
        if (opId === 'missing_values') {
            config[opId].strategy = document.getElementById('mv-strategy').value;
            config[opId].threshold = parseFloat(document.getElementById('mv-threshold').value);
        }
        
        // Handle duplicates (no additional config needed)
        
        // Handle outliers
        if (opId === 'outliers') {
            config[opId].method = document.getElementById('outlier-method').value;
            config[opId].action = document.getElementById('outlier-action').value;
            config[opId].threshold = parseFloat(document.getElementById('outlier-threshold').value);
        }
        
        // Handle data type conversion
        if (opId === 'data_type_conversion') {
            config[opId].auto_detect = document.getElementById('data-auto-detect').checked;
            config[opId].type_mapping = {}; // Empty for now, can be extended
            config[opId].errors = "coerce";
        }
        
        // Handle text cleaning
        if (opId === 'text_cleaning') {
            const operations = [];
            if (document.getElementById('text-lower').checked) operations.push('lowercase');
            if (document.getElementById('text-punct').checked) operations.push('remove_punctuation');
            if (document.getElementById('text-whitespace').checked) operations.push('remove_whitespace');
            if (document.getElementById('text-numbers').checked) operations.push('remove_numbers');
            
            config[opId].operations = operations;
            config[opId].columns = null; // Apply to all text columns
        }
        
        // Handle datetime parsing
        if (opId === 'datetime_parsing') {
            config[opId].date_format = document.getElementById('date-format').value || null;
            config[opId].auto_detect = true;
            config[opId].extract_features = document.getElementById('datetime-extract-features').checked;
            config[opId].errors = "coerce";
        }
        
        // Handle encoding
        if (opId === 'encoding') {
            config[opId].method = document.getElementById('encoding-method').value;
            config[opId].drop_first = document.getElementById('encoding-drop-first').checked;
        }
        
        // Handle typo fix
        if (opId === 'typo_fix') {
            config[opId].method = document.getElementById('typo-method').value;
            config[opId].similarity_threshold = parseInt(document.getElementById('typo-threshold').value);
        }
        
        // Handle normalization
        if (opId === 'normalization') {
            config[opId].method = document.getElementById('normalization-method').value;
        }
    });
    
    console.log('Built operations config:', config);
    return config;
}

async function runPipeline() {
    if (!currentFile) {
        alert('Please upload a file first');
        return;
    }
    
    const operationsConfig = buildOperationsConfig();
    if (Object.keys(operationsConfig).length === 0) {
        alert('Please select at least one operation');
        return;
    }

    document.getElementById('results-section').classList.add('active');
    renderProgressUI(operationsConfig);

    const formData = new FormData();
    formData.append('file_path', currentFile);
    formData.append('operations', JSON.stringify(operationsConfig));
    
    console.log('Sending request with operations:', operationsConfig);
    
    await animateProgress(operationsConfig);

    try {
        const response = await fetch(`${API_BASE_URL}/clean-data`, { method: 'POST', body: formData });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        console.log('Pipeline response:', data);
        
        if (data.status !== 'success') {
            throw new Error(data.detail || 'Unknown error occurred');
        }
        
        Object.keys(operationsConfig).forEach(opId => updateStep(opId, 100, 'Complete!', 'success'));
        showResults(data);

    } catch (error) {
        console.error('Pipeline error:', error);
        const firstOpId = Object.keys(operationsConfig)[0];
        updateStep(firstOpId, 100, `Error: ${error.message}`, 'error');
        alert('Pipeline failed: ' + error.message);
    }
}

function renderProgressUI(operationsConfig) {
    const container = document.getElementById('pipeline-progress-container');
    const stepsHtml = Object.keys(operationsConfig).map(id => `
        <li class="progress-item" data-step-id="${id}">
            <div class="progress-header">
                <div class="progress-icon"><i class="fas fa-hourglass-start"></i></div>
                <div class="progress-text">${OPERATIONS[id].title}</div>
            </div>
            <div class="progress-bar-container"><div class="progress-bar"></div></div>
            <div class="progress-log">Waiting...</div>
        </li>`).join('');
    container.innerHTML = `<ul class="progress-list">${stepsHtml}</ul>`;
}

async function animateProgress(operationsConfig) {
    for (const opId in operationsConfig) {
        updateStep(opId, 0, 'Starting...', 'in-progress', '<i class="fas fa-cogs"></i>');
        for (let i = 0; i <= 100; i += Math.floor(Math.random() * 15) + 5) {
            await new Promise(res => setTimeout(res, 100 + Math.random() * 200));
            updateStep(opId, Math.min(i, 99), `Processing... ${i}%`);
        }
    }
}

function updateStep(stepId, percentage, logMessage, status, icon) {
    const item = document.querySelector(`.progress-item[data-step-id="${stepId}"]`);
    if (!item) return;

    if (percentage !== null) item.querySelector('.progress-bar').style.width = `${percentage}%`;
    if (logMessage) item.querySelector('.progress-log').textContent = logMessage;
    if (icon) item.querySelector('.progress-icon').innerHTML = icon;
    
    if (status) {
        item.classList.remove('success', 'error', 'in-progress');
        item.classList.add(status);
        const iconMap = {
            success: '<i class="fas fa-check-circle"></i>',
            error: '<i class="fas fa-exclamation-circle"></i>'
        };
        if (iconMap[status]) item.querySelector('.progress-icon').innerHTML = iconMap[status];
    }
}

function showResults(data) {
    const container = document.getElementById('final-results-container');
    const [finalRows, finalCols] = data.result.final_data_shape;
    
    container.innerHTML = `
        <div class="results-summary">
            <h3>Cleaning Complete</h3>
            <div id="dataset-info-grid">
                <div class="info-card"><div class="value">${finalRows.toLocaleString()}</div><div class="label">Final Rows</div></div>
                <div class="info-card"><div class="value">${finalCols}</div><div class="label">Final Columns</div></div>
            </div>
        </div>
        <a href="${API_BASE_URL}${data.download_url}" class="download-button" download>
            <i class="fas fa-download"></i> Download Cleaned File
        </a>`;
    container.classList.remove('hidden');
}

function renderVisualizations() {
    // --- Sample Data (Replace with your real data) ---
    // This data could represent the count of missing values per column.
    const columns = ['Age', 'Salary', 'Department', 'Rating'];
    const missingValuesBefore = [50, 25, 10, 5]; // Example: 50 missing 'Age' values
    const missingValuesAfter = [0, 0, 2, 0];   // Example: After cleaning, only 2 missing values remain

    // --- Chart Configuration ---
    const chartConfig = (data) => ({
        type: 'bar',
        data: {
            labels: columns,
            datasets: [{
                label: 'Missing Values Count',
                data: data,
                backgroundColor: 'hsla(var(--accent-hsla), 0.7)',
                borderColor: 'hsl(var(--accent-hsla))',
                borderWidth: 1
            }]
        },
        options: {
            scales: {
                y: {
                    beginAtZero: true
                }
            },
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: false // Hide legend for a cleaner look
                }
            }
        }
    });

    // --- Render Charts ---
    const beforeCtx = document.getElementById('beforeChart').getContext('2d');
    new Chart(beforeCtx, chartConfig(missingValuesBefore));

    const afterCtx = document.getElementById('afterChart').getContext('2d');
    new Chart(afterCtx, chartConfig(missingValuesAfter));
    
    // Make the new step active
    document.getElementById('step5').classList.add('active');
}