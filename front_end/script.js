const API_BASE_URL = 'http://localhost:8000';

const uploadForm = document.getElementById('uploadForm');
const uploadBox = document.getElementById('uploadBox');
const imageInput = document.getElementById('imageInput');
const previewContainer = document.getElementById('previewContainer');
const previewImage = document.getElementById('previewImage');
const predictBtn = document.getElementById('predictBtn');
const changeBtn = document.getElementById('changeBtn');
const resultsSection = document.getElementById('resultsSection');
const loading = document.getElementById('loading');
const errorMessage = document.getElementById('errorMessage');
const resetBtn = document.getElementById('resetBtn');

let selectedFile = null;

uploadBox.addEventListener('click', () => imageInput.click());

uploadBox.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadBox.classList.add('dragover');
});

uploadBox.addEventListener('dragleave', () => uploadBox.classList.remove('dragover'));

uploadBox.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadBox.classList.remove('dragover');
    const files = e.dataTransfer.files;
    if (files.length > 0 && files[0].type.startsWith('image/')) {
        handleFileSelect(files[0]);
    }
});

imageInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) handleFileSelect(e.target.files[0]);
});

function handleFileSelect(file) {
    selectedFile = file;
    const reader = new FileReader();
    reader.onload = (e) => {
        previewImage.src = e.target.result;
        uploadBox.style.display = 'none';
        previewContainer.style.display = 'block';
        resultsSection.style.display = 'none';
        hideError();
    };
    reader.readAsDataURL(file);
}

changeBtn.addEventListener('click', (e) => {
    e.preventDefault();
    selectedFile = null;
    imageInput.value = '';
    previewContainer.style.display = 'none';
    uploadBox.style.display = 'block';
    resultsSection.style.display = 'none';
    hideError();
});

uploadForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    if (!selectedFile) {
        showError('Please select an image');
        return;
    }
    await predictDisease();
});

async function predictDisease() {
    predictBtn.disabled = true;
    loading.style.display = 'flex';
    resultsSection.style.display = 'none';
    hideError();
    
    try {
        const formData = new FormData();
        formData.append('file', selectedFile);
        const response = await fetch(`${API_BASE_URL}/predict`, { method: 'POST', body: formData });
        if (!response.ok) throw new Error('Prediction failed');
        
        const result = await response.json();
        displayResults(result);
    } catch (error) {
        showError('Error: Make sure API server is running at http://localhost:8000');
    } finally {
        predictBtn.disabled = false;
        loading.style.display = 'none';
    }
}

function displayResults(result) {
    const disease = result.disease;
    const confidence = result.confidence;
    const probabilities = result.all_probabilities;
    
    document.getElementById('diseaseName').textContent = disease;
    
    const confidenceFill = document.getElementById('confidenceFill');
    confidenceFill.style.width = (confidence * 100) + '%';
    document.getElementById('confidenceText').textContent = (confidence * 100).toFixed(2) + '%';
    
    const probabilitiesList = document.getElementById('probabilitiesList');
    probabilitiesList.innerHTML = '';
    const sortedProbs = Object.entries(probabilities).sort((a, b) => b[1] - a[1]);
    
    sortedProbs.forEach((entry) => {
        const [diseaseName, probability] = entry;
        const item = document.createElement('div');
        item.className = 'probability-item';
        if (diseaseName === disease) {
            item.style.background = 'rgba(16, 185, 129, 0.1)';
            item.style.borderColor = 'var(--primary-color)';
        }
        
        const label = document.createElement('div');
        label.className = 'probability-label';
        label.textContent = diseaseName;
        
        const bar = document.createElement('div');
        bar.className = 'probability-bar-small';
        const fill = document.createElement('div');
        fill.className = 'probability-bar-fill';
        fill.style.width = (probability * 100) + '%';
        bar.appendChild(fill);
        
        const value = document.createElement('div');
        value.className = 'probability-value';
        value.textContent = (probability * 100).toFixed(1) + '%';
        
        item.appendChild(label);
        item.appendChild(bar);
        item.appendChild(value);
        probabilitiesList.appendChild(item);
    });
    
    resultsSection.style.display = 'block';
}

resetBtn.addEventListener('click', () => {
    selectedFile = null;
    imageInput.value = '';
    previewContainer.style.display = 'none';
    uploadBox.style.display = 'block';
    predictBtn.disabled = false;
    resultsSection.style.display = 'none';
    uploadBox.classList.remove('dragover');
    hideError();
});

function showError(message) {
    errorMessage.textContent = message;
    errorMessage.style.display = 'block';
}

function hideError() {
    errorMessage.style.display = 'none';
}
