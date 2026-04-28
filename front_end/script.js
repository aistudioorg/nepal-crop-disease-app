const API_BASE_URL = 'http://localhost:8000';

const uploadBox        = document.getElementById('uploadBox');
const imageInput       = document.getElementById('imageInput');
const browseBtn        = document.getElementById('browseBtn');
const previewContainer = document.getElementById('previewContainer');
const previewImage     = document.getElementById('previewImage');
const predictBtn       = document.getElementById('predictBtn');
const changeBtn        = document.getElementById('changeBtn');
const resultsSection   = document.getElementById('resultsSection');
const detailsSection   = document.getElementById('detailsSection');
const loading          = document.getElementById('loading');
const errorMessage     = document.getElementById('errorMessage');
const resetBtn         = document.getElementById('resetBtn');
const closeDetailsBtn  = document.getElementById('closeDetailsBtn');

let selectedFile = null, currentDisease = null;

browseBtn.addEventListener('click', (e) => { e.stopPropagation(); imageInput.click(); });
uploadBox.addEventListener('click', () => imageInput.click());
uploadBox.addEventListener('dragover', (e) => { e.preventDefault(); uploadBox.classList.add('dragover'); });
uploadBox.addEventListener('dragleave', () => uploadBox.classList.remove('dragover'));
uploadBox.addEventListener('drop', (e) => {
  e.preventDefault(); uploadBox.classList.remove('dragover');
  const f = e.dataTransfer.files[0];
  if (f && f.type.startsWith('image/')) handleFileSelect(f);
});
imageInput.addEventListener('change', (e) => { if (e.target.files[0]) handleFileSelect(e.target.files[0]); });

function handleFileSelect(file) {
  selectedFile = file;
  const reader = new FileReader();
  reader.onload = (e) => {
    previewImage.src = e.target.result;
    uploadBox.style.display = 'none';
    previewContainer.style.display = 'block';
    resultsSection.style.display = 'none';
    detailsSection.style.display = 'none';
    hideError();
  };
  reader.readAsDataURL(file);
}

changeBtn.addEventListener('click', () => {
  selectedFile = null; imageInput.value = '';
  previewContainer.style.display = 'none';
  uploadBox.style.display = 'block';
  resultsSection.style.display = 'none';
  detailsSection.style.display = 'none';
  hideError();
});

predictBtn.addEventListener('click', async () => {
  if (!selectedFile) { showError('Please select an image first.'); return; }
  await predict();
});

async function predict() {
  predictBtn.disabled = true;
  loading.style.display = 'flex';
  resultsSection.style.display = 'none';
  hideError();
  try {
    const fd = new FormData(); fd.append('file', selectedFile);
    const res = await fetch(`${API_BASE_URL}/predict`, { method: 'POST', body: fd });
    if (!res.ok) throw new Error();
    const data = await res.json();
    currentDisease = data.disease;
    showResults(data);
  } catch {
    showError('Cannot connect — make sure the API server is running at http://localhost:8000');
  } finally {
    predictBtn.disabled = false;
    loading.style.display = 'none';
  }
}

function showResults({ disease, confidence, all_probabilities }) {
  document.getElementById('diseaseName').textContent = disease;
  document.getElementById('confidenceText').textContent = (confidence * 100).toFixed(1) + '%';
  setTimeout(() => { document.getElementById('confFill').style.width = (confidence * 100) + '%'; }, 50);

  const list = document.getElementById('probabilitiesList');
  list.innerHTML = '';
  Object.entries(all_probabilities).sort((a,b) => b[1]-a[1]).forEach(([name, prob]) => {
    const el = document.createElement('div');
    el.className = 'prob-item' + (name === disease ? ' top' : '');
    el.innerHTML = `<span class="prob-name">${name}</span>
      <div class="prob-bar"><div class="prob-fill" style="width:0%"></div></div>
      <span class="prob-val">${(prob*100).toFixed(1)}%</span>`;
    list.appendChild(el);
    setTimeout(() => el.querySelector('.prob-fill').style.width = (prob*100)+'%', 80);
  });

  resultsSection.style.display = 'block';
  resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

document.querySelectorAll('.action-btn').forEach(btn => {
  btn.addEventListener('click', () => fetchDetails(currentDisease, btn.dataset.context));
});

async function fetchDetails(disease, context) {
  loading.style.display = 'flex';
  hideError();
  try {
    const res = await fetch(`${API_BASE_URL}/disease-details`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ disease, context })
    });
    if (!res.ok) throw new Error();
    const data = await res.json();
    const labels = { symptoms:'🔬 Symptoms', causes:'⚠️ Causes', solution_natural:'🌿 Natural Fix', solution_artificial:'🧪 Chemical Fix' };
    document.getElementById('detailsTitle').textContent = labels[context];
    document.getElementById('detailsContent').textContent = data.details;
    detailsSection.style.display = 'flex';
    setTimeout(() => detailsSection.scrollIntoView({ behavior: 'smooth' }), 60);
  } catch {
    showError('Could not load details. Please try again.');
  } finally {
    loading.style.display = 'none';
  }
}

closeDetailsBtn.addEventListener('click', () => { detailsSection.style.display = 'none'; });

resetBtn.addEventListener('click', () => {
  selectedFile = null; currentDisease = null; imageInput.value = '';
  previewContainer.style.display = 'none'; uploadBox.style.display = 'block';
  predictBtn.disabled = false;
  resultsSection.style.display = 'none'; detailsSection.style.display = 'none';
  document.getElementById('confFill').style.width = '0%';
  hideError();
  window.scrollTo({ top: 0, behavior: 'smooth' });
});

function showError(msg) { document.getElementById('errorText').textContent = msg; errorMessage.style.display = 'flex'; }
function hideError() { errorMessage.style.display = 'none'; }