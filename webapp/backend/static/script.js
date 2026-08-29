const preDrop   = document.getElementById('pre-drop');
const postDrop  = document.getElementById('post-drop');
const preInput  = document.getElementById('pre-input');
const postInput = document.getElementById('post-input');
const prePreview    = document.getElementById('pre-preview');
const postPreview   = document.getElementById('post-preview');
const prePlaceholder  = document.getElementById('pre-placeholder');
const postPlaceholder = document.getElementById('post-placeholder');
const btn = document.getElementById('analyze-btn');

let preFile  = null;
let postFile = null;

const COLORS = {
    'No Damage':    '#22c55e',
    'Minor Damage': '#eab308',
    'Major Damage': '#f97316',
    'Destroyed':    '#ef4444',
};

function setupDropZone(zone, input, preview, placeholder, type) {
    zone.addEventListener('click', () => input.click());
    input.addEventListener('change', e => {
        if (e.target.files[0]) loadFile(e.target.files[0], preview, placeholder, type);
    });
    zone.addEventListener('dragover', e => { e.preventDefault(); zone.classList.add('dragover'); });
    zone.addEventListener('dragleave', () => zone.classList.remove('dragover'));
    zone.addEventListener('drop', e => {
        e.preventDefault();
        zone.classList.remove('dragover');
        if (e.dataTransfer.files[0]) loadFile(e.dataTransfer.files[0], preview, placeholder, type);
    });
}

function loadFile(file, preview, placeholder, type) {
    if (!file || !file.type.startsWith('image/')) return;
    if (type === 'pre') preFile = file; else postFile = file;
    const reader = new FileReader();
    reader.onload = e => {
        preview.src = e.target.result;
        preview.hidden = false;
        placeholder.style.opacity = '0';
    };
    reader.readAsDataURL(file);
    btn.disabled = false;
    checkReady();
}

function checkReady() {
    btn.disabled = !(preFile && postFile);
}

setupDropZone(preDrop,  preInput,  prePreview,  prePlaceholder,  'pre');
setupDropZone(postDrop, postInput, postPreview, postPlaceholder, 'post');

function renderStats(stats) {
    // Top stat cards
    const row = document.getElementById('stats-row');
    row.innerHTML = '';

    const buildingPct = stats.building_coverage_pct;
    const breakdown   = stats.damage_breakdown || {};

    // Building coverage card
    row.innerHTML += `<div class="stat-card">
        <div class="stat-value">${buildingPct}%</div>
        <div class="stat-label">Building Coverage</div>
    </div>`;

    // One card per damage class
    Object.entries(breakdown).forEach(([name, info]) => {
        const color = COLORS[name] || '#94a3b8';
        row.innerHTML += `<div class="stat-card">
            <div class="stat-value" style="color:${color}">${info.pct}%</div>
            <div class="stat-label">${name}</div>
        </div>`;
    });

    // Breakdown bars
    const barsEl = document.getElementById('breakdown-bars');
    barsEl.innerHTML = '';
    if (Object.keys(breakdown).length === 0) {
        barsEl.innerHTML = '<p style="font-size:0.8rem;color:#94a3b8">No buildings detected.</p>';
        return;
    }
    Object.entries(breakdown).forEach(([name, info]) => {
        const color = COLORS[name] || '#94a3b8';
        barsEl.innerHTML += `<div class="bar-item">
            <div class="bar-header">
                <span class="bar-name">${name}</span>
                <span class="bar-pct">${info.pct}%</span>
            </div>
            <div class="bar-track">
                <div class="bar-fill" style="width:${info.pct}%;background:${color}"></div>
            </div>
        </div>`;
    });
}

btn.addEventListener('click', async () => {
    btn.disabled = true;
    document.getElementById('loading').hidden = false;
    document.getElementById('result-section').hidden = true;

    const fd = new FormData();
    fd.append('pre', preFile);
    fd.append('post', postFile);

    try {
        const res = await fetch('/api/predict', { method: 'POST', body: fd });
        if (!res.ok) throw new Error(`Server error: ${res.status} ${res.statusText}`);
        const data = await res.json();

        document.getElementById('result-img').src = 'data:image/png;base64,' + data.prediction_base64;

        if (data.stats) renderStats(data.stats);

        document.getElementById('result-section').hidden = false;
        document.getElementById('result-section').scrollIntoView({ behavior: 'smooth', block: 'start' });
    } catch (e) {
        alert('Error: ' + e.message);
    } finally {
        btn.disabled = !(preFile && postFile);
        document.getElementById('loading').hidden = true;
    }
});