'use strict';

// ── Constants ─────────────────────────────────────────────────────────────
const DUCT_COLORS = {
  supply:  '#1565C0',
  return:  '#C62828',
  exhaust: '#E65100',
};
const PRESSURE_COLORS = {
  Low:    '#15803D',
  Medium: '#C2410C',
  High:   '#B91C1C',
};
const STEP_LABELS = [
  'Ingesting PDF and extracting text…',
  'Detecting duct segments via AI vision…',
  'Matching dimension and CFM labels…',
  'Rendering annotated overlay…',
  'Running quality review…',
];

// ── State ─────────────────────────────────────────────────────────────────
let sessionId   = null;
let pollTimer   = null;
let stepTimer   = null;
let currentStep = 0;

// ── View helpers ──────────────────────────────────────────────────────────
function showView(id) {
  document.querySelectorAll('.view').forEach(v => {
    v.classList.toggle('active', v.id === id);
  });
}

function resetToUpload() {
  clearInterval(pollTimer);
  clearInterval(stepTimer);
  sessionId   = null;
  currentStep = 0;
  document.getElementById('file-input').value = '';
  document.getElementById('overlay').innerHTML = '';
  document.getElementById('duct-popup').style.display = 'none';
  showView('view-upload');
}

// ── Upload zone drag-and-drop ─────────────────────────────────────────────
(function wireDragDrop() {
  const zone = document.getElementById('upload-zone');
  zone.addEventListener('dragover', e => { e.preventDefault(); zone.classList.add('drag-over'); });
  zone.addEventListener('dragleave', () => zone.classList.remove('drag-over'));
  zone.addEventListener('drop', e => {
    e.preventDefault();
    zone.classList.remove('drag-over');
    const file = e.dataTransfer.files[0];
    if (file) startUpload(file);
  });
  zone.addEventListener('click', () => document.getElementById('file-input').click());

  document.getElementById('file-input').addEventListener('change', e => {
    if (e.target.files[0]) startUpload(e.target.files[0]);
  });
})();

// ── Upload & pipeline ─────────────────────────────────────────────────────
async function startUpload(file) {
  showView('view-processing');
  startStepAnimation();

  const body = new FormData();
  body.append('file', file);

  try {
    const res  = await fetch('/api/process', { method: 'POST', body });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || 'Upload failed');
    sessionId = data.session_id;
    pollStatus();
  } catch (err) {
    showError(err.message);
  }
}

function startStepAnimation() {
  currentStep = 0;
  updateStep(0);
  stepTimer = setInterval(() => {
    if (currentStep < STEP_LABELS.length - 1) {
      markStepDone(currentStep);
      currentStep++;
      updateStep(currentStep);
    }
  }, 18000); // advance roughly every 18 s (pipeline takes ~2-8 min)
}

function updateStep(idx) {
  document.querySelectorAll('.step').forEach((el, i) => {
    el.classList.remove('active', 'done');
    if (i < idx)  el.classList.add('done');
    if (i === idx) el.classList.add('active');
  });
  const label = document.getElementById('processing-label');
  if (label) label.textContent = STEP_LABELS[idx] || 'Processing…';
}

function markStepDone(idx) {
  const el = document.getElementById(`step-${idx}`);
  if (el) { el.classList.remove('active'); el.classList.add('done'); }
}

function pollStatus() {
  pollTimer = setInterval(async () => {
    try {
      const res  = await fetch(`/api/session/${sessionId}/status`);
      const data = await res.json();
      if (data.status === 'complete') {
        clearInterval(pollTimer);
        clearInterval(stepTimer);
        loadResults();
      } else if (data.status === 'error') {
        clearInterval(pollTimer);
        clearInterval(stepTimer);
        showError(data.error || 'Pipeline failed');
      }
    } catch (_) { /* network hiccup — keep polling */ }
  }, 4000);
}

async function loadResults() {
  try {
    const res  = await fetch(`/api/session/${sessionId}/result`);
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || 'Could not load results');

    const img = document.getElementById('drawing-img');
    img.src = `/api/session/${sessionId}/image`;

    img.onload = () => {
      buildOverlay(data.img_width, data.img_height, data.measurements);
      buildStats(data.measurements, data.summary);
      showView('view-results');
    };
    img.onerror = () => showError('Could not load annotated image.');
  } catch (err) {
    showError(err.message);
  }
}

// ── SVG overlay ───────────────────────────────────────────────────────────
function buildOverlay(imgW, imgH, segments) {
  const svg = document.getElementById('overlay');
  svg.setAttribute('viewBox', `0 0 ${imgW} ${imgH}`);
  svg.innerHTML = '';

  const ns = 'http://www.w3.org/2000/svg';

  segments.forEach(seg => {
    if (!seg.polygon || seg.polygon.length < 3) return;

    const color  = DUCT_COLORS[seg.type] || '#888888';
    const points = seg.polygon.map(p => `${p[0]},${p[1]}`).join(' ');

    const poly = document.createElementNS(ns, 'polygon');
    poly.setAttribute('points', points);
    poly.setAttribute('fill',         color + '33');
    poly.setAttribute('stroke',       color);
    poly.setAttribute('stroke-width', '8');
    poly.classList.add('duct-polygon');

    poly.addEventListener('mouseenter', () => poly.setAttribute('fill', color + '66'));
    poly.addEventListener('mouseleave', () => poly.setAttribute('fill', color + '33'));
    poly.addEventListener('click',      e  => showPopup(seg, e));

    svg.appendChild(poly);
  });
}

// ── Popup ─────────────────────────────────────────────────────────────────
function showPopup(seg, event) {
  const popup    = document.getElementById('duct-popup');
  const color    = DUCT_COLORS[seg.type]         || '#888888';
  const pClass   = seg.pressure_class            || 'Low';
  const pColor   = PRESSURE_COLORS[pClass]       || '#15803D';
  const dimText  = formatDimension(seg);
  const typeText = seg.type
    ? seg.type.charAt(0).toUpperCase() + seg.type.slice(1)
    : 'Unknown';

  popup.innerHTML = `
    <div class="popup-header" style="background:${color}">
      <span class="popup-type">${typeText} Duct</span>
      <button class="popup-close" onclick="closePopup()">×</button>
    </div>
    <div class="popup-body">
      <div class="popup-row">
        <span class="popup-label">Dimension</span>
        <span class="popup-value">${dimText}</span>
      </div>
      <div class="popup-row">
        <span class="popup-label">Pressure Class</span>
        <span class="pressure-badge-inline" style="background:${pColor}">${pClass} Pressure</span>
      </div>
      ${seg.cfm ? `
      <div class="popup-row">
        <span class="popup-label">Airflow</span>
        <span class="popup-value">${seg.cfm} CFM</span>
      </div>` : ''}
      ${seg.length_ft ? `
      <div class="popup-row">
        <span class="popup-label">Length</span>
        <span class="popup-value">${seg.length_ft} ft</span>
      </div>` : ''}
    </div>
  `;

  popup.style.display = 'block';

  // Position relative to drawing container, keeping popup inside viewport
  const container = document.getElementById('drawing-container');
  const cRect     = container.getBoundingClientRect();
  let x = event.clientX - cRect.left + 12;
  let y = event.clientY - cRect.top  + 12;

  // Prevent overflow to the right
  const pWidth = 240;
  if (x + pWidth > container.offsetWidth) x = event.clientX - cRect.left - pWidth - 12;

  popup.style.left = `${x}px`;
  popup.style.top  = `${y}px`;
}

function closePopup() {
  document.getElementById('duct-popup').style.display = 'none';
}

// Close popup when clicking outside
document.addEventListener('click', e => {
  const popup = document.getElementById('duct-popup');
  if (popup && !popup.contains(e.target) && !e.target.closest('.duct-polygon')) {
    closePopup();
  }
});

// ── Stat sidebar ──────────────────────────────────────────────────────────
function buildStats(segments, summary) {
  const total    = segments.length;
  const labelled = segments.filter(s => !s.unmatched).length;
  const supply   = segments.filter(s => s.type === 'supply').length;
  const ret      = segments.filter(s => s.type === 'return').length;
  const exhaust  = segments.filter(s => s.type === 'exhaust').length;
  const score    = summary?.review_score != null
    ? (summary.review_score * 100).toFixed(0) + '%' : '—';

  const grid = document.getElementById('stat-grid');
  grid.innerHTML = [
    ['Segments',   total],
    ['Labelled',   labelled],
    ['Supply',     supply],
    ['Return',     ret],
    ['Exhaust',    exhaust],
    ['Score',      score],
  ].map(([label, val]) => `
    <div class="stat-card">
      <div class="stat-value">${val}</div>
      <div class="stat-label">${label}</div>
    </div>
  `).join('');
}

// ── Helpers ───────────────────────────────────────────────────────────────
function formatDimension(seg) {
  if (seg.is_round && seg.diameter_in) return `${seg.diameter_in}" Ø`;
  if (seg.width_in  && seg.height_in)  return `${seg.width_in}" × ${seg.height_in}"`;
  return '—';
}

function showError(msg) {
  showView('view-upload');
  alert(`Error: ${msg}\n\nPlease try again.`);
}
