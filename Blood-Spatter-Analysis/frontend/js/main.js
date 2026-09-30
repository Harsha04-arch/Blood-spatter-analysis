/**
 * dextahhh — Shared JavaScript Utilities
 * API base URL, toast notifications, helpers
 */

const API = {
  BASE: window.location.origin,

  async get(path) {
    const r = await fetch(this.BASE + path);
    return this._handle(r);
  },

  async post(path, body, isFormData = false) {
    const opts = { method: 'POST' };
    if (isFormData) {
      opts.body = body;
    } else {
      opts.headers = { 'Content-Type': 'application/json' };
      opts.body = JSON.stringify(body);
    }
    const r = await fetch(this.BASE + path, opts);
    return this._handle(r);
  },

  async delete(path) {
    const r = await fetch(this.BASE + path, { method: 'DELETE' });
    return this._handle(r);
  },

  async put(path, body) {
    const r = await fetch(this.BASE + path, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    return this._handle(r);
  },

  async _handle(r) {
    const data = await r.json().catch(() => ({ error: 'Invalid JSON response' }));
    if (!r.ok) throw { status: r.status, ...data };
    return data;
  },
};

/* ── Toast Notifications ──────────────────────────────────────────────── */
function toast(msg, type = 'info', duration = 4000) {
  let container = document.getElementById('toast-container');
  if (!container) {
    container = document.createElement('div');
    container.id = 'toast-container';
    document.body.appendChild(container);
  }
  const icons = { success: '✓', error: '✕', warning: '⚠', info: 'ℹ' };
  const el = document.createElement('div');
  el.className = `toast toast-${type}`;
  el.innerHTML = `
    <span style="font-size:16px;flex-shrink:0">${icons[type] || 'ℹ'}</span>
    <span>${msg}</span>
  `;
  container.appendChild(el);
  setTimeout(() => {
    el.style.opacity = '0';
    el.style.transform = 'translateX(40px)';
    el.style.transition = 'all 0.3s';
    setTimeout(() => el.remove(), 300);
  }, duration);
}

/* ── Modal ────────────────────────────────────────────────────────────── */
function showModal({ title, content, onConfirm, confirmText = 'Confirm', confirmClass = 'btn-primary' }) {
  const backdrop = document.createElement('div');
  backdrop.className = 'modal-backdrop';
  backdrop.innerHTML = `
    <div class="modal">
      <div class="modal-title">${title}</div>
      <div>${content}</div>
      <div class="modal-actions">
        <button class="btn btn-secondary" id="modal-cancel">Cancel</button>
        <button class="btn ${confirmClass}" id="modal-confirm">${confirmText}</button>
      </div>
    </div>
  `;
  document.body.appendChild(backdrop);
  const close = () => backdrop.remove();
  backdrop.querySelector('#modal-cancel').onclick = close;
  backdrop.querySelector('#modal-confirm').onclick = () => { close(); onConfirm && onConfirm(); };
  backdrop.onclick = (e) => { if (e.target === backdrop) close(); };
}

/* ── Active Nav ────────────────────────────────────────────────────────── */
function markActiveNav() {
  const page = window.location.pathname.split('/').pop() || 'index.html';
  document.querySelectorAll('.nav-item').forEach(a => {
    const href = a.getAttribute('href') || '';
    if (href.includes(page) || (page === 'index.html' && href === 'index.html')) {
      a.classList.add('active');
    }
  });
}

/* ── Format ───────────────────────────────────────────────────────────── */
function fmtNum(n, decimals = 1) {
  return n == null ? 'N/A' : Number(n).toFixed(decimals);
}
function fmtPct(n) {
  return n == null ? 'N/A' : Number(n).toFixed(1) + '%';
}
function fmtDate(iso) {
  if (!iso) return 'N/A';
  return new Date(iso).toLocaleString();
}

/* ── Confidence Bar ───────────────────────────────────────────────────── */
function renderConfBars(container, probabilities, colors) {
  const entries = Object.entries(probabilities).sort((a, b) => b[1] - a[1]);
  const palette = colors || ['#3b82f6', '#8b5cf6', '#06b6d4', '#10b981'];
  container.innerHTML = entries.map(([cls, pct], i) => `
    <div class="conf-row">
      <div class="conf-row-label" title="${cls}">${cls}</div>
      <div class="conf-bar-wrap">
        <div class="conf-bar" style="width:${pct}%;background:${palette[i % palette.length]}"></div>
      </div>
      <div class="conf-pct">${fmtPct(pct)}</div>
    </div>
  `).join('');
}

/* ── Storage helpers (current case) ───────────────────────────────────── */
const Store = {
  setCase(caseObj) { localStorage.setItem('bpa_current_case', JSON.stringify(caseObj)); },
  getCase() {
    try { return JSON.parse(localStorage.getItem('bpa_current_case')); } catch { return null; }
  },
  clearCase() { localStorage.removeItem('bpa_current_case'); },

  setAnalysis(key, data) { localStorage.setItem(`bpa_analysis_${key}`, JSON.stringify(data)); },
  getAnalysis(key) {
    try { return JSON.parse(localStorage.getItem(`bpa_analysis_${key}`)); } catch { return null; }
  },
};

/* ── Tabs ─────────────────────────────────────────────────────────────── */
function initTabs(container) {
  const buttons = container.querySelectorAll('.tab-btn');
  const panels  = container.querySelectorAll('.tab-panel');
  buttons.forEach(btn => {
    btn.addEventListener('click', () => {
      buttons.forEach(b => b.classList.remove('active'));
      panels.forEach(p => p.classList.remove('active'));
      btn.classList.add('active');
      const target = document.getElementById(btn.dataset.tab);
      if (target) target.classList.add('active');
    });
  });
}

/* ── File upload drag-and-drop enhancement ──────────────────────────── */
function enhanceUploadZone(zone) {
  zone.addEventListener('dragover', e => { e.preventDefault(); zone.classList.add('drag-over'); });
  zone.addEventListener('dragleave', () => zone.classList.remove('drag-over'));
  zone.addEventListener('drop', e => { e.preventDefault(); zone.classList.remove('drag-over'); });
}

document.addEventListener('DOMContentLoaded', () => {
  markActiveNav();
  document.querySelectorAll('.upload-zone').forEach(enhanceUploadZone);
  document.querySelectorAll('[data-tabs]').forEach(initTabs);
});
