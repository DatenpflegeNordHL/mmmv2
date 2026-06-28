"""
MMM Web Server — Browser-based audio sanitization via drag-and-drop.

Launch with: mmm server [--host 127.0.0.1] [--port 8778] [--max-size 500]
"""

import atexit
import os
import shutil
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Dict, Any, Optional

from flask import Flask, request, jsonify, send_file, Response
from mutagen import File as MutagenFile
from werkzeug.utils import secure_filename

from .gpu_web_sanitizer import cuda_available, gpu_web_sanitize

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ALLOWED_EXTENSIONS: frozenset[str] = frozenset({"mp3", "wav", "flac"})
DEFAULT_MAX_FILE_SIZE: int = 95 * 1024 * 1024  # Cloudflare-safe default
DEFAULT_STALE_AGE: int = 3600  # 1 hour

# ---------------------------------------------------------------------------
# Embedded frontend — CSS
# ---------------------------------------------------------------------------

CSS_STYLES = """\
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
:root{
  --bg:#080514;
  --panel:#130c2a;
  --panel-soft:rgba(16,11,35,.76);
  --text:#f7f0ff;
  --muted:#beb4d7;
  --cyan:#34f7ff;
  --magenta:#ff3df2;
  --purple:#8c52ff;
  --amber:#ffb342;
  --red:#ff365d;
  --green:#58ffb5;
  --border:rgba(52,247,255,.42);
}
body{
  font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Arial,sans-serif;
  background:
    radial-gradient(circle at 50% 38%,rgba(255,92,0,.28) 0 7%,transparent 7.5%),
    radial-gradient(circle at 18% 10%,rgba(255,61,242,.20),transparent 31%),
    radial-gradient(circle at 82% 4%,rgba(52,247,255,.18),transparent 26%),
    linear-gradient(180deg,#14082d 0%,#090617 48%,#05030d 100%);
  color:var(--text);min-height:100vh;
  display:flex;flex-direction:column;align-items:center;
  overflow-x:hidden;position:relative;
}
body::before{
  content:"";position:fixed;inset:45% -10% 0;pointer-events:none;opacity:.42;
  background:
    linear-gradient(rgba(52,247,255,.35) 1px,transparent 1px),
    linear-gradient(90deg,rgba(255,61,242,.28) 1px,transparent 1px);
  background-size:100% 32px,72px 100%;
  transform:perspective(440px) rotateX(63deg);transform-origin:top center;
  filter:drop-shadow(0 0 12px rgba(52,247,255,.22));
}
body::after{
  content:"";position:fixed;inset:0;pointer-events:none;
  background:
    linear-gradient(transparent 0 96%,rgba(255,255,255,.035) 97%),
    radial-gradient(ellipse at 50% 54%,rgba(255,179,66,.14),transparent 29%);
  background-size:100% 5px,100% 100%;mix-blend-mode:screen;opacity:.44;
}
[hidden]{display:none!important}
.app-shell{
  width:min(100%,760px);min-height:100vh;padding:clamp(1rem,3vw,2.6rem);
  display:flex;flex-direction:column;position:relative;z-index:1;
}
header{text-align:center;padding:1.3rem 0 .9rem}
.title-row{display:inline-flex;align-items:flex-start;justify-content:center;gap:.8rem;position:relative}
.hero-title{
  font-size:clamp(2.35rem,8vw,5.1rem);line-height:.92;font-weight:900;
  letter-spacing:0;text-transform:uppercase;
  color:#fff;
  text-shadow:
    0 0 5px rgba(255,255,255,.85),
    0 0 16px rgba(52,247,255,.82),
    0 0 34px rgba(255,61,242,.72),
    0 0 66px rgba(140,82,255,.55);
}
.version-sign{
  margin-top:.2rem;padding:.22rem .5rem;border:1px solid var(--cyan);
  border-radius:7px;background:rgba(8,5,20,.74);color:#fff;font-weight:900;
  font-size:clamp(.82rem,2vw,1rem);line-height:1;letter-spacing:.05em;
  box-shadow:0 0 10px rgba(52,247,255,.75),inset 0 0 10px rgba(255,61,242,.28);
  text-shadow:0 0 7px var(--cyan),0 0 12px var(--magenta);
  transform:rotate(4deg);position:relative;
}
.version-sign::before{
  content:"";position:absolute;left:50%;top:-17px;width:1px;height:17px;
  background:linear-gradient(var(--magenta),var(--cyan));
  box-shadow:0 0 8px var(--cyan);
}
header .subtitle{
  color:var(--muted);font-size:clamp(.98rem,2.4vw,1.12rem);margin-top:.75rem;
  text-shadow:0 0 12px rgba(52,247,255,.35);
}
main{
  width:100%;padding:clamp(.75rem,2vw,1rem);
  background:linear-gradient(180deg,rgba(255,255,255,.055),rgba(255,255,255,.018));
  border:1px solid rgba(255,255,255,.09);border-radius:28px;
  box-shadow:0 26px 80px rgba(0,0,0,.44),0 0 70px rgba(140,82,255,.16);
  backdrop-filter:blur(12px);
}
.legal-panel{
  display:flex;gap:.8rem;align-items:flex-start;
  background:linear-gradient(135deg,rgba(45,22,6,.84),rgba(18,10,24,.82));
  border:1px solid rgba(255,179,66,.82);border-radius:16px;
  padding:.95rem 1rem;font-size:.82rem;color:#ffe0a6;
  margin-bottom:1.05rem;line-height:1.45;
  box-shadow:0 0 22px rgba(255,179,66,.22),inset 0 0 20px rgba(255,179,66,.08);
}
.legal-icon{
  flex:0 0 auto;color:var(--amber);filter:drop-shadow(0 0 8px rgba(255,179,66,.8));
}
.dropzone{
  border:2px dashed rgba(52,247,255,.72);border-radius:24px;
  padding:clamp(2.4rem,7vw,4.5rem) 1rem;text-align:center;cursor:pointer;
  transition:border-color .2s,box-shadow .2s,background .2s,transform .2s;
  background:radial-gradient(circle at 50% 0%,rgba(52,247,255,.14),transparent 44%),rgba(9,7,24,.72);
  box-shadow:0 0 0 1px rgba(255,61,242,.22),inset 0 0 34px rgba(52,247,255,.09),0 0 42px rgba(52,247,255,.12);
}
.dropzone:hover,.dropzone.dragover{
  border-color:var(--magenta);background:radial-gradient(circle at 50% 0%,rgba(255,61,242,.18),transparent 44%),rgba(13,8,34,.82);
  box-shadow:0 0 0 1px rgba(52,247,255,.34),inset 0 0 44px rgba(255,61,242,.11),0 0 58px rgba(255,61,242,.24);
  transform:translateY(-1px);
}
.dropzone-icon{
  font-size:3.25rem;margin-bottom:.7rem;color:var(--cyan);
  text-shadow:0 0 8px #fff,0 0 18px var(--cyan),0 0 32px var(--magenta);
}
.dropzone p{color:#f4edff;font-size:1rem;font-weight:700}
.dropzone .hint{font-size:.82rem;margin-top:.45rem;color:var(--muted);font-weight:500}
.dropzone .filename{
  color:var(--cyan);font-weight:800;margin-top:.85rem;word-break:break-all;
  text-shadow:0 0 12px rgba(52,247,255,.7);
}
.options-panel{
  background:var(--panel-soft);border:1px solid rgba(52,247,255,.25);
  border-radius:18px;padding:1rem;margin-top:1rem;
  box-shadow:inset 0 0 24px rgba(140,82,255,.11);
}
.option-group{display:flex;align-items:center;gap:0.6rem;margin-bottom:0.8rem}
.option-group label{color:var(--muted);font-size:0.85rem;min-width:110px}
.option-group select{
  background:#080514;color:var(--text);border:1px solid rgba(52,247,255,.36);
  border-radius:8px;padding:0.42rem 0.55rem;font-size:0.85rem;
}
.option-group input[type=checkbox]{accent-color:var(--magenta)}
.toggle-label{color:var(--muted);font-size:0.8rem}
.btn-primary{
  display:block;width:100%;padding:0.7rem;margin-top:0.5rem;
  background:linear-gradient(90deg,var(--cyan),var(--magenta));color:#fff;
  border:none;border-radius:999px;font-size:0.95rem;font-weight:800;
  cursor:pointer;transition:filter .2s,transform .2s;
  box-shadow:0 0 24px rgba(52,247,255,.28),0 0 24px rgba(255,61,242,.18);
}
.btn-primary:hover{filter:brightness(1.12);transform:translateY(-1px)}
.btn-primary:disabled{background:#211a32;color:#7c728f;cursor:not-allowed;box-shadow:none}
.status-area{text-align:center;margin-top:1.5rem}
.spinner{
  width:36px;height:36px;border:3px solid rgba(52,247,255,.2);border-top-color:var(--cyan);
  border-radius:50%;margin:0 auto 0.8rem;animation:spin .8s linear infinite;
  box-shadow:0 0 16px rgba(52,247,255,.5);
}
@keyframes spin{to{transform:rotate(360deg)}}
#statusText{color:var(--muted);font-size:0.9rem}
.progress-bar{
  width:100%;height:7px;background:#171127;border-radius:999px;
  margin-top:0.8rem;overflow:hidden;
}
.progress-fill{
  height:100%;background:linear-gradient(90deg,var(--cyan),var(--magenta));border-radius:999px;
  transition:width .3s;width:0%;
}
.result-area{text-align:center;margin-top:1.5rem}
.success-icon{font-size:2rem;margin-bottom:0.5rem}
#resultText{color:var(--green);font-size:0.95rem;margin-bottom:0.8rem;text-shadow:0 0 12px rgba(88,255,181,.55)}
.stats-panel{
  background:var(--panel-soft);border:1px solid rgba(52,247,255,.22);border-radius:14px;
  padding:0.8rem;margin-bottom:1rem;text-align:left;font-size:0.8rem;
}
.stats-panel .stat-row{display:flex;justify-content:space-between;padding:0.2rem 0}
.stats-panel .stat-label{color:var(--muted)}
.stats-panel .stat-value{color:var(--text);font-weight:600}
.btn-download{
  display:inline-block;padding:0.65rem 1.55rem;background:linear-gradient(90deg,var(--cyan),var(--purple));color:#fff;
  border-radius:999px;text-decoration:none;font-weight:800;font-size:0.9rem;
  transition:filter .2s;box-shadow:0 0 20px rgba(52,247,255,.24);
}
.btn-download:hover{filter:brightness(1.12)}
.btn-secondary{
  display:inline-block;padding:0.5rem 1.2rem;margin-top:0.8rem;
  background:rgba(8,5,20,.4);color:var(--muted);border:1px solid rgba(52,247,255,.34);
  border-radius:999px;font-size:0.85rem;cursor:pointer;transition:border-color .2s,color .2s,box-shadow .2s;
}
.btn-secondary:hover{border-color:var(--cyan);color:var(--cyan);box-shadow:0 0 18px rgba(52,247,255,.25)}
.error-area{text-align:center;margin-top:1.5rem}
.error-state{padding:1rem;border-radius:18px;background:rgba(31,4,18,.44);border:1px solid rgba(255,54,93,.38);box-shadow:0 0 28px rgba(255,54,93,.13)}
.error-icon{
  font-size:2.5rem;margin-bottom:0.5rem;color:var(--red);
  text-shadow:0 0 7px #fff,0 0 18px var(--red),0 0 32px rgba(255,54,93,.72);
}
#errorText{color:var(--red);font-size:1rem;font-weight:800;margin-bottom:0.8rem;text-shadow:0 0 12px rgba(255,54,93,.55)}
.retry-button{border-color:rgba(255,54,93,.72);color:#ffd7df;box-shadow:0 0 18px rgba(255,54,93,.18)}
.retry-button:hover{border-color:var(--red);color:#fff;box-shadow:0 0 24px rgba(255,54,93,.34)}
footer.footer-credits{
  margin-top:auto;padding:1.4rem .4rem .2rem;text-align:center;color:#a99fc6;font-size:.78rem;line-height:1.6;
  text-shadow:0 0 10px rgba(140,82,255,.25);
}
footer.footer-credits .credit-secondary{color:#7f7598}
@media (max-width:560px){
  .app-shell{padding:.8rem}
  main{border-radius:20px;padding:.8rem}
  .title-row{gap:.45rem}
  .version-sign{transform:rotate(3deg);padding:.18rem .42rem}
  .legal-panel{font-size:.76rem}
  .option-group{align-items:flex-start;flex-direction:column;gap:.35rem}
}
"""

# ---------------------------------------------------------------------------
# Embedded frontend — JavaScript
# ---------------------------------------------------------------------------

JS_APP = """\
(function(){
  const $ = id => document.getElementById(id);
  const dropzone = $('dropzone');
  const fileInput = $('fileInput');
  const options = $('options');
  const status = $('status');
  const result = $('result');
  const error = $('error');
  let selectedFile = null;

  // --- Drop zone ---
  dropzone.addEventListener('click', () => fileInput.click());
  dropzone.addEventListener('dragover', e => { e.preventDefault(); dropzone.classList.add('dragover'); });
  dropzone.addEventListener('dragleave', () => dropzone.classList.remove('dragover'));
  dropzone.addEventListener('drop', e => {
    e.preventDefault();
    dropzone.classList.remove('dragover');
    if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
  });
  fileInput.addEventListener('change', () => { if (fileInput.files.length) handleFile(fileInput.files[0]); });

  function handleFile(file) {
    const ext = file.name.split('.').pop().toLowerCase();
    if (!['mp3','wav','flac'].includes(ext)) {
      showError('Unsupported file type. Please use MP3, WAV, or FLAC.');
      return;
    }
    const maxBytes = parseInt(document.body.dataset.maxSize || '524288000', 10);
    if (file.size > maxBytes) {
      showError('File too large. Maximum size: ' + formatBytes(maxBytes));
      return;
    }
    selectedFile = file;
    // Show filename in dropzone
    let fn = dropzone.querySelector('.filename');
    if (!fn) { fn = document.createElement('p'); fn.className='filename'; dropzone.querySelector('.dropzone-content').appendChild(fn); }
    fn.textContent = file.name + ' (' + formatBytes(file.size) + ')';
    options.hidden = false;
    result.hidden = true;
    error.hidden = true;
  }

  // --- Process ---
  $('processBtn').addEventListener('click', processFile);

  function processFile() {
    if (!selectedFile) return;
    const fmt = $('formatSelect').value;
    const paranoid = $('paranoidToggle').checked;

    // Show status
    options.hidden = true;
    status.hidden = false;
    result.hidden = true;
    error.hidden = true;
    $('statusText').textContent = 'Uploading...';
    $('progressFill').style.width = '0%';
    $('processBtn').disabled = true;

    const formData = new FormData();
    formData.append('file', selectedFile);
    formData.append('format', fmt);
    formData.append('paranoid', paranoid ? 'true' : 'false');

    const xhr = new XMLHttpRequest();
    xhr.upload.addEventListener('progress', e => {
      if (e.lengthComputable) {
        const pct = Math.round((e.loaded / e.total) * 50); // Upload = first 50%
        $('progressFill').style.width = pct + '%';
        $('statusText').textContent = 'Uploading... ' + pct*2 + '%';
      }
    });
    xhr.addEventListener('load', () => {
      if (xhr.status === 202) {
        const data = JSON.parse(xhr.responseText);
        if (data.job_id) {
          $('statusText').textContent = data.message || 'GPU job queued...';
          $('progressFill').style.width = '25%';
          pollJob(data.job_id);
        } else {
          showError(data.error || 'Unable to start processing job.');
        }
      } else if (xhr.status === 429) {
        showError('Server is busy processing another file. Please wait and try again.');
      } else {
        let msg = 'Server error.';
        try { msg = JSON.parse(xhr.responseText).error || msg; } catch(e) {}
        showError(msg);
      }
    });
    xhr.addEventListener('error', () => {
      showError('Network error. Is the server still running?');
      $('processBtn').disabled = false;
    });

    xhr.upload.addEventListener('loadend', () => {
      $('statusText').textContent = 'Upload complete. Starting GPU processing...';
      $('progressFill').style.width = '25%';
    });

    // Start upload
    xhr.open('POST', '/api/upload');
    xhr.send(formData);
  }

  function pollJob(jobId) {
    const poll = () => {
      fetch('/api/job/' + encodeURIComponent(jobId), {cache:'no-store'})
        .then(resp => resp.json().then(data => ({ok: resp.ok, data})))
        .then(({ok, data}) => {
          if (!ok) {
            showError(data.error || 'Unable to read job status.');
            return;
          }
          $('statusText').textContent = data.message || 'Processing audio...';
          $('progressFill').style.width = Math.max(25, Math.min(data.progress || 25, 99)) + '%';
          if (data.status === 'complete') {
            $('progressFill').style.width = '100%';
            showResult(data.result || {});
            $('processBtn').disabled = false;
            return;
          }
          if (data.status === 'failed') {
            showError(data.error || 'Sanitization failed.');
            return;
          }
          setTimeout(poll, 1200);
        })
        .catch(() => showError('Network error while reading job status.'));
    };
    poll();
  }

  function showResult(data) {
    status.hidden = true;
    result.hidden = false;
    $('resultText').textContent = 'File sanitized successfully!';
    const stats = data.stats || {};
    const rows = [
      statRow('Engine', stats.processing_engine || 'N/A'),
      statRow('GPU', stats.gpu_acceleration ? (stats.gpu_device || 'Enabled') : 'Fallback/CPU'),
      statRow('Signal changed', formatBool(stats.signal_changed)),
      statRow('Signal delta', formatSignalDelta(stats.signal_delta_ratio, stats.signal_delta_db)),
      statRow('Hash changed', formatBool(stats.output_hash_changed)),
      statRow('Metadata clean', formatBool(stats.metadata_clean)),
      statRow('Metadata removed', stats.metadata_removed || 0),
      statRow('Processing time', formatTime(stats.processing_time)),
      statRow('Speed', stats.processing_speed || 'N/A'),
    ];
    $('statsPanel').innerHTML = rows.join('');
    $('downloadLink').href = '/api/download/' + data.download_token;
    $('downloadLink').download = data.filename || 'cleaned_audio';
  }

  function showError(msg) {
    status.hidden = true;
    options.hidden = true;
    error.hidden = false;
    $('errorText').textContent = msg;
    $('processBtn').disabled = false;
  }

  $('resetBtn').addEventListener('click', resetUI);
  $('retryBtn').addEventListener('click', resetUI);

  function resetUI() {
    selectedFile = null;
    fileInput.value = '';
    const fn = dropzone.querySelector('.filename');
    if (fn) fn.remove();
    options.hidden = true;
    status.hidden = true;
    result.hidden = true;
    error.hidden = true;
    $('processBtn').disabled = false;
  }

  function formatBytes(b) {
    if (b < 1024) return b + ' B';
    if (b < 1048576) return (b/1024).toFixed(1) + ' KB';
    return (b/1048576).toFixed(1) + ' MB';
  }
  function formatTime(s) {
    if (s == null) return 'N/A';
    return parseFloat(s).toFixed(1) + 's';
  }
  function formatSignalDelta(ratio, db) {
    if (ratio == null) return 'N/A';
    const percent = (parseFloat(ratio) * 100).toFixed(4) + '%';
    if (db == null) return percent;
    return percent + ' (' + parseFloat(db).toFixed(1) + ' dB)';
  }
  function formatBool(value) {
    if (value === true) return 'yes';
    if (value === false) return 'no';
    return 'N/A';
  }
  function statRow(label, value) {
    return '<div class="stat-row"><span class="stat-label">' + escapeHtml(label) +
      '</span><span class="stat-value">' + escapeHtml(String(value)) + '</span></div>';
  }
  function escapeHtml(value) {
    return value
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#39;');
  }
})();
"""

# ---------------------------------------------------------------------------
# Embedded frontend — HTML shell
# ---------------------------------------------------------------------------

HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>MMM - Audio Sanitizer</title>
<style>{css}</style>
</head>
<body data-max-size="{max_size}">
<div class="app-shell">
<header>
  <div class="title-row">
    <h1 class="hero-title">Melodic Metadata Massacrer</h1>
    <span class="version-sign" aria-label="Version 2.0">2.0</span>
  </div>
  <p class="subtitle">Browser-based audio sanitizer</p>
</header>
<main>
  <div class="legal-panel">
    <svg class="legal-icon" width="26" height="26" viewBox="0 0 24 24" aria-hidden="true">
      <path fill="currentColor" d="M12 2 1 21h22L12 2Zm0 6.2 5.6 9.8H6.4L12 8.2Zm-.8 2.8h1.6v4.8h-1.6V11Zm0 6.1h1.6v1.6h-1.6v-1.6Z"/>
    </svg>
    <p>
      LEGAL DISCLAIMER: This tool is for AUTHORIZED SECURITY RESEARCH ONLY.
      Use only on files you own or have explicit permission to modify.
      You are responsible for compliance with applicable laws.
    </p>
  </div>

  <div id="dropzone" class="dropzone">
    <div class="dropzone-content">
      <div class="dropzone-icon">&#127925;</div>
      <p>Drag &amp; drop audio file here</p>
      <p class="hint">or click to select &middot; MP3, WAV, FLAC &middot; max {max_size_mb} MB</p>
      <input type="file" id="fileInput" accept=".mp3,.wav,.flac" hidden>
    </div>
  </div>

  <div id="options" class="options-panel" hidden>
    <div class="option-group">
      <label for="formatSelect">Output format:</label>
      <select id="formatSelect">
        <option value="preserve" selected>Preserve original</option>
        <option value="mp3">MP3</option>
        <option value="wav">WAV</option>
        <option value="flac">FLAC</option>
      </select>
    </div>
    <div class="option-group">
      <label for="paranoidToggle">Paranoid mode:</label>
      <input type="checkbox" id="paranoidToggle">
      <span class="toggle-label">Maximum destruction</span>
    </div>
    <button id="processBtn" class="btn-primary">Sanitize with GPU</button>
  </div>

  <div id="status" class="status-area" hidden>
    <div class="spinner" id="spinner"></div>
    <p id="statusText">Processing...</p>
    <div class="progress-bar"><div class="progress-fill" id="progressFill"></div></div>
  </div>

  <div id="result" class="result-area" hidden>
    <div class="success-icon">&#9989;</div>
    <p id="resultText"></p>
    <div id="statsPanel" class="stats-panel"></div>
    <a id="downloadLink" class="btn-download" href="#">Download cleaned file</a><br>
    <button id="resetBtn" class="btn-secondary">Process another file</button>
  </div>

  <div id="error" class="error-area error-state" hidden>
    <div class="error-icon">&#10060;</div>
    <p id="errorText"></p>
    <button id="retryBtn" class="btn-secondary retry-button">Try again</button>
  </div>
</main>
<footer class="footer-credits">
  <div>MMM v2.0.0 — Browser-based audio sanitizer</div>
  <div class="credit-secondary">Original open-source credit: geeknik/mmm • Retrowave redesign by Dirty D. Noir</div>
</footer>
</div>
<script>{js}</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _validate_extension(filename: str) -> bool:
    """Check file extension against the allowlist."""
    if not filename or "." not in filename:
        return False
    ext = filename.rsplit(".", 1)[-1].lower()
    return ext in ALLOWED_EXTENSIONS


def _validate_audio_content(file_path: Path) -> bool:
    """Validate that the saved upload is readable as an audio file."""
    try:
        audio_file = MutagenFile(file_path)
        if audio_file is not None and getattr(audio_file, "info", None) is not None:
            return True
    except Exception:
        pass

    try:
        import soundfile as sf

        info = sf.info(str(file_path))
        return bool(info.samplerate and info.frames >= 0)
    except Exception:
        return False


def _cleanup_old_files(directory: Path, max_age: int = DEFAULT_STALE_AGE) -> None:
    """Remove files older than *max_age* seconds from *directory*."""
    now = time.time()
    try:
        for entry in directory.iterdir():
            if entry.is_file():
                age = now - entry.stat().st_mtime
                if age > max_age:
                    entry.unlink(missing_ok=True)
    except OSError:
        pass


def _cleanup_download_registry(
    registry: Dict[str, Dict[str, Any]], max_age: int = DEFAULT_STALE_AGE
) -> None:
    """Drop stale download tokens and unlink stale files."""
    now = time.time()
    stale_tokens = []
    for token, entry in registry.items():
        created = float(entry.get("created", 0))
        if now - created > max_age:
            stale_tokens.append(token)

    for token in stale_tokens:
        entry = registry.pop(token, None)
        if not entry:
            continue
        stale_path = Path(str(entry.get("path", "")))
        stale_path.unlink(missing_ok=True)


def _cleanup_job_registry(
    registry: Dict[str, Dict[str, Any]], max_age: int = DEFAULT_STALE_AGE
) -> None:
    """Drop stale background job metadata."""
    now = time.time()
    stale_tokens = [
        token
        for token, entry in registry.items()
        if now - float(entry.get("created", 0)) > max_age
    ]
    for token in stale_tokens:
        registry.pop(token, None)


# ---------------------------------------------------------------------------
# Flask application factory
# ---------------------------------------------------------------------------

def create_app(
    max_file_size: int = DEFAULT_MAX_FILE_SIZE,
) -> Flask:
    """Create and configure the Flask application."""
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = max_file_size
    app.config["SECRET_KEY"] = os.urandom(32)

    # Secure temp directory — cleaned up on exit
    temp_dir = Path(tempfile.mkdtemp(prefix="mmm_server_"))
    app.config["UPLOAD_FOLDER"] = temp_dir
    atexit.register(shutil.rmtree, str(temp_dir), True)

    # One-at-a-time processing lock
    app.config["PROCESSING_LOCK"] = threading.Lock()

    # Download registry: token -> {"path": Path, "filename": str, "created": float}
    app.config["DOWNLOAD_REGISTRY"]: Dict[str, Dict[str, Any]] = {}
    app.config["DOWNLOAD_REGISTRY_LOCK"] = threading.Lock()
    app.config["JOB_REGISTRY"]: Dict[str, Dict[str, Any]] = {}
    app.config["JOB_REGISTRY_LOCK"] = threading.Lock()

    max_size_mb = max_file_size // (1024 * 1024)

    # --- Routes ---

    @app.route("/")
    def index() -> Response:
        html = HTML_TEMPLATE.format(
            css=CSS_STYLES,
            js=JS_APP,
            max_size=max_file_size,
            max_size_mb=max_size_mb,
        )
        return Response(html, mimetype="text/html")

    @app.route("/api/status")
    def api_status() -> tuple:
        with app.config["DOWNLOAD_REGISTRY_LOCK"]:
            _cleanup_download_registry(app.config["DOWNLOAD_REGISTRY"])
        with app.config["JOB_REGISTRY_LOCK"]:
            _cleanup_job_registry(app.config["JOB_REGISTRY"])
        lock: threading.Lock = app.config["PROCESSING_LOCK"]
        busy = not lock.acquire(blocking=False)
        if not busy:
            lock.release()
        return jsonify({
            "busy": busy,
            "version": "2.0.0",
            "max_file_size_mb": max_size_mb,
            "gpu_available": cuda_available(),
        }), 200

    @app.route("/api/upload", methods=["POST"])
    def api_upload() -> tuple:
        lock: threading.Lock = app.config["PROCESSING_LOCK"]
        with app.config["DOWNLOAD_REGISTRY_LOCK"]:
            _cleanup_download_registry(app.config["DOWNLOAD_REGISTRY"])
        with app.config["JOB_REGISTRY_LOCK"]:
            _cleanup_job_registry(app.config["JOB_REGISTRY"])

        if not lock.acquire(blocking=False):
            return jsonify({"error": "Server is busy processing another file. Please wait."}), 429

        try:
            return _handle_upload(app, lock)
        except Exception:
            lock.release()
            app.logger.exception("Upload handler failed")
            return jsonify({"error": "Sanitization failed. Please try a different file."}), 500

    @app.route("/api/job/<job_id>")
    def api_job(job_id: str) -> tuple:
        with app.config["JOB_REGISTRY_LOCK"]:
            _cleanup_job_registry(app.config["JOB_REGISTRY"])
            job = app.config["JOB_REGISTRY"].get(job_id)
            if job is None:
                return jsonify({"error": "Job not found or expired."}), 404
            return jsonify(job), 200

    @app.route("/api/download/<token>")
    def api_download(token: str) -> tuple:
        registry: dict = app.config["DOWNLOAD_REGISTRY"]
        with app.config["DOWNLOAD_REGISTRY_LOCK"]:
            _cleanup_download_registry(registry)
            entry = registry.pop(token, None)

        if entry is None:
            return jsonify({"error": "File not found or expired."}), 404

        file_path = Path(entry["path"])
        if not file_path.exists():
            return jsonify({"error": "File not found or expired."}), 404

        filename = entry["filename"]

        response = send_file(
            str(file_path),
            as_attachment=True,
            download_name=filename,
        )
        # Clean up file after response is sent to client
        response.call_on_close(lambda: file_path.unlink(missing_ok=True))
        return response

    @app.errorhandler(413)
    def _too_large(e):
        return jsonify({"error": f"File too large. Maximum size: {max_size_mb} MB."}), 413

    return app


def _handle_upload(app: Flask, processing_lock: threading.Lock) -> tuple:
    """Validate an upload and start a background processing job."""
    temp_dir: Path = app.config["UPLOAD_FOLDER"]
    jobs: dict = app.config["JOB_REGISTRY"]

    def fail_response(payload: dict, status_code: int) -> tuple:
        processing_lock.release()
        return jsonify(payload), status_code

    # Reap stale files before processing
    _cleanup_old_files(temp_dir)
    with app.config["DOWNLOAD_REGISTRY_LOCK"]:
        _cleanup_download_registry(app.config["DOWNLOAD_REGISTRY"])
    with app.config["JOB_REGISTRY_LOCK"]:
        _cleanup_job_registry(jobs)

    # Validate upload
    if "file" not in request.files:
        return fail_response({"error": "No file provided."}, 400)

    f = request.files["file"]
    if not f.filename:
        return fail_response({"error": "No file selected."}, 400)

    safe_name = secure_filename(f.filename)
    if not safe_name or not _validate_extension(safe_name):
        return fail_response({"error": "Unsupported file type. Use MP3, WAV, or FLAC."}, 400)

    # Parse options
    output_format = request.form.get("format", "preserve")
    if output_format not in ("preserve", "mp3", "wav", "flac"):
        output_format = "preserve"

    paranoid = request.form.get("paranoid", "false").lower() == "true"

    # Save to temp directory with UUID prefix
    unique_name = f"{uuid.uuid4().hex}_{safe_name}"
    input_path = temp_dir / unique_name
    f.save(str(input_path))

    if not _validate_audio_content(input_path):
        input_path.unlink(missing_ok=True)
        return fail_response({"error": "Invalid or unsupported audio file."}, 400)

    try:
        job_id = uuid.uuid4().hex
        with app.config["JOB_REGISTRY_LOCK"]:
            jobs[job_id] = {
                "id": job_id,
                "status": "queued",
                "progress": 25,
                "message": "Upload complete. Waiting for GPU worker...",
                "created": time.time(),
            }

        worker = threading.Thread(
            target=_process_upload_job,
            args=(app, job_id, input_path, safe_name, output_format, paranoid, processing_lock),
            daemon=True,
        )
        worker.start()
        return jsonify({
            "success": True,
            "job_id": job_id,
            "message": "Upload accepted. GPU processing started.",
        }), 202
    except Exception:
        input_path.unlink(missing_ok=True)
        raise


def _process_upload_job(
    app: Flask,
    job_id: str,
    input_path: Path,
    safe_name: str,
    output_format: str,
    paranoid: bool,
    processing_lock: threading.Lock,
) -> None:
    """Run sanitization in a background thread."""
    try:
        _update_job(app, job_id, status="processing", progress=35, message="Starting CUDA sanitizer...")
        fmt = None if output_format == "preserve" else output_format
        result = _sanitize_gpu_first(app, input_path, fmt, paranoid, job_id)

        if not result.get("success"):
            _update_job(
                app,
                job_id,
                status="failed",
                progress=100,
                message="Sanitization failed.",
                error=result.get("error", "Sanitization failed."),
            )
            return

        _update_job(app, job_id, progress=95, message="Preparing download...")
        output_path = Path(result["output_file"])
        token = uuid.uuid4().hex
        stem = Path(safe_name).stem
        ext = output_path.suffix
        download_name = f"{stem}_clean{ext}"

        with app.config["DOWNLOAD_REGISTRY_LOCK"]:
            app.config["DOWNLOAD_REGISTRY"][token] = {
                "path": str(output_path),
                "filename": download_name,
                "created": time.time(),
            }

        _update_job(
            app,
            job_id,
            status="complete",
            progress=100,
            message="Sanitization complete.",
            result={
                "success": True,
                "download_token": token,
                "filename": download_name,
                "stats": result.get("stats", {}),
            },
        )
    except Exception as exc:
        app.logger.exception("Background upload job failed")
        _update_job(
            app,
            job_id,
            status="failed",
            progress=100,
            message="Sanitization failed.",
            error=str(exc),
        )
    finally:
        input_path.unlink(missing_ok=True)
        processing_lock.release()


def _sanitize_gpu_first(
    app: Flask,
    input_path: Path,
    output_format: Optional[str],
    paranoid: bool,
    job_id: str,
) -> Dict[str, Any]:
    try:
        _update_job(app, job_id, progress=45, message="Processing spectral pass on GPU...")
        return gpu_web_sanitize(
            input_file=input_path,
            output_file=None,
            paranoid_mode=paranoid,
            output_format=output_format,
            verbose=False,
        )
    except Exception as exc:
        app.logger.warning("GPU sanitizer failed; falling back to preserving sanitizer: %s", exc)
        _update_job(
            app,
            job_id,
            progress=60,
            message="GPU path failed; using safe CPU fallback...",
        )

        from .preserving_sanitizer import preserving_sanitize

        result = preserving_sanitize(
            input_file=input_path,
            output_file=None,
            paranoid_mode=paranoid,
            threat_count=0,
            output_format=output_format,
            verbose=False,
        )
        stats = result.setdefault("stats", {})
        stats["processing_engine"] = "cpu_preserving_fallback"
        stats["gpu_acceleration"] = False
        stats["gpu_fallback_error"] = str(exc)
        return result


def _update_job(app: Flask, job_id: str, **updates: Any) -> None:
    with app.config["JOB_REGISTRY_LOCK"]:
        job = app.config["JOB_REGISTRY"].get(job_id)
        if job is None:
            return
        job.update(updates)


# ---------------------------------------------------------------------------
# Entry point (called from CLI)
# ---------------------------------------------------------------------------

def run_server(
    host: str = "127.0.0.1",
    port: int = 8778,
    max_file_size: int = DEFAULT_MAX_FILE_SIZE,
) -> None:
    """Create and run the MMM web server."""
    from .ui.console import ConsoleManager
    from .ui.banners import BannerManager

    console = ConsoleManager()
    banner = BannerManager()

    banner.show_main_banner()

    console.warning(
        "LEGAL DISCLAIMER: This tool is for AUTHORIZED SECURITY RESEARCH ONLY"
    )
    console.info("   Use only on files you own or have explicit permission to modify\n")

    if host != "127.0.0.1":
        console.warning(
            f"Binding to {host} — the server will be accessible from the network!"
        )
        console.warning(
            "For production hosting, run the Flask app behind gunicorn/uwsgi and a "
            "reverse proxy instead of exposing Werkzeug directly."
        )

    max_mb = max_file_size // (1024 * 1024)
    console.success(f"Starting MMM web server at http://{host}:{port}")
    console.info(f"   Max upload size: {max_mb} MB")
    console.info("   Press Ctrl+C to stop\n")

    app = create_app(max_file_size=max_file_size)

    # debug=False is mandatory — Werkzeug debugger enables remote code execution
    app.run(host=host, port=port, debug=False, threaded=True)
