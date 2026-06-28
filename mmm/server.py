"""
MMV2 Web Server - browser-based audio quality engine via drag-and-drop.

Launch with: mmm server [--host 127.0.0.1] [--port 8778] [--max-size 500]
"""

import atexit
import hashlib
import os
import shutil
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Dict, Any, Optional, Union

from flask import Flask, request, jsonify, send_file, Response
from mutagen import File as MutagenFile
from werkzeug.utils import secure_filename

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ALLOWED_EXTENSIONS: frozenset[str] = frozenset({"mp3", "wav", "flac", "aiff", "aif"})
DEFAULT_MAX_FILE_SIZE: int = 95 * 1024 * 1024  # Cloudflare-safe default
DEFAULT_STALE_AGE: int = 3600  # 1 hour
ENGINE_VERSION: str = "1.045"
QUALITY_MODES: frozenset[str] = frozenset(
    {"analyze_only", "safe_master", "naturalize", "full_release"}
)
METADATA_CLEAN_MODE: str = "metadata_clean"
LEGACY_MODE_ALIASES: frozenset[str] = frozenset({"legacy_sanitize", METADATA_CLEAN_MODE})
OUTPUT_FORMATS: frozenset[str] = frozenset({"preserve", "mp3", "wav", "flac"})
LOUDNESS_TARGETS: Dict[str, Dict[str, Union[float, str]]] = {
    "streaming_safe": {"label": "Streaming Safe", "target_lufs": -14.0},
    "club_loud": {"label": "Club/Loud", "target_lufs": -9.0},
    "conservative": {"label": "Conservative", "target_lufs": -16.0},
}
TRUE_PEAK_CEILINGS: frozenset[str] = frozenset({"-1.0", "-1.5", "-2.0"})
SAMPLE_RATE_OVERRIDES: frozenset[str] = frozenset({"preserve", "44100", "48000"})
BIT_DEPTH_OVERRIDES: frozenset[str] = frozenset({"preserve", "16", "24", "32"})

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
  width:min(100%,1040px);min-height:100vh;padding:clamp(.55rem,1.35vw,1.1rem);
  display:flex;flex-direction:column;position:relative;z-index:1;
}
.hero-header{
  position:relative;text-align:center;margin:0 auto clamp(.55rem,1.2vw,1rem);
  max-width:min(92vw,860px);padding:clamp(.25rem,.8vw,.55rem) 0 0;
}
.title-lockup{
  position:relative;display:inline-flex;align-items:flex-start;justify-content:center;
  gap:clamp(.55rem,1.3vw,.95rem);
}
.hero-title{
  margin:0;display:flex;flex-direction:column;align-items:center;
  font-size:clamp(2.7rem,5.25vw,4.45rem);line-height:.9;font-weight:900;
  letter-spacing:.025em;text-transform:uppercase;color:#f8fbff;
  text-shadow:
    0 0 6px rgba(116,220,255,.68),
    0 0 16px rgba(180,92,255,.38),
    0 0 30px rgba(255,64,214,.18);
}
.hero-title span{display:block;white-space:nowrap}
.version-sign{
  position:relative;flex:0 0 auto;margin-top:clamp(.16rem,.55vw,.42rem);
  padding:.24rem .56rem;border:2px solid rgba(94,234,255,.95);
  border-radius:.48rem;background:linear-gradient(180deg,rgba(20,31,58,.88),rgba(17,17,38,.9));
  color:#fff;font-size:clamp(.82rem,1.18vw,1.02rem);font-weight:900;
  letter-spacing:.06em;line-height:1;text-shadow:
    0 0 5px rgba(255,92,231,.88),
    0 0 12px rgba(94,234,255,.74);
  box-shadow:
    0 0 9px rgba(94,234,255,.58),
    0 0 18px rgba(255,64,214,.24),
    inset 0 0 9px rgba(94,234,255,.16);
  transform:rotate(2deg);
}
.version-sign::before,.version-sign::after{
  content:"";position:absolute;top:-1.05rem;width:1px;height:.95rem;
  background:linear-gradient(to bottom,rgba(94,234,255,.82),transparent);
  box-shadow:0 0 7px rgba(94,234,255,.58);
}
.version-sign::before{left:24%}
.version-sign::after{right:24%}
.hero-subtitle{
  margin:clamp(.42rem,.9vw,.72rem) 0 0;color:rgba(220,230,255,.78);
  font-size:clamp(.84rem,1.05vw,1rem);font-weight:500;
  text-shadow:0 0 10px rgba(94,234,255,.22);
}
main{
  width:100%;padding:clamp(.55rem,1.25vw,.85rem);
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
  margin-top:.35rem;padding:.2rem .4rem .05rem;text-align:center;color:#a99fc6;font-size:.64rem;line-height:1.25;
  text-shadow:0 0 10px rgba(140,82,255,.25);
}
footer.footer-credits .credit-secondary{color:#7f7598}
@media (max-width:560px){
  .app-shell{padding:.8rem}
  main{border-radius:20px;padding:.8rem}
  .hero-header{margin-bottom:1rem;padding-top:.45rem}
  .title-lockup{gap:.5rem}
  .hero-title{font-size:clamp(2.25rem,11.2vw,3.35rem)}
  .hero-title span{white-space:normal}
  .version-sign{margin-top:.12rem;font-size:.78rem;padding:.2rem .43rem;transform:rotate(2deg)}
  .legal-panel{font-size:.76rem}
  .option-group{align-items:flex-start;flex-direction:column;gap:.35rem}
}
.product-kicker{
  color:#7dd3fc;text-transform:uppercase;font-size:.68rem;letter-spacing:.16em;
  margin-bottom:.35rem;text-shadow:0 0 14px rgba(125,211,252,.35);
}
.quality-hero-title{
  display:block;text-transform:none;font-size:clamp(1.7rem,3.15vw,2.95rem);
  letter-spacing:0;line-height:.96;max-width:920px;
}
.beta-badge{
  display:inline-flex;margin-top:.55rem;padding:.24rem .58rem;border:1px solid rgba(125,211,252,.38);
  border-radius:999px;background:rgba(8,12,28,.58);color:#dbeafe;font-size:.68rem;
}
.quality-console{width:min(100%,1100px);margin-inline:auto}
.console-card{
  position:relative;padding:.72rem;border-radius:22px;
  background:linear-gradient(180deg,rgba(18,24,45,.86),rgba(9,10,24,.9));
  border:1px solid rgba(148,163,184,.2);
  box-shadow:0 28px 90px rgba(0,0,0,.48),0 0 80px rgba(79,70,229,.18),inset 0 0 44px rgba(15,23,42,.65);
}
.console-topbar,.console-hint-row,.action-row{display:flex;align-items:center;justify-content:space-between;gap:.75rem;flex-wrap:wrap}
.console-topbar{margin-bottom:.55rem}
.engine-brand{display:flex;align-items:center;gap:.55rem}
.logo-mark{
  display:inline-flex;align-items:center;justify-content:center;width:52px;height:30px;border-radius:10px;
  background:linear-gradient(135deg,#111827,#312e81);border:1px solid rgba(125,211,252,.38);
  color:#fff;font-size:.72rem;font-weight:900;letter-spacing:.08em;
}
.engine-version{color:#a5b4fc;font-size:.74rem}
.engine-status{
  display:inline-flex;align-items:center;gap:.45rem;color:#bbf7d0;font-size:.7rem;font-weight:800;letter-spacing:.08em;
}
.active-dot{width:9px;height:9px;border-radius:50%;background:#22c55e;box-shadow:0 0 14px #22c55e}
.quality-dropzone{padding:clamp(.7rem,1.5vw,1.05rem) .7rem;border-color:rgba(125,211,252,.55)}
.quality-dropzone .dropzone-icon{font-size:1.55rem;margin-bottom:.2rem}
.visualizer-card{
  margin-top:.65rem;padding:.58rem;border-radius:18px;
  background:linear-gradient(180deg,rgba(8,13,30,.8),rgba(2,6,23,.72));
  border:1px solid rgba(125,211,252,.2);
  box-shadow:inset 0 0 28px rgba(14,165,233,.08),0 0 36px rgba(168,85,247,.1);
}
.visualizer-header{display:flex;align-items:center;justify-content:space-between;gap:.75rem;margin-bottom:.38rem}
.visualizer-title{color:#e2e8f0;font-size:.74rem;font-weight:850}
.wave-canvas{
  display:block;width:100%;height:84px;border-radius:14px;
  background:linear-gradient(90deg,rgba(14,165,233,.08),rgba(168,85,247,.14),rgba(34,197,94,.08));
  border:1px solid rgba(148,163,184,.14);
}
.visualizer-controls{display:flex;align-items:center;justify-content:space-between;gap:.6rem;margin-top:.44rem;flex-wrap:wrap}
.visualizer-button{
  border:1px solid rgba(125,211,252,.42);border-radius:999px;background:rgba(2,6,23,.64);
  color:#e0f2fe;font-size:.72rem;font-weight:850;padding:.35rem .7rem;cursor:pointer;
  box-shadow:0 0 16px rgba(14,165,233,.12);
}
.visualizer-button:disabled{opacity:.45;cursor:not-allowed}
.visualizer-button:not(:disabled):hover{border-color:#67e8f9;color:#fff;box-shadow:0 0 22px rgba(14,165,233,.24)}
.visualizer-status{color:#94a3b8;font-size:.7rem;text-align:right}
.console-hint-row{padding:.55rem .2rem 0;color:#94a3b8;font-size:.68rem}
.control-strip,.analysis-preview,.spectral-risk-grid,.timeline-panel{
  margin-top:.65rem;border-radius:18px;background:rgba(8,13,30,.72);
  border:1px solid rgba(148,163,184,.16);box-shadow:inset 0 0 28px rgba(30,41,59,.45);
  padding:.68rem;
}
.mode-group{display:grid;grid-template-columns:repeat(5,1fr);gap:.42rem;margin-bottom:.58rem}
.mode-button{
  border:1px solid rgba(148,163,184,.24);border-radius:14px;background:rgba(15,23,42,.68);
  color:#cbd5e1;padding:.45rem .38rem;font-size:.72rem;font-weight:750;cursor:pointer;
}
.mode-button.active{color:#fff;border-color:rgba(125,211,252,.7);box-shadow:0 0 18px rgba(14,165,233,.2)}
.control-grid{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:.48rem}
.control-grid label{display:flex;flex-direction:column;gap:.24rem;color:#94a3b8;font-size:.68rem}
.control-grid select{
  background:#020617;color:#f8fafc;border:1px solid rgba(148,163,184,.28);border-radius:9px;padding:.38rem .46rem;font-size:.72rem;
}
.action-row{margin-top:.58rem;justify-content:flex-start}
.action-row .btn-primary{width:auto;min-width:160px;margin:0}
.analysis-preview{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:.48rem}
.meter-card,.metric-card,.spectral-risk-grid div{
  background:linear-gradient(180deg,rgba(15,23,42,.82),rgba(2,6,23,.75));
  border:1px solid rgba(148,163,184,.15);border-radius:13px;padding:.58rem;
}
.lufs-card{grid-column:span 2}
.metric-card span,.meter-card span,.spectral-risk-grid span{display:block;color:#94a3b8;font-size:.65rem;margin-bottom:.22rem}
.metric-card strong,.meter-card strong,.spectral-risk-grid strong{color:#f8fafc;font-size:.82rem}
.lufs-card small{display:block;margin-top:.12rem;color:#c4b5fd;font-size:.64rem}
.metric-card.readiness strong{color:#67e8f9}
.lufs-meter{height:7px;border-radius:999px;background:#111827;margin-top:.7rem;overflow:hidden}
.lufs-meter span{display:block;height:100%;width:0;background:linear-gradient(90deg,#22c55e,#38bdf8,#a855f7);transition:width .35s}
.spectral-risk-grid{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:.48rem}
.timeline-panel h2{font-size:.78rem;color:#e2e8f0;margin-bottom:.48rem}
.timeline-panel ol{display:grid;grid-template-columns:repeat(8,minmax(0,1fr));gap:.4rem;list-style:none;counter-reset:steps}
.timeline-panel li{
  counter-increment:steps;position:relative;padding:.46rem .42rem .46rem 1.62rem;border-radius:11px;
  background:rgba(15,23,42,.58);border:1px solid rgba(148,163,184,.14);color:#94a3b8;font-size:.62rem;
}
.timeline-panel li::before{
  content:counter(steps);position:absolute;left:.38rem;top:.42rem;width:.92rem;height:.92rem;border-radius:50%;
  display:grid;place-items:center;background:#1e293b;color:#cbd5e1;font-size:.56rem;font-weight:800;
}
.timeline-panel li.done{border-color:rgba(34,197,94,.45);color:#dcfce7}
.timeline-panel li.done::before{background:#16a34a;color:#fff}
.download-actions{display:flex;gap:.6rem;justify-content:center;flex-wrap:wrap;margin:.8rem 0}
.btn-download.subtle{background:linear-gradient(90deg,#1e293b,#334155);box-shadow:none}
@media (max-width:760px){
  .mode-group,.control-grid,.analysis-preview,.spectral-risk-grid,.timeline-panel ol{grid-template-columns:1fr}
  .lufs-card{grid-column:auto}
  .action-row .btn-primary{width:100%}
}
@media (min-width:900px){
  main.quality-console{
    display:grid;grid-template-columns:minmax(0,1.05fr) minmax(320px,.95fr);
    gap:.62rem;align-items:start;
  }
  .console-card,.visualizer-card,.control-strip,.timeline-panel{grid-column:1}
  .analysis-preview,.spectral-risk-grid,#status,#result,#error{grid-column:2}
  .console-card{grid-row:1}
  .analysis-preview{grid-row:1 / span 2;margin-top:0}
  .control-strip{grid-row:2}
  .spectral-risk-grid{grid-row:3;margin-top:0}
  .timeline-panel{grid-row:3;margin-top:0}
  #status,#result,#error{grid-row:4;margin-top:.55rem}
  .analysis-preview{grid-template-columns:repeat(2,minmax(0,1fr))}
  .spectral-risk-grid{grid-template-columns:repeat(5,minmax(0,1fr))}
  .timeline-panel ol{grid-template-columns:repeat(8,minmax(0,1fr));gap:.28rem}
  .timeline-panel li{padding:.38rem .28rem .38rem 1.25rem;font-size:.52rem;line-height:1.08}
  .timeline-panel li::before{left:.3rem;top:.38rem;width:.74rem;height:.74rem;font-size:.48rem}
  .control-grid{grid-template-columns:repeat(3,minmax(0,1fr))}
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
  const playPreviewBtn = $('playPreviewBtn');
  const visualizerStatus = $('visualizerStatus');
  const previewAudio = $('previewAudio');
  let selectedFile = null;
  let selectedMode = 'safe_master';
  let selectedObjectUrl = null;
  const visualizer = {
    raf: null,
    mode: 'idle',
    waveform: null,
    progress: 0,
    audioCtx: null,
    analyser: null,
    source: null,
    freqData: null,
    lastNow: 0
  };
  startVisualizer('idle');

  document.querySelectorAll('.mode-button').forEach(btn => {
    btn.addEventListener('click', () => setMode(btn.dataset.mode));
  });
  $('analyzeOnlyBtn').addEventListener('click', () => { setMode('analyze_only'); processFile(); });
  $('safeMasterBtn').addEventListener('click', () => { setMode('safe_master'); processFile(); });
  $('naturalizeBtn').addEventListener('click', () => { setMode('naturalize'); processFile(); });
  $('loudnessTarget').addEventListener('change', () => {
    const label = $('loudnessTarget').selectedOptions[0].textContent;
    $('metricTarget').textContent = 'Target: ' + label;
  });

  function setMode(mode) {
    selectedMode = mode;
    document.querySelectorAll('.mode-button').forEach(btn => {
      btn.classList.toggle('active', btn.dataset.mode === mode);
    });
    const labels = {
      analyze_only: 'Analyze Only',
      safe_master: 'Analyze & Master',
      naturalize: 'Naturalize Pass',
      full_release: 'Analyze & Master',
      metadata_clean: 'Metadata Clean'
    };
    $('processBtn').textContent = labels[mode] || 'Analyze & Master';
  }

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
    if (!['mp3','wav','flac','aiff','aif'].includes(ext)) {
      showError('Unsupported file type. Please use WAV, FLAC, AIFF, or MP3.');
      return;
    }
    const maxBytes = parseInt(document.body.dataset.maxSize || '524288000', 10);
    if (file.size > maxBytes) {
      showError('File too large. Maximum size: ' + formatBytes(maxBytes));
      return;
    }
    selectedFile = file;
    setPreviewAudio(URL.createObjectURL(file), 'Input preview ready');
    // Show filename in dropzone
    let fn = dropzone.querySelector('.filename');
    if (!fn) { fn = document.createElement('p'); fn.className='filename'; dropzone.querySelector('.dropzone-content').appendChild(fn); }
    fn.textContent = file.name + ' (' + formatBytes(file.size) + ')';
    options.hidden = false;
    result.hidden = true;
    error.hidden = true;
    resetMetrics();
    setGeneratedWaveform(file.name);
    startVisualizer('idle');
  }

  // --- Process ---
  $('processBtn').addEventListener('click', processFile);

  function processFile() {
    if (!selectedFile) return;
    const fmt = $('formatSelect').value;
    const paranoid = false;
    const mode = selectedMode;

    // Show status
    options.hidden = true;
    status.hidden = false;
    result.hidden = true;
    error.hidden = true;
    $('statusText').textContent = 'Uploading...';
    $('progressFill').style.width = '0%';
    $('processBtn').disabled = true;
    setTimeline(['upload']);
    startVisualizer('processing');

    const formData = new FormData();
    formData.append('file', selectedFile);
    formData.append('format', fmt);
    formData.append('paranoid', paranoid ? 'true' : 'false');
    formData.append('mode', mode);
    formData.append('loudness_target', $('loudnessTarget').value);
    formData.append('true_peak_ceiling', $('truePeakCeiling').value);
    formData.append('sample_rate_override', $('sampleRateOverride').value);
    formData.append('bit_depth_override', $('bitDepthOverride').value);

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
          $('statusText').textContent = data.message || 'Quality job queued...';
          $('progressFill').style.width = '25%';
          visualizer.progress = .25;
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
      $('statusText').textContent = 'Upload complete. Starting local engine...';
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
          const progressValue = Math.max(25, Math.min(data.progress || 25, 99));
          $('progressFill').style.width = progressValue + '%';
          visualizer.progress = progressValue / 100;
          setTimeline(data.processing_steps || []);
          if (data.status === 'complete') {
            $('progressFill').style.width = '100%';
            showResult(data.result || {});
            $('processBtn').disabled = false;
            return;
          }
          if (data.status === 'failed') {
            showError(data.error || 'Processing failed.');
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
    $('resultText').textContent = data.mode === 'analyze_only'
      ? 'Analysis report generated.'
      : (data.mode === 'metadata_clean' ? 'Metadata clean complete.' : 'Mastering pass complete.');
    const stats = data.stats || {};
    updateMetrics(data);
    const rows = [
      statRow('Engine', data.engine_version ? 'MMV2 ' + data.engine_version : (stats.processing_engine || 'N/A')),
      statRow('Mode', data.mode || 'N/A'),
      statRow('Output format', data.output_format || 'N/A'),
      statRow('Loudness target', data.loudness_target || 'N/A'),
      statRow('Limiter ceiling', data.true_peak_ceiling ? data.true_peak_ceiling + ' dBTP' : 'N/A'),
      statRow('Acceleration', stats.gpu_acceleration ? (stats.gpu_device || 'GPU') : 'CPU'),
      statRow('Preview protected', formatBool(data.preview_protected)),
      statRow('Peak safety', data.peak_safety || 'Available after processing'),
      statRow('Signal changed', formatBool(stats.signal_changed)),
      statRow('Signal delta', formatSignalDelta(stats.signal_delta_ratio, stats.signal_delta_db)),
      statRow('Hash changed', formatBool(stats.output_hash_changed)),
      statRow('Metadata clean', formatBool(stats.metadata_clean)),
      statRow('Metadata removed', stats.metadata_removed || 0),
      statRow('Processing time', formatTime(stats.processing_time)),
      statRow('Speed', stats.processing_speed || 'N/A'),
    ];
    $('statsPanel').innerHTML = rows.join('');
    if (data.download_token) {
      $('downloadLink').hidden = false;
      $('downloadLink').href = '/api/download/' + data.download_token;
      $('downloadLink').download = data.filename || 'master_audio';
      $('downloadLink').textContent = data.mode === 'metadata_clean' ? 'Download Clean Audio' : 'Download Master';
      setPreviewAudio('/api/preview/' + data.download_token, 'Processed audio ready');
    } else {
      $('downloadLink').hidden = true;
    }
    setReportLink('jsonReportLink', data.report_artifacts && data.report_artifacts.json_download_token);
    setReportLink('htmlReportLink', data.report_artifacts && data.report_artifacts.html_download_token);
    loadWaveformArtifact(data.waveform_artifact && data.waveform_artifact.json_download_token);
    setTimeline(data.processing_steps || []);
    startVisualizer('result');
  }

  function showError(msg) {
    status.hidden = true;
    options.hidden = true;
    error.hidden = false;
    $('errorText').textContent = msg;
    $('processBtn').disabled = false;
    setTimeline([]);
    startVisualizer('error');
  }

  $('resetBtn').addEventListener('click', resetUI);
  $('retryBtn').addEventListener('click', resetUI);

  function resetUI() {
    selectedFile = null;
    setPreviewAudio('', 'Drop audio to preview waveform');
    fileInput.value = '';
    const fn = dropzone.querySelector('.filename');
    if (fn) fn.remove();
    options.hidden = true;
    status.hidden = true;
    result.hidden = true;
    error.hidden = true;
    $('processBtn').disabled = false;
    resetMetrics();
    visualizer.waveform = null;
    startVisualizer('idle');
  }

  function setReportLink(id, token) {
    const link = $(id);
    if (!token) {
      link.hidden = true;
      return;
    }
    link.hidden = false;
    link.href = '/api/download/' + token;
  }

  function updateMetrics(data) {
    const before = data.metrics_before || {};
    const after = data.metrics_after || before;
    const metrics = after || {};
    if (data.mode === 'metadata_clean' && !data.metrics_after) {
      resetMetrics('Metadata clean path');
      return;
    }
    $('metricLufs').textContent = formatMetric(metrics.integrated_lufs, ' LUFS');
    $('metricPeak').textContent = formatMetric(metrics.estimated_true_peak_dbtp, ' dBTP');
    $('metricPlr').textContent = formatMetric(metrics.peak_to_loudness_ratio, ' dB');
    $('metricStereo').textContent = formatMetric(metrics.stereo_correlation, '');
    $('metricDc').textContent = Array.isArray(metrics.dc_offset) ? metrics.dc_offset.map(v => Number(v).toExponential(2)).join(' / ') : 'Not measured';
    $('metricClipping').textContent = metrics.clipping_sample_count == null ? 'Not measured' : String(metrics.clipping_sample_count);
    $('metricLowWidth').textContent = formatMetric(metrics.low_end_width, '');
    const harsh = metrics.band_energy && metrics.band_energy.harsh_5000_9000_hz;
    $('metricHarsh').textContent = harsh == null ? 'Not measured' : riskLabel(harsh, 0.18);
    const score = metrics.release_readiness && metrics.release_readiness.score;
    $('metricReadiness').textContent = score == null ? 'Available after processing' : score + '/100';
    if (metrics.integrated_lufs != null) {
      const width = Math.max(0, Math.min(100, (Number(metrics.integrated_lufs) + 30) / 20 * 100));
      $('lufsMeterFill').style.width = width + '%';
    }
    updateBands(metrics.band_energy || {});
  }

  function resetMetrics(message) {
    const value = message || 'Not measured yet';
    ['metricPeak','metricPlr','metricStereo','metricDc','metricClipping','metricLowWidth','metricHarsh'].forEach(id => $(id).textContent = value);
    $('metricLufs').textContent = message || 'Pending analysis';
    $('metricReadiness').textContent = message || 'Available after processing';
    $('lufsMeterFill').style.width = '0%';
    updateBands({});
  }

  function updateBands(bands) {
    $('bandSub').textContent = formatBand(bands.sub_20_60_hz);
    $('bandLowMid').textContent = formatBand(bands.low_mid_120_350_hz);
    $('bandPresence').textContent = formatBand(bands.presence_2000_5000_hz);
    $('bandHarsh').textContent = formatBand(bands.harsh_5000_9000_hz);
    $('bandAir').textContent = formatBand(bands.air_9000_16000_hz);
  }

  function setTimeline(steps) {
    const done = new Set(Array.isArray(steps) ? steps : []);
    document.querySelectorAll('#timelineList li').forEach(item => {
      item.classList.toggle('done', done.has(item.dataset.step));
    });
  }

  if (playPreviewBtn) {
    playPreviewBtn.addEventListener('click', async () => {
      if (!previewAudio || !previewAudio.src) return;
      try {
        await ensureAudioAnalyser();
        if (previewAudio.paused) {
          await previewAudio.play();
          playPreviewBtn.textContent = 'Pause';
          startVisualizer('playing');
        } else {
          previewAudio.pause();
          playPreviewBtn.textContent = 'Play';
          startVisualizer('result');
        }
      } catch (err) {
        visualizerStatus.textContent = 'Playback unavailable in this browser';
      }
    });
  }

  if (previewAudio) {
    previewAudio.addEventListener('ended', () => {
      playPreviewBtn.textContent = 'Play';
      startVisualizer('result');
    });
    previewAudio.addEventListener('pause', () => {
      if (playPreviewBtn) playPreviewBtn.textContent = 'Play';
    });
    previewAudio.addEventListener('play', () => {
      if (playPreviewBtn) playPreviewBtn.textContent = 'Pause';
      startVisualizer('playing');
    });
  }

  function setPreviewAudio(src, statusText) {
    if (!previewAudio) return;
    if (selectedObjectUrl && selectedObjectUrl !== src) {
      URL.revokeObjectURL(selectedObjectUrl);
      selectedObjectUrl = null;
    }
    if (src && src.startsWith('blob:')) selectedObjectUrl = src;
    previewAudio.pause();
    previewAudio.removeAttribute('src');
    if (src) previewAudio.src = src;
    previewAudio.load();
    if (playPreviewBtn) {
      playPreviewBtn.disabled = !src;
      playPreviewBtn.textContent = 'Play';
    }
    if (visualizerStatus) visualizerStatus.textContent = statusText || 'Visualizer ready';
  }

  function setGeneratedWaveform(seed) {
    const points = 180;
    let hash = 0;
    const text = seed || 'mmv2';
    for (let i = 0; i < text.length; i++) hash = ((hash << 5) - hash + text.charCodeAt(i)) | 0;
    const peaks = [];
    const rms = [];
    for (let i = 0; i < points; i++) {
      const t = i / Math.max(1, points - 1);
      const wobble = Math.sin(t * 18 + hash * .01) * .18 + Math.sin(t * 49 + hash * .004) * .09;
      const env = .24 + .62 * Math.pow(Math.sin(Math.PI * t), .35);
      const value = Math.max(.04, Math.min(.98, env * (.58 + wobble)));
      peaks.push(value);
      rms.push(value * (.45 + .12 * Math.sin(t * 24 + 1.7)));
    }
    visualizer.waveform = {peaks, rms, duration: previewAudio && previewAudio.duration ? previewAudio.duration : 0};
  }

  function loadWaveformArtifact(token) {
    if (!token) return;
    fetch('/api/preview/' + encodeURIComponent(token), {cache:'no-store'})
      .then(resp => resp.ok ? resp.json() : null)
      .then(data => {
        if (!data || !Array.isArray(data.peaks)) return;
        visualizer.waveform = data;
        if (visualizerStatus) {
          const duration = data.duration ? formatDuration(data.duration) : 'processed';
          visualizerStatus.textContent = 'Master waveform loaded · ' + duration;
        }
      })
      .catch(() => {});
  }

  async function ensureAudioAnalyser() {
    if (!previewAudio) return;
    if (!visualizer.audioCtx) {
      const AudioCtx = window.AudioContext || window.webkitAudioContext;
      if (!AudioCtx) throw new Error('AudioContext unavailable');
      visualizer.audioCtx = new AudioCtx();
      visualizer.analyser = visualizer.audioCtx.createAnalyser();
      visualizer.analyser.fftSize = 1024;
      visualizer.freqData = new Uint8Array(visualizer.analyser.frequencyBinCount);
    }
    if (!visualizer.source) {
      visualizer.source = visualizer.audioCtx.createMediaElementSource(previewAudio);
      visualizer.source.connect(visualizer.analyser);
      visualizer.analyser.connect(visualizer.audioCtx.destination);
    }
    if (visualizer.audioCtx.state === 'suspended') await visualizer.audioCtx.resume();
  }

  function startVisualizer(mode) {
    visualizer.mode = mode || visualizer.mode || 'idle';
    if (visualizer.raf) return;
    const tick = now => {
      visualizer.lastNow = now || 0;
      drawWavePlaceholder(now || 0);
      visualizer.raf = requestAnimationFrame(tick);
    };
    visualizer.raf = requestAnimationFrame(tick);
  }

  function drawWavePlaceholder(now) {
    const canvas = $('waveCanvas');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const w = canvas.width;
    const h = canvas.height;
    ctx.clearRect(0,0,w,h);
    drawVisualizerBackdrop(ctx, w, h, now);
    const waveform = visualizer.waveform || buildFallbackWaveform(now);
    drawWaveformLayer(ctx, w, h, waveform, now);
    drawSpectrumLayer(ctx, w, h, now);
    drawPlayhead(ctx, w, h, now);
    ctx.globalAlpha = 1;
  }

  function drawVisualizerBackdrop(ctx, w, h, now) {
    const bg = ctx.createLinearGradient(0,0,w,h);
    bg.addColorStop(0,'rgba(2,6,23,.88)');
    bg.addColorStop(.55,'rgba(30,27,75,.72)');
    bg.addColorStop(1,'rgba(5,46,64,.58)');
    ctx.fillStyle = bg;
    ctx.fillRect(0,0,w,h);
    ctx.globalAlpha = .28;
    ctx.strokeStyle = 'rgba(125,211,252,.22)';
    ctx.lineWidth = 1;
    for (let x = 0; x <= w; x += 36) {
      ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x + Math.sin((now || 0) * .001 + x) * 5, h); ctx.stroke();
    }
    for (let y = 18; y < h; y += 22) {
      ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke();
    }
    ctx.globalAlpha = 1;
  }

  function drawWaveformLayer(ctx, w, h, waveform, now) {
    const peaks = waveform.peaks || [];
    const rms = waveform.rms || peaks;
    if (!peaks.length) return;
    const center = h * .5;
    const barW = Math.max(2, w / peaks.length);
    const pulse = .88 + .12 * Math.sin((now || 0) * .004);
    for (let i = 0; i < peaks.length; i++) {
      const x = i * barW;
      const peak = Math.min(1, Math.max(.015, Number(peaks[i]) || 0));
      const r = Math.min(1, Math.max(.01, Number(rms[i]) || peak * .55));
      const height = Math.max(2, peak * h * .42 * pulse);
      const rmsHeight = Math.max(1, r * h * .32);
      const g = ctx.createLinearGradient(x, center - height, x, center + height);
      g.addColorStop(0,'rgba(103,232,249,.92)');
      g.addColorStop(.52,'rgba(168,85,247,.95)');
      g.addColorStop(1,'rgba(52,211,153,.78)');
      ctx.fillStyle = g;
      ctx.globalAlpha = .75;
      ctx.fillRect(x, center - height, Math.max(1, barW * .58), height * 2);
      ctx.fillStyle = 'rgba(255,255,255,.42)';
      ctx.globalAlpha = .34;
      ctx.fillRect(x, center - rmsHeight, Math.max(1, barW * .28), rmsHeight * 2);
    }
    ctx.globalAlpha = 1;
  }

  function drawSpectrumLayer(ctx, w, h, now) {
    let bins = null;
    if (visualizer.analyser && visualizer.freqData && previewAudio && !previewAudio.paused) {
      visualizer.analyser.getByteFrequencyData(visualizer.freqData);
      bins = visualizer.freqData;
    }
    const count = 28;
    const baseY = h - 6;
    for (let i = 0; i < count; i++) {
      const binValue = bins ? bins[Math.min(bins.length - 1, Math.floor(i / count * bins.length))] / 255 : 0;
      const idleValue = .2 + .24 * Math.sin((now || 0) * .0022 + i * .72);
      const value = visualizer.mode === 'processing'
        ? Math.max(idleValue, visualizer.progress * (.35 + i / count * .55))
        : (bins ? binValue : Math.max(.08, idleValue));
      const barH = Math.max(2, value * h * .28);
      const x = 10 + i * ((w - 20) / count);
      ctx.fillStyle = i < count * .28 ? 'rgba(34,211,238,.52)' : i < count * .72 ? 'rgba(168,85,247,.46)' : 'rgba(251,191,36,.42)';
      ctx.fillRect(x, baseY - barH, Math.max(2, (w - 28) / count * .45), barH);
    }
  }

  function drawPlayhead(ctx, w, h, now) {
    let progress = visualizer.mode === 'processing' ? visualizer.progress : 0;
    if (previewAudio && previewAudio.duration && Number.isFinite(previewAudio.duration)) {
      progress = previewAudio.currentTime / previewAudio.duration;
    } else if (visualizer.mode === 'idle' || visualizer.mode === 'error') {
      progress = ((now || 0) * .00008) % 1;
    }
    progress = Math.max(0, Math.min(1, progress || 0));
    const x = progress * w;
    ctx.globalAlpha = .95;
    ctx.strokeStyle = 'rgba(255,255,255,.86)';
    ctx.lineWidth = 1.5;
    ctx.beginPath(); ctx.moveTo(x, 6); ctx.lineTo(x, h - 6); ctx.stroke();
    const glow = ctx.createRadialGradient(x, h*.5, 0, x, h*.5, 42);
    glow.addColorStop(0,'rgba(56,189,248,.32)');
    glow.addColorStop(1,'rgba(56,189,248,0)');
    ctx.fillStyle = glow;
    ctx.fillRect(Math.max(0, x - 45), 0, 90, h);
    ctx.globalAlpha = 1;
  }

  function buildFallbackWaveform(now) {
    const points = 150;
    const peaks = [];
    const rms = [];
    for (let i = 0; i < points; i++) {
      const t = i / points;
      const value = .25 + .42 * Math.abs(Math.sin(t * 16 + (now || 0) * .0012)) + .12 * Math.sin(t * 53);
      peaks.push(Math.max(.05, Math.min(.96, value)));
      rms.push(Math.max(.03, value * .48));
    }
    return {peaks, rms};
  }

  function formatDuration(seconds) {
    const total = Math.max(0, Math.round(Number(seconds) || 0));
    const min = Math.floor(total / 60);
    const sec = String(total % 60).padStart(2, '0');
    return min + ':' + sec;
  }

  function formatMetric(value, suffix) {
    if (value == null || Number.isNaN(Number(value))) return 'Not measured';
    return Number(value).toFixed(2) + suffix;
  }
  function formatBand(value) {
    if (value == null) return 'Pending';
    return (Number(value) * 100).toFixed(1) + '%';
  }
  function riskLabel(value, threshold) {
    return Number(value) > threshold ? 'Review' : 'Clean';
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
<title>MMV2 Audio Quality Engine</title>
<style>{css}</style>
</head>
<body data-max-size="{max_size}">
<div class="app-shell">
<header class="hero-header">
  <p class="product-kicker">MMV2 Audio Quality Engine</p>
  <h1 class="hero-title quality-hero-title">Release-Ready Stereo Mastering</h1>
  <p class="hero-subtitle">Upload a finished stereo mix and generate a loudness-safe master with transparent quality metrics.</p>
  <span class="beta-badge">Local engine beta</span>
</header>
<main class="quality-console">
  <section class="console-card" aria-label="Mastering upload console">
    <div class="console-topbar">
      <div class="engine-brand">
        <span class="logo-mark">MMV2</span>
        <span class="engine-version">Engine {engine_version}</span>
      </div>
      <div class="engine-status"><span class="active-dot"></span>ENGINE ACTIVE</div>
    </div>

    <div id="dropzone" class="dropzone quality-dropzone">
      <div class="dropzone-content">
        <div class="dropzone-icon">&#9835;</div>
        <p>Drag and drop your audio track here</p>
        <p class="hint">WAV, FLAC, AIFF or MP3 &middot; max {max_size_mb} MB</p>
        <input type="file" id="fileInput" accept=".mp3,.wav,.flac,.aiff,.aif" hidden>
      </div>
    </div>

    <div class="console-hint-row">
      <span>Mono and stereo accepted</span>
      <span>Metadata parsed safely</span>
      <span>Preview available</span>
    </div>
  </section>

  <section class="visualizer-card" aria-label="Audio preview visualizer">
    <div class="visualizer-header">
      <span class="visualizer-title">Preview Visualizer</span>
      <span id="visualizerStatus" class="visualizer-status">Drop audio to preview waveform</span>
    </div>
    <canvas id="waveCanvas" class="wave-canvas" width="720" height="104" aria-hidden="true"></canvas>
    <div class="visualizer-controls">
      <button id="playPreviewBtn" class="visualizer-button" type="button" disabled>Play</button>
    </div>
    <audio id="previewAudio" preload="metadata" hidden></audio>
  </section>

  <section id="options" class="control-strip" hidden>
    <div class="control-group mode-group" role="group" aria-label="Processing mode">
      <button class="mode-button" data-mode="analyze_only" type="button">Analyze Only</button>
      <button class="mode-button active" data-mode="safe_master" type="button">Safe Master</button>
      <button class="mode-button" data-mode="naturalize" type="button">Naturalize Pass</button>
      <button class="mode-button" data-mode="full_release" type="button">Full Release Pass</button>
      <button class="mode-button" data-mode="metadata_clean" type="button">Metadata Clean</button>
    </div>
    <div class="control-grid">
      <label>Output format
        <select id="formatSelect">
          <option value="preserve" selected>Preserve</option>
          <option value="wav">WAV</option>
          <option value="flac">FLAC</option>
          <option value="mp3">MP3</option>
        </select>
      </label>
      <label>Loudness target
        <select id="loudnessTarget">
          <option value="streaming_safe" selected>Streaming Safe</option>
          <option value="club_loud">Club/Loud</option>
          <option value="conservative">Conservative</option>
        </select>
      </label>
      <label>True peak ceiling
        <select id="truePeakCeiling">
          <option value="-1.0">-1.0 dBTP</option>
          <option value="-1.5" selected>-1.5 dBTP</option>
          <option value="-2.0">-2.0 dBTP</option>
        </select>
      </label>
      <label>Sample rate
        <select id="sampleRateOverride">
          <option value="preserve" selected>Preserve</option>
          <option value="44100">44.1 kHz</option>
          <option value="48000">48 kHz</option>
        </select>
      </label>
      <label>Bit depth
        <select id="bitDepthOverride">
          <option value="preserve" selected>Preserve</option>
          <option value="16">16-bit</option>
          <option value="24">24-bit</option>
          <option value="32">32-bit float</option>
        </select>
      </label>
    </div>
    <div class="action-row">
      <button id="processBtn" class="btn-primary">Analyze &amp; Master</button>
      <button id="analyzeOnlyBtn" class="btn-secondary" type="button">Analyze Only</button>
      <button id="safeMasterBtn" class="btn-secondary" type="button">Safe Master</button>
      <button id="naturalizeBtn" class="btn-secondary" type="button">Naturalize Pass</button>
    </div>
  </section>

  <section class="analysis-preview" aria-label="Realtime analysis preview">
    <div class="meter-card lufs-card">
      <span class="metric-label">Integrated LUFS</span>
      <strong id="metricLufs">Pending analysis</strong>
      <small id="metricTarget">Target: Streaming Safe</small>
      <div class="lufs-meter"><span id="lufsMeterFill"></span></div>
    </div>
    <div class="metric-card"><span>True Peak</span><strong id="metricPeak">Not measured yet</strong></div>
    <div class="metric-card"><span>PLR / Crest</span><strong id="metricPlr">Not measured yet</strong></div>
    <div class="metric-card"><span>Stereo Correlation</span><strong id="metricStereo">Not measured yet</strong></div>
    <div class="metric-card"><span>DC Offset</span><strong id="metricDc">Not measured yet</strong></div>
    <div class="metric-card"><span>Clipping Count</span><strong id="metricClipping">Not measured yet</strong></div>
    <div class="metric-card"><span>Low-End Width</span><strong id="metricLowWidth">Not measured yet</strong></div>
    <div class="metric-card"><span>Harshness Risk</span><strong id="metricHarsh">Not measured yet</strong></div>
    <div class="metric-card readiness"><span>Release Readiness</span><strong id="metricReadiness">Available after processing</strong></div>
  </section>

  <section class="spectral-risk-grid" aria-label="Spectral risk checks">
    <div><span>Ultra Low</span><strong id="bandSub">Pending</strong></div>
    <div><span>Low Mid</span><strong id="bandLowMid">Pending</strong></div>
    <div><span>Presence</span><strong id="bandPresence">Pending</strong></div>
    <div><span>Harsh</span><strong id="bandHarsh">Pending</strong></div>
    <div><span>Air</span><strong id="bandAir">Pending</strong></div>
  </section>

  <section class="timeline-panel">
    <h2>Processing Timeline</h2>
    <ol id="timelineList">
      <li data-step="upload">Upload validated</li>
      <li data-step="metadata">Metadata parsed</li>
      <li data-step="loudness">Loudness measured</li>
      <li data-step="dc">DC offset checked</li>
      <li data-step="spectrum">Low/high spectrum checked</li>
      <li data-step="render">Master chain rendered</li>
      <li data-step="limit">Preview limited</li>
      <li data-step="report">Report generated</li>
    </ol>
  </section>

  <div id="status" class="status-area" hidden>
    <div class="spinner" id="spinner"></div>
    <p id="statusText">Processing...</p>
    <div class="progress-bar"><div class="progress-fill" id="progressFill"></div></div>
  </div>

  <div id="result" class="result-area" hidden>
    <div class="success-icon">&#9989;</div>
    <p id="resultText"></p>
    <div id="statsPanel" class="stats-panel"></div>
    <div class="download-actions">
      <a id="downloadLink" class="btn-download" href="#">Download Master</a>
      <a id="jsonReportLink" class="btn-download subtle" href="#" hidden>Download Report JSON</a>
      <a id="htmlReportLink" class="btn-download subtle" href="#" hidden>Download HTML Report</a>
    </div>
    <button id="resetBtn" class="btn-secondary">Process another file</button>
  </div>

  <div id="error" class="error-area error-state" hidden>
    <div class="error-icon">&#10060;</div>
    <p id="errorText"></p>
    <button id="retryBtn" class="btn-secondary retry-button">Try again</button>
  </div>
</main>
<footer class="footer-credits">
  <div>MMV2 Audio Quality Engine — Local stereo mastering and release checks</div>
  <div class="credit-secondary">Private beta for controlled audio quality review</div>
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


def _parse_quality_options(form: Any) -> dict[str, Any]:
    """Parse optional quality-engine controls with conservative defaults."""
    loudness_target = form.get("loudness_target", "streaming_safe")
    if loudness_target not in LOUDNESS_TARGETS:
        loudness_target = "streaming_safe"

    true_peak_ceiling = form.get("true_peak_ceiling", "-1.5")
    if true_peak_ceiling not in TRUE_PEAK_CEILINGS:
        true_peak_ceiling = "-1.5"

    sample_rate_override = form.get("sample_rate_override", "preserve")
    if sample_rate_override not in SAMPLE_RATE_OVERRIDES:
        sample_rate_override = "preserve"

    bit_depth_override = form.get("bit_depth_override", "preserve")
    if bit_depth_override not in BIT_DEPTH_OVERRIDES:
        bit_depth_override = "preserve"

    return {
        "loudness_target": loudness_target,
        "loudness_target_label": str(LOUDNESS_TARGETS[loudness_target]["label"]),
        "true_peak_ceiling": float(true_peak_ceiling),
        "sample_rate_override": sample_rate_override,
        "bit_depth_override": bit_depth_override,
    }


def _gpu_status() -> dict[str, Any]:
    """Return CUDA status without making torch a hard web dependency."""
    status: dict[str, Any] = {
        "available": False,
        "backend": "cpu",
        "device": None,
        "vram_total_gb": None,
        "vram_free_gb": None,
        "compute_capability": None,
        "error": None,
    }
    try:
        import torch
    except Exception as exc:
        status["error"] = str(exc)
        return status

    try:
        if not torch.cuda.is_available():
            return status
        device_index = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(device_index)
        free_bytes, total_bytes = torch.cuda.mem_get_info(device_index)
        status.update(
            {
                "available": True,
                "backend": "cuda",
                "device": torch.cuda.get_device_name(device_index),
                "vram_total_gb": round(total_bytes / 1024**3, 2),
                "vram_free_gb": round(free_bytes / 1024**3, 2),
                "compute_capability": f"{props.major}.{props.minor}",
                "error": None,
            }
        )
    except Exception as exc:
        status["error"] = str(exc)
    return status


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
            engine_version=ENGINE_VERSION,
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
        gpu = _gpu_status()
        return jsonify({
            "busy": busy,
            "version": "2.0.0",
            "engine_version": ENGINE_VERSION,
            "max_file_size_mb": max_size_mb,
            "gpu_available": gpu["available"],
            "gpu": gpu,
            "metadata_modes": [METADATA_CLEAN_MODE],
            "quality_modes": sorted(QUALITY_MODES),
            "loudness_targets": LOUDNESS_TARGETS,
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
            return jsonify({"error": "Processing failed. Please try a different file."}), 500

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
        return _serve_registered_file(app, token, as_attachment=True, consume=True)

    @app.route("/api/preview/<token>")
    def api_preview(token: str) -> tuple:
        return _serve_registered_file(app, token, as_attachment=False, consume=False)

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
        return fail_response({"error": "Unsupported file type. Use WAV, FLAC, AIFF, or MP3."}, 400)

    # Parse options
    output_format = request.form.get("format", "preserve")
    if output_format not in OUTPUT_FORMATS:
        output_format = "preserve"

    paranoid = request.form.get("paranoid", "false").lower() == "true"
    mode = request.form.get("mode", "safe_master")
    if mode in LEGACY_MODE_ALIASES:
        mode = METADATA_CLEAN_MODE
    elif mode not in QUALITY_MODES:
        mode = "safe_master"
    quality_options = _parse_quality_options(request.form)

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
                "message": "Upload complete. Waiting for local engine...",
                "created": time.time(),
                "engine_version": ENGINE_VERSION,
                "mode": mode,
                "output_format": output_format,
                **quality_options,
                "processing_steps": ["upload"],
            }

        worker = threading.Thread(
            target=_process_upload_job,
            args=(
                app,
                job_id,
                input_path,
                safe_name,
                output_format,
                paranoid,
                mode,
                quality_options,
                processing_lock,
            ),
            daemon=True,
        )
        worker.start()
        return jsonify({
            "success": True,
            "job_id": job_id,
            "message": "Upload accepted. Local processing started.",
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
    mode: str,
    quality_options: dict[str, Any],
    processing_lock: threading.Lock,
) -> None:
    """Run sanitization in a background thread."""
    try:
        if mode in QUALITY_MODES:
            result = _process_quality_job(
                app,
                job_id,
                input_path,
                safe_name,
                output_format,
                mode,
                quality_options,
            )
        else:
            _update_job(
                app,
                job_id,
                status="processing",
                progress=35,
                message="Removing file metadata...",
                processing_steps=["upload", "metadata"],
            )
            fmt = None if output_format == "preserve" else output_format
            result = _process_metadata_clean_job(
                app,
                job_id,
                input_path,
                safe_name,
                fmt,
                strict_verification=paranoid,
            )

        if not result.get("success"):
            _update_job(
                app,
                job_id,
                status="failed",
                progress=100,
                message="Processing failed.",
                error=result.get("error", "Processing failed."),
            )
            return

        _update_job(app, job_id, progress=95, message="Preparing download...")
        stem = Path(safe_name).stem
        token = None
        download_name = None
        output_path_value = result.get("output_file")
        if output_path_value:
            output_path = Path(output_path_value)
            ext = output_path.suffix
            suffix = "master" if mode in QUALITY_MODES else "clean"
            download_name = f"{stem}_{suffix}{ext}"
            token = _register_download(app, output_path, download_name)

        final_steps = result.get("timeline_steps")
        if not final_steps:
            final_steps = (
                ["upload", "metadata", "loudness", "dc", "spectrum", "render", "limit", "report"]
                if mode in QUALITY_MODES
                else ["upload", "metadata", "render", "report"]
            )

        _update_job(
            app,
            job_id,
            status="complete",
            progress=100,
            message="Processing complete.",
            processing_steps=final_steps,
            result={
                "success": True,
                "engine_version": ENGINE_VERSION,
                "mode": mode,
                "output_format": output_format,
                **quality_options,
                "download_token": token,
                "filename": download_name,
                "stats": result.get("stats", {}),
                "metrics_before": result.get("metrics_before"),
                "metrics_after": result.get("metrics_after"),
                "waveform_artifact": result.get("waveform_artifact"),
                "report_artifacts": result.get("report_artifacts", {}),
                "processing_steps": final_steps,
                "preview_protected": result.get("preview_protected"),
                "peak_safety": result.get("peak_safety"),
            },
        )
    except Exception as exc:
        app.logger.exception("Background upload job failed")
        _update_job(
            app,
            job_id,
            status="failed",
            progress=100,
            message="Processing failed.",
            error=str(exc),
        )
    finally:
        input_path.unlink(missing_ok=True)
        processing_lock.release()


def _process_metadata_clean_job(
    app: Flask,
    job_id: str,
    input_path: Path,
    safe_name: str,
    output_format: Optional[str],
    *,
    strict_verification: bool,
) -> Dict[str, Any]:
    """Run a metadata-only clean export for web uploads."""
    from .sanitization.metadata_cleaner import MetadataCleaner

    _update_job(
        app,
        job_id,
        progress=55,
        message="Writing metadata-clean export...",
        processing_steps=["upload", "metadata", "render"],
    )

    temp_dir: Path = app.config["UPLOAD_FOLDER"]
    output_ext = _metadata_clean_output_extension(input_path, output_format)
    output_path = temp_dir / f"{uuid.uuid4().hex}_{Path(safe_name).stem}.clean{output_ext}"
    before_hash = _sha256_path(input_path)

    if output_format and output_ext != input_path.suffix.lower():
        result = _transcode_without_metadata(input_path, output_path, output_ext)
    else:
        result = MetadataCleaner().clean_file(input_path, output_path)
    if not result.get("success"):
        return {
            "success": False,
            "error": "; ".join(result.get("errors", [])) or "Metadata clean export failed.",
            "stats": result,
        }

    cleaner = MetadataCleaner()
    metadata_clean = not cleaner._verify_metadata_present(output_path)
    after_hash = _sha256_path(output_path)
    if strict_verification and not metadata_clean:
        output_path.unlink(missing_ok=True)
        return {
            "success": False,
            "error": "Metadata verification failed after export.",
            "stats": result,
        }

    stats = {
        "processing_engine": "metadata_clean_export",
        "gpu_acceleration": False,
        "gpu_device": None,
        "metadata_removed": int(result.get("tags_removed", 0)) + int(result.get("chunks_removed", 0)),
        "metadata_clean": metadata_clean,
        "output_hash_changed": before_hash != after_hash,
        "methods_used": list(result.get("methods_used", [])),
        "strict_metadata_verification": bool(strict_verification),
    }
    methods_used = list(stats.get("methods_used", []))
    if strict_verification:
        methods_used.append("metadata_verification")
    stats["methods_used"] = methods_used

    return {
        "success": True,
        "output_file": str(output_path),
        "stats": stats,
        "timeline_steps": ["upload", "metadata", "render", "report"],
        "preview_protected": "N/A",
        "peak_safety": "N/A",
        "metrics_before": None,
        "metrics_after": None,
    }


def _process_quality_job(
    app: Flask,
    job_id: str,
    input_path: Path,
    safe_name: str,
    output_format: str,
    mode: str,
    quality_options: dict[str, Any],
) -> Dict[str, Any]:
    """Run the local audio quality engine for web jobs."""
    from audio_engine.analysis.readiness import analyze_quality
    from audio_engine.dsp.pipeline import render_safe_master
    from audio_engine.guardrails.limits import DEFAULT_LIMITS, GuardrailLimits
    from audio_engine.naturalize.movement import render_naturalized_master
    from audio_engine.reports.html_report import write_html_report
    from audio_engine.reports.json_report import write_json_report

    temp_dir: Path = app.config["UPLOAD_FOLDER"]
    stem = Path(safe_name).stem
    output_ext = _quality_output_extension(input_path, output_format)
    output_path = temp_dir / f"{uuid.uuid4().hex}_{stem}.master{output_ext}"
    json_report_path = temp_dir / f"{uuid.uuid4().hex}_{stem}.quality.json"
    html_report_path = temp_dir / f"{uuid.uuid4().hex}_{stem}.quality.html"
    waveform_path = temp_dir / f"{uuid.uuid4().hex}_{stem}.waveform.json"

    limits = GuardrailLimits(
        **{
            **DEFAULT_LIMITS.to_dict(),
            "limiter_ceiling_dbtp": float(quality_options["true_peak_ceiling"]),
            "export_default_bit_depth": _bit_depth_for_writer(
                str(quality_options["bit_depth_override"])
            ),
        }
    )
    target_lufs = float(LOUDNESS_TARGETS[str(quality_options["loudness_target"])]["target_lufs"])
    output_sample_rate = _sample_rate_for_writer(str(quality_options["sample_rate_override"]))

    _update_job(
        app,
        job_id,
        status="processing",
        progress=35,
        message="Metadata parsed safely. Measuring loudness...",
        processing_steps=["upload", "metadata", "loudness"],
    )
    before = analyze_quality(input_path)
    waveform_token = None
    try:
        waveform = _build_waveform_artifact(input_path)
        write_json_report(waveform, waveform_path)
        waveform_token = _register_download(app, waveform_path, f"{stem}_waveform.json")
    except Exception as exc:
        app.logger.info("Waveform artifact skipped: %s", exc)

    if mode == "analyze_only":
        report = {
            "action": "analyze_only",
            "input": str(input_path),
            "engine_version": ENGINE_VERSION,
            "mode": mode,
            "quality_options": quality_options,
            "metrics": before,
            "metrics_before": before,
            "metrics_after": before,
            "processing_steps": ["upload", "metadata", "loudness", "dc", "spectrum", "report"],
        }
    else:
        _update_job(
            app,
            job_id,
            progress=62,
            message="Rendering conservative master chain...",
            processing_steps=["upload", "metadata", "loudness", "dc", "spectrum", "render"],
        )
        if mode == "naturalize":
            report = render_naturalized_master(
                input_path,
                output_path,
                limits=limits,
                target_lufs=target_lufs,
                output_sample_rate=output_sample_rate,
            )
        else:
            report = render_safe_master(
                input_path,
                output_path,
                limits=limits,
                target_lufs=target_lufs,
                output_sample_rate=output_sample_rate,
            )

    report["engine_version"] = ENGINE_VERSION
    report["mode"] = mode
    report["output_format"] = output_format
    report["quality_options"] = quality_options
    report["sample_rate_override"] = quality_options["sample_rate_override"]
    report["bit_depth_override"] = quality_options["bit_depth_override"]

    _update_job(
        app,
        job_id,
        progress=82,
        message="Generating quality reports...",
        processing_steps=["upload", "metadata", "loudness", "dc", "spectrum", "render", "limit", "report"],
    )
    write_json_report(report, json_report_path)
    write_html_report(report if mode != "analyze_only" else {"metrics": before}, html_report_path)
    json_token = _register_download(app, json_report_path, f"{stem}_quality_report.json")
    html_token = _register_download(app, html_report_path, f"{stem}_quality_report.html")

    metrics_before = report.get("before") or report.get("metrics_before") or before
    metrics_after = report.get("after") or report.get("metrics_after") or metrics_before
    loudness_delta = None
    if metrics_before and metrics_after:
        loudness_delta = float(metrics_after["integrated_lufs"] - metrics_before["integrated_lufs"])

    stats = {
        "processing_engine": "mmv2_audio_quality_engine",
        "gpu_acceleration": False,
        "metadata_removed": 0,
        "integrated_lufs": metrics_after.get("integrated_lufs"),
        "short_term_lufs_curve": metrics_after.get("short_term_loudness_curve"),
        "true_peak_dbtp": metrics_after.get("estimated_true_peak_dbtp"),
        "loudness_target": quality_options["loudness_target"],
        "target_lufs": target_lufs,
        "loudness_delta": loudness_delta,
        "channels": metrics_after.get("channels"),
        "sample_rate": metrics_after.get("sample_rate"),
        "dc_offset": metrics_after.get("dc_offset"),
        "clipping_sample_count": metrics_after.get("clipping_sample_count"),
    }

    output_file = None if mode == "analyze_only" else str(output_path)
    timeline_steps = (
        ["upload", "metadata", "loudness", "dc", "spectrum", "report"]
        if mode == "analyze_only"
        else ["upload", "metadata", "loudness", "dc", "spectrum", "render", "limit", "report"]
    )
    return {
        "success": True,
        "output_file": output_file,
        "stats": stats,
        "metrics_before": metrics_before,
        "metrics_after": metrics_after,
        "waveform_artifact": {
            "json_download_token": waveform_token,
        } if waveform_token else None,
        "report_artifacts": {
            "json_download_token": json_token,
            "html_download_token": html_token,
        },
        "processing_steps": report.get("processing_steps", []),
        "timeline_steps": timeline_steps,
        "preview_protected": mode != "analyze_only",
        "peak_safety": "Protected" if mode != "analyze_only" else "Analysis only",
    }


def _register_download(app: Flask, file_path: Path, filename: str) -> str:
    """Register a downloadable artifact and return its token."""
    token = uuid.uuid4().hex
    with app.config["DOWNLOAD_REGISTRY_LOCK"]:
        app.config["DOWNLOAD_REGISTRY"][token] = {
            "path": str(file_path),
            "filename": filename,
            "created": time.time(),
        }
    return token


def _serve_registered_file(
    app: Flask,
    token: str,
    *,
    as_attachment: bool,
    consume: bool,
) -> tuple:
    """Serve a registered artifact, optionally consuming its token."""
    registry: dict = app.config["DOWNLOAD_REGISTRY"]
    with app.config["DOWNLOAD_REGISTRY_LOCK"]:
        _cleanup_download_registry(registry)
        entry = registry.pop(token, None) if consume else registry.get(token)

    if entry is None:
        return jsonify({"error": "File not found or expired."}), 404

    file_path = Path(entry["path"])
    if not file_path.exists():
        return jsonify({"error": "File not found or expired."}), 404

    response = send_file(
        str(file_path),
        as_attachment=as_attachment,
        download_name=entry["filename"],
    )
    if consume:
        response.call_on_close(lambda: file_path.unlink(missing_ok=True))
    return response


def _quality_output_extension(input_path: Path, output_format: str) -> str:
    """Choose a safe local output extension for the quality engine."""
    if output_format in {"wav", "flac", "mp3"}:
        return f".{output_format}"
    if output_format == "preserve" and input_path.suffix.lower() in {".wav", ".flac", ".aiff", ".aif"}:
        return input_path.suffix.lower()
    return ".wav"


def _metadata_clean_output_extension(input_path: Path, output_format: Optional[str]) -> str:
    """Choose an output extension for metadata-only export."""
    if output_format in {"wav", "flac", "mp3"}:
        return f".{output_format}"
    if input_path.suffix.lower() in {".wav", ".flac", ".aiff", ".aif", ".mp3"}:
        return input_path.suffix.lower()
    return ".wav"


def _transcode_without_metadata(input_path: Path, output_path: Path, output_ext: str) -> dict[str, Any]:
    """Transcode audio while asking ffmpeg to drop container metadata."""
    from pydub import AudioSegment

    output_path.parent.mkdir(parents=True, exist_ok=True)
    export_format = output_ext.lstrip(".").replace("aif", "aiff")
    audio = AudioSegment.from_file(str(input_path))
    parameters = ["-map_metadata", "-1"]
    if export_format == "mp3":
        parameters.extend(["-write_id3v1", "0", "-id3v2_version", "0", "-write_xing", "0"])
    audio.export(str(output_path), format=export_format, parameters=parameters)
    return {
        "success": True,
        "tags_removed": 0,
        "chunks_removed": 0,
        "methods_used": ["metadata_free_transcode"],
        "errors": [],
    }


def _sha256_path(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _bit_depth_for_writer(bit_depth_override: str) -> int:
    if bit_depth_override == "16":
        return 16
    if bit_depth_override == "32":
        return 32
    return 24


def _sample_rate_for_writer(sample_rate_override: str) -> Optional[int]:
    if sample_rate_override in {"44100", "48000"}:
        return int(sample_rate_override)
    return None


def _build_waveform_artifact(input_path: Path, points: int = 240) -> dict[str, Any]:
    """Build lightweight waveform peaks/RMS arrays for frontend/report use."""
    import numpy as np
    import soundfile as sf

    audio, sample_rate = sf.read(str(input_path), dtype="float32", always_2d=True)
    mono = np.mean(audio, axis=1)
    if mono.size == 0:
        raise ValueError("Cannot build waveform for empty audio.")
    gpu_artifact = _build_waveform_artifact_cuda(mono, int(sample_rate), int(audio.shape[1]), points)
    if gpu_artifact is not None:
        return gpu_artifact
    chunk = max(1, int(np.ceil(mono.size / points)))
    peaks = []
    rms = []
    for start in range(0, mono.size, chunk):
        frame = mono[start : start + chunk]
        peaks.append(float(np.max(np.abs(frame))) if frame.size else 0.0)
        rms.append(float(np.sqrt(np.mean(frame**2))) if frame.size else 0.0)
    return {
        "peaks": peaks,
        "rms": rms,
        "duration": float(mono.size / sample_rate),
        "sample_rate": int(sample_rate),
        "channels": int(audio.shape[1]),
        "backend": "cpu",
    }


def _build_waveform_artifact_cuda(
    mono_audio: Any,
    sample_rate: int,
    channels: int,
    points: int,
) -> Optional[dict[str, Any]]:
    """Use CUDA for small waveform summaries when torch is available."""
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        mono_bytes = int(getattr(mono_audio, "nbytes", 0))
        free_bytes, _total_bytes = torch.cuda.mem_get_info()
        if mono_bytes <= 0 or mono_bytes > 192 * 1024 * 1024 or free_bytes < mono_bytes * 3:
            return None
        device = torch.device("cuda")
        tensor = torch.as_tensor(mono_audio, device=device, dtype=torch.float32)
        padded = int(((tensor.numel() + points - 1) // points) * points)
        if padded != tensor.numel():
            tensor = torch.nn.functional.pad(tensor, (0, padded - tensor.numel()))
        frames = tensor.reshape(points, -1)
        peaks = torch.max(torch.abs(frames), dim=1).values.detach().cpu().tolist()
        rms = torch.sqrt(torch.mean(frames * frames, dim=1)).detach().cpu().tolist()
        return {
            "peaks": [float(value) for value in peaks],
            "rms": [float(value) for value in rms],
            "duration": float(len(mono_audio) / sample_rate),
            "sample_rate": int(sample_rate),
            "channels": int(channels),
            "backend": "cuda",
        }
    except Exception:
        return None


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

    console = ConsoleManager()

    console.info("MMV2 Audio Quality Engine web interface")
    console.info("   Use only on files you own or have explicit permission to process\n")

    if host != "127.0.0.1":
        console.warning(
            f"Binding to {host} — the server will be accessible from the network!"
        )
        console.warning(
            "For production hosting, run the Flask app behind gunicorn/uwsgi and a "
            "reverse proxy instead of exposing Werkzeug directly."
        )

    max_mb = max_file_size // (1024 * 1024)
    console.success(f"Starting MMV2 web server at http://{host}:{port}")
    console.info(f"   Max upload size: {max_mb} MB")
    console.info("   Press Ctrl+C to stop\n")

    app = create_app(max_file_size=max_file_size)

    # debug=False is mandatory — Werkzeug debugger enables remote code execution
    app.run(host=host, port=port, debug=False, threaded=True)
