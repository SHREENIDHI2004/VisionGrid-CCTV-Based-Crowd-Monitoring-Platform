document.addEventListener('DOMContentLoaded', () => {

  // ── Tab Navigation ──────────────────────────────────────────────────────
  const tabs = document.querySelectorAll('.tab-btn');
  tabs.forEach(tab => {
    tab.addEventListener('click', () => {
      tabs.forEach(t => t.classList.remove('active'));
      document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));

      tab.classList.add('active');
      const targetId = `tab-${tab.getAttribute('data-tab')}`;
      const targetContent = document.getElementById(targetId);
      if (targetContent) {
        targetContent.classList.add('active');
      }
    });
  });

  // Global Status Indicator Helpers
  const statusDot = document.getElementById('status-dot');
  const statusTxt = document.getElementById('status-txt');
  function setGlobalStatus(msg, color) {
    statusTxt.textContent = msg;
    statusDot.className = 'dot' + (color ? ' ' + color : '');
  }

  // ════════════════════════════════════════════════════════════════════════
  // TAB 1 — IMAGE FUNCTIONALITY
  // ════════════════════════════════════════════════════════════════════════
  const imgDropZone   = document.getElementById('img-drop-zone');
  const imgFileInput  = document.getElementById('img-file-input');
  const imgBrowse     = document.getElementById('img-browse');
  const imgOriginal   = document.getElementById('img-original');
  const imgResult     = document.getElementById('img-result');
  const imgDownload   = document.getElementById('img-download');
  const imgOverlay    = document.getElementById('img-overlay');

  const imgConf       = document.getElementById('img-conf');
  const imgIou        = document.getElementById('img-iou');

  const imgStatTotal  = document.getElementById('img-stat-total');
  const imgStatAvg    = document.getElementById('img-stat-avg');
  const imgStatMax    = document.getElementById('img-stat-max');
  const imgStatMin    = document.getElementById('img-stat-min');
  const imgTableBody  = document.getElementById('img-table-body');

  let imgRawBase64    = null; // For downloading the result

  // File selection triggers
  imgBrowse.addEventListener('click', () => imgFileInput.click());
  imgDropZone.addEventListener('click', (e) => {
    if (e.target !== imgBrowse) imgFileInput.click();
  });

  // Drag & drop handlers
  ['dragenter', 'dragover'].forEach(eventName => {
    imgDropZone.addEventListener(eventName, (e) => {
      e.preventDefault();
      imgDropZone.classList.add('dragover');
    }, false);
  });
  ['dragleave', 'drop'].forEach(eventName => {
    imgDropZone.addEventListener(eventName, (e) => {
      e.preventDefault();
      imgDropZone.classList.remove('dragover');
    }, false);
  });

  imgDropZone.addEventListener('drop', (e) => {
    const dt = e.dataTransfer;
    const files = dt.files;
    if (files.length > 0) {
      handleImageUpload(files[0]);
    }
  });

  imgFileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
      handleImageUpload(e.target.files[0]);
    }
  });

  function handleImageUpload(file) {
    if (!file.type.startsWith('image/')) {
      alert('Please upload an image file.');
      return;
    }
    // Show original preview
    const reader = new FileReader();
    reader.onload = (e) => {
      imgOriginal.src = e.target.result;
    };
    reader.readAsDataURL(file);

    // Send to backend
    runImageInference(file);
  }

  async function runImageInference(file) {
    imgOverlay.classList.remove('hidden');
    imgDownload.disabled = true;
    setGlobalStatus('Analyzing Image...', 'org');

    const formData = new FormData();
    formData.append('image', file);
    formData.append('conf', imgConf.value);
    formData.append('iou', imgIou.value);

    try {
      const response = await fetch('/detect', {
        method: 'POST',
        body: formData
      });
      const data = await response.json();
      if (data.error) throw new Error(data.error);

      // Display result
      imgResult.src = data.image;
      imgRawBase64 = data.image;
      imgDownload.disabled = false;

      // Update metrics
      const detections = data.detections || [];
      const total = detections.length;
      imgStatTotal.textContent = total;

      if (total > 0) {
        const confs = detections.map(d => d.confidence);
        const avg = confs.reduce((a, b) => a + b, 0) / total;
        const max = Math.max(...confs);
        const min = Math.min(...confs);

        imgStatAvg.textContent = `${(avg * 100).toFixed(0)}%`;
        imgStatMax.textContent = `${(max * 100).toFixed(0)}%`;
        imgStatMin.textContent = `${(min * 100).toFixed(0)}%`;

        // Fill detections table
        imgTableBody.innerHTML = detections.map((det, idx) => {
          const confVal = (det.confidence * 100).toFixed(0);
          const boxCoords = det.box.map(v => Math.round(v)).join(', ');
          return `
            <tr>
              <td><span class="pid-badge">#${idx + 1}</span></td>
              <td>
                <div class="conf-wrap">
                  <span style="min-width:35px;font-weight:600;">${confVal}%</span>
                  <div class="conf-bar-bg">
                    <div class="conf-bar-fg" style="width: ${confVal}%"></div>
                  </div>
                </div>
              </td>
              <td><code>[${boxCoords}]</code></td>
            </tr>
          `;
        }).join('');
      } else {
        imgStatAvg.textContent = '0%';
        imgStatMax.textContent = '0%';
        imgStatMin.textContent = '0%';
        imgTableBody.innerHTML = `<tr><td colspan="3" class="empty-td">No persons detected</td></tr>`;
      }

      setGlobalStatus('Image Analyzed', 'green');
    } catch (e) {
      console.error(e);
      alert('Error analyzing image: ' + e.message);
      setGlobalStatus('Analysis Failed', 'red');
    } finally {
      imgOverlay.classList.add('hidden');
    }
  }

  // Programmatic base64 downloader (no browser security warnings)
  imgDownload.addEventListener('click', (e) => {
    e.preventDefault();
    if (!imgRawBase64) return;

    const [meta, b64] = imgRawBase64.split(';base64,');
    if (!b64) return;
    const mime = meta.split(':')[1] || 'image/jpeg';
    const binary = atob(b64);
    const bytes = new Uint8Array(binary.length);
    for (let i = 0; i < binary.length; i++) {
      bytes[i] = binary.charCodeAt(i);
    }
    const blob = new Blob([bytes], { type: mime });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'person_detection_result.jpg';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    setTimeout(() => URL.revokeObjectURL(url), 2000);
  });


  // ════════════════════════════════════════════════════════════════════════
  // TAB 2 — CCTV FUNCTIONALITY
  // ════════════════════════════════════════════════════════════════════════
  const cctvSrc       = document.getElementById('cctv-src');
  const cctvConf      = document.getElementById('cctv-conf');
  const cctvStart     = document.getElementById('cctv-start');
  const cctvStop      = document.getElementById('cctv-stop');
  const cctvResetHm   = document.getElementById('cctv-reset-hm');
  const cctvLive      = document.getElementById('cctv-live');
  const cctvHeat      = document.getElementById('cctv-heat');
  const cctvMainFeed  = document.getElementById('cctv-main-feed');
  const cctvHeatFeed  = document.getElementById('cctv-heat-feed');
  const cctvPlaceholder = document.getElementById('cctv-placeholder');

  const cctvTotal     = document.getElementById('cctv-total');
  const cctvSusp      = document.getElementById('cctv-susp');
  const cctvDensity   = document.getElementById('cctv-density');
  const densityGrid   = document.getElementById('density-grid');
  const suspList      = document.getElementById('susp-list');
  const cctvAlertLog  = document.getElementById('cctv-alert-log');
  const cctvClearLog  = document.getElementById('cctv-clear-log');

  let cctvPolling     = null;
  let loggedAlerts    = new Set();

  // Create 16 density cells
  for (let i = 0; i < 16; i++) {
    const cell = document.createElement('div');
    cell.className = 'dcell d0';
    cell.id = `dc-${i}`;
    cell.textContent = '';
    densityGrid.appendChild(cell);
  }

  // Toggle View Modes
  cctvLive.addEventListener('click', () => {
    cctvLive.classList.add('active');
    cctvHeat.classList.remove('active');
    cctvMainFeed.classList.remove('hidden');
    cctvHeatFeed.classList.add('hidden');
  });
  cctvHeat.addEventListener('click', () => {
    cctvHeat.classList.add('active');
    cctvLive.classList.remove('active');
    cctvHeatFeed.classList.remove('hidden');
    cctvMainFeed.classList.add('hidden');
  });

  // Start CCTV Stream
  cctvStart.addEventListener('click', async () => {
    cctvStart.disabled = true;
    setGlobalStatus('CCTV Initializing...', 'org');

    try {
      const res = await fetch('/api/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          source: cctvSrc.value.trim() || '0',
          conf: parseFloat(cctvConf.value) || 0.35,
          iou: 0.45
        })
      });
      const data = await res.json();
      if (data.error) throw new Error(data.error);

      // Bind streams
      cctvMainFeed.src = `/video_feed?t=${Date.now()}`;
      cctvHeatFeed.src = `/heatmap_feed?t=${Date.now()}`;
      cctvPlaceholder.classList.add('hidden');

      cctvStop.disabled = false;
      setGlobalStatus('CCTV Live Stream', 'green');
      startCctvPolling();
    } catch (e) {
      alert('Failed to start CCTV: ' + e.message);
      setGlobalStatus('CCTV Error', 'red');
      cctvStart.disabled = false;
    }
  });

  // Stop CCTV Stream
  cctvStop.addEventListener('click', async () => {
    cctvStop.disabled = true;
    await fetch('/api/stop', { method: 'POST' });
    stopCctvPolling();

    cctvMainFeed.src = '';
    cctvHeatFeed.src = '';
    cctvPlaceholder.classList.remove('hidden');
    cctvStart.disabled = false;
    setGlobalStatus('CCTV Stopped', '');
    resetCctvStats();
  });

  cctvResetHm.addEventListener('click', () => fetch('/api/reset_heatmap', { method: 'POST' }));
  cctvClearLog.addEventListener('click', () => {
    loggedAlerts.clear();
    cctvAlertLog.innerHTML = '';
  });

  function startCctvPolling() {
    if (cctvPolling) clearInterval(cctvPolling);
    cctvPolling = setInterval(pollCctvStats, 1000);
  }
  function stopCctvPolling() {
    if (cctvPolling) {
      clearInterval(cctvPolling);
      cctvPolling = null;
    }
  }

  async function pollCctvStats() {
    try {
      const res = await fetch('/api/stats');
      const data = await res.json();
      if (data.error) return;

      // Update counters
      cctvTotal.textContent = data.total || 0;
      cctvSusp.textContent = data.suspicious_count || 0;

      // Density grid
      const grid = data.density_grid || Array(4).fill(Array(4).fill(0));
      const flat = grid.flat();
      flat.forEach((cnt, idx) => {
        const cell = document.getElementById(`dc-${idx}`);
        if (cell) {
          cell.textContent = cnt || '';
          cell.className = 'dcell ' + getDensityGridClass(cnt);
        }
      });

      const maxCell = Math.max(...flat);
      cctvDensity.textContent = maxCell === 0 ? 'CLEAR' : maxCell <= 2 ? 'LOW' : maxCell <= 4 ? 'MED' : 'HIGH';
      cctvDensity.className = 'stat-val ' + (maxCell === 0 ? 'green' : maxCell <= 2 ? 'green' : maxCell <= 4 ? 'warn' : 'danger');

      // Suspicious list
      const suspicious = data.suspicious || {};
      const entries = Object.entries(suspicious);
      if (entries.length === 0) {
        suspList.innerHTML = '<p class="empty-msg">All clear</p>';
      } else {
        suspList.innerHTML = entries.map(([id, reason]) => `
          <div class="susp-item">
            <span class="susp-id">ID #${id}</span>
            <span class="susp-reason">${reason}</span>
          </div>
        `).join('');
      }

      // Alerts
      const alerts = data.alerts || [];
      alerts.forEach(alert => {
        const uniqueKey = alert.time + '-' + alert.msg;
        if (!loggedAlerts.has(uniqueKey)) {
          loggedAlerts.add(uniqueKey);
          const item = document.createElement('div');
          item.className = `alert-item ${alert.type || ''}`;
          item.innerHTML = `<span class="alert-time">${alert.time}</span><span>${alert.msg}</span>`;
          cctvAlertLog.prepend(item);
        }
      });
      while (cctvAlertLog.children.length > 30) {
        cctvAlertLog.removeChild(cctvAlertLog.lastChild);
      }

    } catch (e) {
      console.error(e);
    }
  }

  function getDensityGridClass(n) {
    if (n === 0) return 'd0';
    if (n === 1) return 'd1';
    if (n === 2) return 'd2';
    if (n <= 4)  return 'd3';
    if (n <= 7)  return 'd4';
    return 'd5';
  }

  function resetCctvStats() {
    cctvTotal.textContent = '0';
    cctvSusp.textContent = '0';
    cctvDensity.textContent = 'CLEAR';
    cctvDensity.className = 'stat-val';
    for (let i = 0; i < 16; i++) {
      const cell = document.getElementById(`dc-${i}`);
      if (cell) {
        cell.className = 'dcell d0';
        cell.textContent = '';
      }
    }
    suspList.innerHTML = '<p class="empty-msg">All clear</p>';
  }


  // ════════════════════════════════════════════════════════════════════════
  // TAB 3 — VIDEO FUNCTIONALITY
  // ════════════════════════════════════════════════════════════════════════
  const vidDropZone   = document.getElementById('vid-drop-zone');
  const vidFileInput  = document.getElementById('vid-file-input');
  const vidBrowse     = document.getElementById('vid-browse');
  const vidFilename   = document.getElementById('vid-filename');
  const vidProcess    = document.getElementById('vid-process');
  const vidDownload   = document.getElementById('vid-download');
  const progressBar   = document.getElementById('progress-bar');
  const vidStatusTxt  = document.getElementById('vid-status-txt');
  const vidPct        = document.getElementById('vid-pct');
  const vidStateLabel = document.getElementById('vid-state-label');
  const vidProgressVal= document.getElementById('vid-progress-val');

  let selectedVideoFile = null;
  let videoPollInterval = null;

  vidBrowse.addEventListener('click', () => vidFileInput.click());
  vidDropZone.addEventListener('click', (e) => {
    if (e.target !== vidBrowse) vidFileInput.click();
  });

  // Drag & drop handlers for video
  ['dragenter', 'dragover'].forEach(eventName => {
    vidDropZone.addEventListener(eventName, (e) => {
      e.preventDefault();
      vidDropZone.classList.add('dragover');
    }, false);
  });
  ['dragleave', 'drop'].forEach(eventName => {
    vidDropZone.addEventListener(eventName, (e) => {
      e.preventDefault();
      vidDropZone.classList.remove('dragover');
    }, false);
  });

  vidDropZone.addEventListener('drop', (e) => {
    const dt = e.dataTransfer;
    const files = dt.files;
    if (files.length > 0) {
      handleVideoSelect(files[0]);
    }
  });

  vidFileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
      handleVideoSelect(e.target.files[0]);
    }
  });

  function handleVideoSelect(file) {
    if (!file.type.startsWith('video/')) {
      alert('Please upload a video file.');
      return;
    }
    selectedVideoFile = file;
    vidFilename.textContent = `Selected: ${file.name} (${(file.size / (1024 * 1024)).toFixed(1)} MB)`;
    vidFilename.classList.remove('hidden');
    vidProcess.disabled = false;
    vidDownload.disabled = true;

    // Reset progress UI
    progressBar.style.width = '0%';
    vidPct.textContent = '0%';
    vidProgressVal.textContent = '0%';
    vidStatusTxt.textContent = 'Ready to process';
    vidStateLabel.textContent = 'idle';
  }

  vidProcess.addEventListener('click', async () => {
    if (!selectedVideoFile) return;

    vidProcess.disabled = true;
    vidDownload.disabled = true;
    setGlobalStatus('Uploading & Processing Video...', 'org');

    const formData = new FormData();
    formData.append('video', selectedVideoFile);
    formData.append('conf', document.getElementById('vid-conf').value);

    try {
      const response = await fetch('/process_video', {
        method: 'POST',
        body: formData
      });
      const data = await response.json();
      if (data.error) throw new Error(data.error);

      // Start polling status
      startVideoProgressPolling();
    } catch (e) {
      alert('Error starting video processing: ' + e.message);
      setGlobalStatus('Video Error', 'red');
      vidProcess.disabled = false;
    }
  });

  function startVideoProgressPolling() {
    if (videoPollInterval) clearInterval(videoPollInterval);
    videoPollInterval = setInterval(pollVideoProgress, 1000);
  }

  async function pollVideoProgress() {
    try {
      const res = await fetch('/video_progress');
      const data = await res.json();

      const status = data.status || 'idle';
      const progress = data.progress || 0;

      vidStateLabel.textContent = status.toUpperCase();
      vidProgressVal.textContent = `${progress}%`;
      progressBar.style.width = `${progress}%`;
      vidPct.textContent = `${progress}%`;

      if (status === 'processing') {
        vidStatusTxt.textContent = 'Processing frame-by-frame...';
        setGlobalStatus(`Processing Video (${progress}%)`, 'org');
      } else if (status === 'done') {
        clearInterval(videoPollInterval);
        vidStatusTxt.textContent = 'Processing completed!';
        setGlobalStatus('Video Processing Finished', 'green');
        vidDownload.disabled = false;
        vidProcess.disabled = false;
      } else if (status === 'error') {
        clearInterval(videoPollInterval);
        vidStatusTxt.textContent = 'Error: ' + (data.error || 'Unknown error');
        setGlobalStatus('Video Error', 'red');
        vidProcess.disabled = false;
      }
    } catch (e) {
      console.error('Progress polling error: ', e);
    }
  }

  // Video download click
  vidDownload.addEventListener('click', () => {
    window.location.href = '/download_video';
  });

});
