const CLOUDFRONT_BASE_URL = "https://d2lhrf3zzm7l1f.cloudfront.net/models";
const BACKEND_UPLOAD_URL = "https://cnpnehrhma.ap-south-1.awsapprunner.com/upload";
const sessionId = crypto.randomUUID(); // Unique session ID for uploads

function getModelUrl(variant) {
  const map = {
    SINGLEPOSE_LIGHTNING: "singlepose-lightning",
    SINGLEPOSE_THUNDER: "singlepose-thunder",
    MULTIPOSE_LIGHTNING: "multipose-lightning"
  };
  return `${CLOUDFRONT_BASE_URL}/${map[variant]}/model.json`;
}

// DOM & State
const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const statsEl = document.getElementById('stats');
const toastEl = document.getElementById('toast');
const rawJsonEl = document.getElementById('rawJson');
const derivedEl = document.getElementById('derived');
const errEl = document.getElementById('err');
const variantSel = document.getElementById('variant');
const useWebcamBtn = document.getElementById('useWebcam');
const imageInput = document.getElementById('imageInput');
const videoInput = document.getElementById('videoInput');
const exportJsonBtn = document.getElementById('exportJson');
const storeBtn = document.getElementById('storeFrame');
const autoStoreChk = document.getElementById('autoStore');
const storeStatusEl = document.getElementById('storeStatus');
const variantBadge = document.getElementById('variantBadge');

let detector = null;
let running = false;
let fpsSamples = [];
let allDetections = [];
let autoTimer = null;
let webcamStream = null;
let lastFrameTs = 0;
let uploadedVideoBase64 = null;

function setWebcamBtn(mode) {
  useWebcamBtn.textContent = mode === 'stop' ? 'Stop Webcam' : 'Start Webcam';
  useWebcamBtn.classList.toggle('btn-stop', mode === 'stop');
  useWebcamBtn.classList.toggle('btn-start', mode !== 'stop');
}

function showError(e) {
  errEl.classList.remove('hidden');
  errEl.textContent = (e && e.stack) ? e.stack : String(e || 'Unknown error');
  console.error(e);
}
window.addEventListener('error', ev => showError(ev.error || ev.message));
window.addEventListener('unhandledrejection', ev => showError(ev.reason));
window.addEventListener('beforeunload', stopWebcam);

function showToast(msg) {
  toastEl.textContent = msg;
  toastEl.hidden = false;
  setTimeout(() => (toastEl.hidden = true), 2200);
}

function variantNiceName(val) {
  if (val === 'SINGLEPOSE_THUNDER') return 'SinglePose Thunder (CloudFront)';
  if (val === 'MULTIPOSE_LIGHTNING') return 'MultiPose Lightning (CloudFront)';
  return 'SinglePose Lightning (CloudFront)';
}

function updateBadges() {
  variantBadge.textContent = `Variant: ${variantNiceName(variantSel.value)}`;
}
variantSel.addEventListener('change', updateBadges);

async function ensureTfReady() {
  await tf.setBackend('webgl').catch(() => tf.setBackend('cpu'));
  await tf.ready();
}

function mapModelType(v) {
  const t = poseDetection.movenet.modelType;
  if (v === 'SINGLEPOSE_THUNDER') return t.SINGLEPOSE_THUNDER;
  if (v === 'MULTIPOSE_LIGHTNING') return t.MULTIPOSE_LIGHTNING;
  return t.SINGLEPOSE_LIGHTNING;
}

async function createDetector() {
  if (detector) return detector;
  await ensureTfReady();
  const modelType = mapModelType(variantSel.value);
  const modelUrl = getModelUrl(variantSel.value);

  detector = await poseDetection.createDetector(
    poseDetection.SupportedModels.MoveNet,
    {
      modelType,
      modelUrl,
      enableSmoothing: true,
      multiPoseMaxDimension: 256,
      minPoseScore: 0.15
    }
  );
  return detector;
}

async function restartDetector() {
  try { await detector?.dispose?.(); } catch (_) {}
  detector = null;
  await createDetector();
}

function resizeCanvas(w, h) {
  canvas.width = w;
  canvas.height = h;
}

function drawPoses(poses) {
  if (!video.paused && !video.ended && video.readyState >= 2) {
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
  }

  poses.forEach(p => {
    (p.keypoints || []).forEach(k => {
      if (k.score > 0.3) {
        ctx.beginPath();
        ctx.arc(k.x, k.y, 6, 0, Math.PI * 2);
        ctx.fill();
      }
    });
  });

  ctx.save();
  ctx.font = '12px system-ui, sans-serif';
  const label = variantNiceName(variantSel.value).replace(' (CloudFront)', '');
  const text = `Variant: ${label}`;
  const pad = 6;
  const w = ctx.measureText(text).width + pad * 2;
  ctx.fillStyle = 'rgba(0,0,0,0.55)';
  ctx.fillRect(8, 8, w, 22);
  ctx.fillStyle = '#fff';
  ctx.fillText(text, 8 + pad, 8 + 15);
  ctx.restore();
  lastFrameTs = Date.now();
}

function updateStats(start) {
  const dt = performance.now() - start;
  const fps = 1000 / dt;
  fpsSamples.push(fps);
  if (fpsSamples.length > 30) fpsSamples.shift();
  const avgFps = fpsSamples.reduce((a, b) => a + b, 0) / fpsSamples.length;
  statsEl.textContent = `FPS: ${avgFps.toFixed(1)} | Frame latency: ${dt.toFixed(1)} ms`;
  return dt;
}

function pretty(obj) {
  rawJsonEl.textContent = JSON.stringify(obj || {}, null, 2);
}

function renderDerived(obj) {
  if (!obj || !obj.poses) {
    derivedEl.innerHTML = '<p>Waiting for detections…</p>';
    return;
  }
  const kp = obj.poses[0]?.keypoints || [];
  let html = `<p><b>Poses:</b> ${obj.poses.length}</p>`;
  html += `<table class="kptable"><thead><tr><th>Keypoint</th><th>Score</th><th>X</th><th>Y</th></tr></thead><tbody>`;
  kp.forEach((k, i) => {
    html += `<tr><td>${k.name || 'kp_' + i}</td><td>${k.score?.toFixed(2)}</td><td>${k.x?.toFixed(1)}</td><td>${k.y?.toFixed(1)}</td></tr>`;
  });
  html += "</tbody></table>";
  derivedEl.innerHTML = html;
}

async function loop() {
  if (!running) return;
  if (!detector) await createDetector();

  const start = performance.now();
  let poses = [];

  try {
    poses = await detector.estimatePoses(video);
  } catch (e) {
    showError(e);
    running = false;
    return;
  }

  drawPoses(poses);
  const latency = updateStats(start);
  const frame = { ts: Date.now(), poses, variant: variantSel.value, frameLatencyMs: latency };
  allDetections.push(frame);
  pretty(frame);
  renderDerived(frame);

  requestAnimationFrame(loop);
}

function stopWebcam() {
  webcamStream?.getTracks().forEach(t => t.stop());
  webcamStream = null;
  running = false;
  video.srcObject = null;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  statsEl.textContent = '';
  if (autoTimer) clearInterval(autoTimer);
  setWebcamBtn('start');
}

async function startWebcam() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ video: true });
    webcamStream = stream;
    video.srcObject = stream;
    await video.play();
    await new Promise(res => video.readyState >= 2 ? res() : video.onloadedmetadata = res);
    resizeCanvas(video.videoWidth, video.videoHeight);
    await restartDetector();
    running = true;
    setWebcamBtn('stop');
    loop();
  } catch (e) {
    showToast('Webcam failed. Try uploading.');
    showError(e);
    setWebcamBtn('start');
  }
}

async function handleImage(file) {
  if (webcamStream) stopWebcam();
  const img = new Image();
  img.onload = async () => {
    resizeCanvas(img.width, img.height);
    ctx.drawImage(img, 0, 0);
    lastFrameTs = Date.now();
    await restartDetector();
    const start = performance.now();
    const poses = await detector.estimatePoses(img);
    drawPoses(poses);
    const latency = updateStats(start);
    const frame = { ts: Date.now(), poses, variant: variantSel.value, frameLatencyMs: latency };
    allDetections.push(frame);
    pretty(frame);
    renderDerived(frame);
  };
  img.src = URL.createObjectURL(file);
}

async function handleVideo(file) {
  if (webcamStream) stopWebcam();
  uploadedVideoBase64 = null;

  const reader = new FileReader();
  reader.onload = () => {
    uploadedVideoBase64 = reader.result;
  };
  reader.readAsDataURL(file);

  const url = URL.createObjectURL(file);
  video.src = url;
  await video.play();
  await new Promise(res => video.readyState >= 2 ? res() : video.onloadedmetadata = res);
  resizeCanvas(video.videoWidth, video.videoHeight);
  await restartDetector();
  running = true;
  loop();
}

function sendCurrentFrame() {
  const latest = allDetections[allDetections.length - 1];
  if (!latest || !latest.poses || latest.poses.length === 0) {
    showToast("No pose data to send.");
    return;
  }

  const image_base64 = canvas.toDataURL("image/jpeg", 0.9);

  const payload = {
    session_id: sessionId,
    poses: latest.poses,
    image_base64,
    video_base64: uploadedVideoBase64 || null
  };

  fetch(BACKEND_UPLOAD_URL, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload)
  })
    .then(res => res.json())
    .then(data => {
      storeStatusEl.textContent = data.pose_key || "Uploaded";
      showToast("Frame uploaded.");
    })
    .catch(err => {
      console.error(err);
      showToast("Upload failed.");
    });
}

// Events
useWebcamBtn.onclick = () => webcamStream ? stopWebcam() : startWebcam();
imageInput.onchange = e => e.target.files?.[0] && handleImage(e.target.files[0]);
videoInput.onchange = e => e.target.files?.[0] && handleVideo(e.target.files[0]);
exportJsonBtn.onclick = () => {
  const blob = new Blob([JSON.stringify(allDetections, null, 2)], { type: "application/json" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "detections.json";
  a.click();
};
storeBtn.onclick = sendCurrentFrame;
autoStoreChk.onchange = (e) => {
  if (e.target.checked) {
    autoTimer = setInterval(() => {
      if (canvas.width === 0 || canvas.height === 0 || lastFrameTs === 0) return;
      sendCurrentFrame();
    }, 5000);
    showToast("Auto-store ON");
  } else {
    clearInterval(autoTimer);
    autoTimer = null;
    showToast("Auto-store OFF");
  }
};
variantSel.addEventListener('change', async () => {
  updateBadges();
  const wasRunning = running;
  running = false;
  if (autoTimer) clearInterval(autoTimer);
  await restartDetector();
  const active = !!video.srcObject || (!video.paused && !video.ended);
  if (active && wasRunning) {
    running = true;
    requestAnimationFrame(loop);
  }
});
updateBadges();
