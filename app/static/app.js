// Tenge Scan: phone front end. Canvas HUD over the live camera; detection, classification and explanations come from /api/*.
// URL options: ?demo=1 uses a sample photo instead of the camera (testing on a computer).
const W = 390; let H = 844;
const G = "#00ff41", GD = "rgba(0,255,65,.28)", TXT = "#b9ffc9", DIM = "#3f9a55", WARN = "#ffb454";
const CLS = ["5", "10", "20", "50", "100"];
const CCOL = ["#6ee7ff", "#ffd166", G, "#ff8fab", "#c4a1ff"];
const SAMPLES = ["zz_single20", "20_tenge_3A", "50_tenge_2A", "5_tenge_4R", "100_tenge_5A", "10_tenge_3A"];
const params = new URLSearchParams(location.search);

const cv = document.getElementById("c"), ctx = cv.getContext("2d"), video = document.getElementById("v");
let S = 1, DPR = Math.min(window.devicePixelRatio || 1, 3);
function resize() {
  const wide = innerWidth / innerHeight > 0.62;
  if (wide) { H = 844; S = Math.min(innerWidth / W, innerHeight / H) * 0.96; cv.className = "frame"; }
  else { H = W * innerHeight / innerWidth; S = innerWidth / W; cv.className = ""; }
  cv.style.width = W * S + "px"; cv.style.height = H * S + "px";
  cv.width = Math.round(W * S * DPR); cv.height = Math.round(H * S * DPR);
}
addEventListener("resize", resize); resize();

// ---------- helpers
const ease = t => t < 0 ? 0 : t > 1 ? 1 : t * t * (3 - 2 * t);
const lerp = (a, b, t) => a + (b - a) * t;
const now = () => performance.now();
function rr(x, y, w, h, r) { r = Math.max(0, Math.min(r, w / 2, h / 2)); ctx.beginPath(); ctx.moveTo(x + r, y); ctx.arcTo(x + w, y, x + w, y + h, r); ctx.arcTo(x + w, y + h, x, y + h, r); ctx.arcTo(x, y + h, x, y, r); ctx.arcTo(x, y, x + w, y, r); ctx.closePath(); }
function cut(x, y, w, h, c) { ctx.beginPath(); ctx.moveTo(x + c, y); ctx.lineTo(x + w, y); ctx.lineTo(x + w, y + h - c); ctx.lineTo(x + w - c, y + h); ctx.lineTo(x, y + h); ctx.lineTo(x, y + c); ctx.closePath(); }
function txt(s, x, y, size, color = TXT, align = "left", weight = "600", alpha = 1) {
  ctx.save(); ctx.globalAlpha *= alpha; ctx.font = `${weight} ${size}px ui-monospace, "SF Mono", Menlo, monospace`;
  if (!ctx.shadowBlur) { ctx.shadowColor = "rgba(0,0,0,.75)"; ctx.shadowBlur = 3; }
  ctx.fillStyle = color; ctx.textAlign = align; ctx.textBaseline = "alphabetic"; ctx.fillText(s, x, y); ctx.restore();
}
function glowText(s, x, y, size, color, align) { ctx.save(); ctx.shadowColor = color; ctx.shadowBlur = 14; txt(s, x, y, size, color, align, "700"); ctx.restore(); }
function brackets(cx, cy, w, h, len, color = G, lw = 2.5, alpha = 1) {
  const x = cx - w / 2, y = cy - h / 2;
  ctx.save(); ctx.globalAlpha *= alpha; ctx.strokeStyle = color; ctx.lineWidth = lw; ctx.lineCap = "round"; ctx.shadowColor = color; ctx.shadowBlur = 10;
  ctx.beginPath();
  ctx.moveTo(x, y + len); ctx.lineTo(x, y); ctx.lineTo(x + len, y);
  ctx.moveTo(x + w - len, y); ctx.lineTo(x + w, y); ctx.lineTo(x + w, y + len);
  ctx.moveTo(x + w, y + h - len); ctx.lineTo(x + w, y + h); ctx.lineTo(x + w - len, y + h);
  ctx.moveTo(x + len, y + h); ctx.lineTo(x, y + h); ctx.lineTo(x, y + h - len);
  ctx.stroke(); ctx.restore();
}
function scanLine(t, period, top = 0, bottom = H, alpha = 1) {
  const p = (t % period) / period, y = lerp(top, bottom, p < .5 ? p * 2 : 2 - p * 2);
  const g = ctx.createLinearGradient(0, y - 70, 0, y); g.addColorStop(0, "rgba(0,255,65,0)"); g.addColorStop(1, "rgba(0,255,65,.28)");
  ctx.save(); ctx.globalAlpha = alpha; ctx.fillStyle = g; ctx.fillRect(0, y - 70, W, 70);
  ctx.strokeStyle = G; ctx.lineWidth = 1.6; ctx.shadowColor = G; ctx.shadowBlur = 12; ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(W, y); ctx.stroke(); ctx.restore();
}
function screenFrame(a = 1) {
  brackets(W / 2, H / 2, W - 28, H - 56, 26, G, 2, a);
  ctx.save(); ctx.globalAlpha = .12 * a; ctx.strokeStyle = G; ctx.lineWidth = 1;
  for (let i = 1; i < 3; i++) { ctx.beginPath(); ctx.moveTo(W * i / 3, 30); ctx.lineTo(W * i / 3, H - 30); ctx.moveTo(14, H * i / 3); ctx.lineTo(W - 14, H * i / 3); ctx.stroke(); }
  ctx.restore();
}
function topHud(label, blink) {
  txt("TENGE·SCAN", 28, 58, 13, G, "left", "700"); txt("DINOv2 ViT-S/14", 28, 74, 10, DIM, "left", "500");
  if (label) { ctx.save(); if (blink && Math.floor(now() / 500) % 2) ctx.globalAlpha = .25; ctx.fillStyle = "#ff4d5e"; ctx.beginPath(); ctx.arc(W - 118, 54, 4, 0, 7); ctx.fill(); ctx.restore(); txt(label, W - 108, 58, 11, TXT, "left", "600"); }
}

// ---------- sound (synthesised, quiet; the audio context starts on the first tap)
let AC = null;
function audio() { try { AC = AC || new (window.AudioContext || window.webkitAudioContext)(); if (AC.state === "suspended") AC.resume(); } catch (e) {} return AC; }
function tone(f0, f1, dur, vol, type = "sine", delay = 0) {
  const a = audio(); if (!a) return; const t = a.currentTime + delay, o = a.createOscillator(), g = a.createGain();
  o.type = type; o.frequency.setValueAtTime(f0, t); o.frequency.exponentialRampToValueAtTime(f1, t + dur);
  g.gain.setValueAtTime(0.0001, t); g.gain.exponentialRampToValueAtTime(vol, t + .012); g.gain.exponentialRampToValueAtTime(0.0001, t + dur);
  o.connect(g).connect(a.destination); o.start(t); o.stop(t + dur + .02);
}
function noise(dur, vol, hp) {
  const a = audio(); if (!a) return; const n = Math.floor(a.sampleRate * dur), buf = a.createBuffer(1, n, a.sampleRate), d = buf.getChannelData(0);
  for (let i = 0; i < n; i++) d[i] = (Math.random() * 2 - 1) * (1 - i / n);
  const s = a.createBufferSource(), f = a.createBiquadFilter(), g = a.createGain(); f.type = "highpass"; f.frequency.value = hp; g.gain.value = vol;
  s.buffer = buf; s.connect(f).connect(g).connect(a.destination); s.start();
}
const sndDetect = () => { tone(740, 1180, .09, .05, "sine"); tone(1180, 1180, .12, .03, "sine", .09); };
const sndCrop = () => { noise(.07, .05, 2500); tone(190, 70, .16, .08, "triangle"); };

// ---------- image source: camera or demo photo
let SRC = null, camError = "", demoIdx = 0;
const srcSize = () => SRC ? (SRC.tagName === "VIDEO" ? [SRC.videoWidth, SRC.videoHeight] : [SRC.naturalWidth, SRC.naturalHeight]) : [0, 0];
const srcReady = () => { const [w, h] = srcSize(); return w > 0 && h > 0 && (SRC.tagName !== "VIDEO" || SRC.readyState >= 2); };
function useDemo(msg) {
  camError = msg || ""; const im = new Image(); im.src = `/static/samples/${SAMPLES[demoIdx++ % SAMPLES.length]}.jpg`; SRC = im;
}
async function startCamera() {
  if (params.get("demo")) { demoIdx = +params.get("demo") - 1 || 0; useDemo(""); return; }
  try {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) throw new Error("no camera API (needs https)");
    const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: { ideal: "environment" }, width: { ideal: 1920 }, height: { ideal: 1080 } }, audio: false });
    video.srcObject = stream; await video.play(); SRC = video;
    try { navigator.wakeLock && navigator.wakeLock.request("screen"); } catch (e) {}
  } catch (e) { useDemo("camera unavailable: demo photo"); }
}
function viewCover() { const [w, h] = srcSize(); return Math.max(W / w, H / h); }
function drawSource(scale, cx, cy, srcX, srcY) {      // draws SRC so that source point (srcX, srcY) lands at screen (cx, cy)
  const [w, h] = srcSize(); ctx.drawImage(SRC, cx - srcX * scale, cy - srcY * scale, w * scale, h * scale);
}

// ---------- server calls
const dcv = document.createElement("canvas"), dctx = dcv.getContext("2d");
let detectBusy = false, lastDetect = 0, serverInfo = null;
fetch("/api/info").then(r => r.json()).then(j => serverInfo = j).catch(() => {});
function grab(maxSide, quality, canvas = dcv, cx2 = dctx) {
  const [w, h] = srcSize(), k = Math.min(1, maxSide / Math.max(w, h));
  canvas.width = Math.round(w * k); canvas.height = Math.round(h * k); cx2.drawImage(SRC, 0, 0, canvas.width, canvas.height);
  return new Promise(r => canvas.toBlob(r, "image/jpeg", quality));
}
async function detectTick() {
  if (detectBusy || !srcReady() || now() - lastDetect < 110) return;
  detectBusy = true; lastDetect = now();
  try { const blob = await grab(480, .6); const d = await (await fetch("/api/detect", { method: "POST", body: blob })).json(); onDetect(d); } catch (e) {}
  detectBusy = false;
}
let track = null, lastSound = 0;
const FOCUS_MIN = +(params.get("sharpmin") || 80);      // absolute floor of the focus measure (Laplacian variance of the coin region)
function settled(tr) {                                   // focus has converged: sharp, near its peak, and no longer changing
  const l = tr.hist.slice(-4); if (l.length < 4) return false;
  const mx = Math.max(...l), mn = Math.min(...l), mean = l.reduce((a, b) => a + b, 0) / 4;
  return mn >= FOCUS_MIN && mn >= 0.8 * tr.peak && (mx - mn) / mean < 0.25;
}
function onDetect(d) {
  const t = now();
  if (state !== "search" && state !== "lock") return;
  if (!d.found) { if (track && !track.manual && t - track.last > 700) track = null; return; }
  if (track && track.manual) return;
  if (!track) { track = { cx: d.cx, cy: d.cy, r: d.r, born: t, stableSince: t, n: 1, last: t, hist: [d.sharp], peak: d.sharp }; if (t - lastSound > 1500) { lastSound = t; sndDetect(); } return; }
  const dist = Math.hypot(d.cx - track.cx, d.cy - track.cy), dr = Math.abs(d.r - track.r) / track.r;
  if (dist > 0.05 || dr > 0.2) { track.stableSince = t; track.n = 0; }
  const a = .5; track.cx += (d.cx - track.cx) * a; track.cy += (d.cy - track.cy) * a; track.r += (d.r - track.r) * a; track.n++; track.last = t;
  track.hist.push(d.sharp); if (track.hist.length > 8) track.hist.shift(); track.peak = Math.max(d.sharp, track.peak * 0.97);
}
function trackScreen(tr, scale) {          // normalised source coordinates -> screen (live view, cover fit)
  const [w, h] = srcSize(); return { x: (tr.cx * w - w / 2) * scale + W / 2, y: (tr.cy * h - h / 2) * scale + H / 2, r: tr.r * Math.min(w, h) * scale };
}

// ---------- state machine
let state = "home", t0 = now(), resultT0 = now(), sheetP = 0, scrollY = 0, scrollMax = 0, dt0 = now();
let cap = null, res = null, resErr = "", imgs = {}, frozen = null;
function setState(s) {
  state = s; t0 = now();
  if (s === "search") { track = null; res = null; resErr = ""; cap = null; imgs = {}; sheetP = 0; }
  if (s === "zoom") sndCrop();
  if (s === "result") resultT0 = t0;
  if (s === "details") { scrollY = 0; }
}
function capture() {
  const scale0 = viewCover(), tr = { ...track }, p = trackScreen(tr, scale0), [w, h] = srcSize();
  frozen = document.createElement("canvas"); frozen.width = w; frozen.height = h; frozen.getContext("2d").drawImage(SRC, 0, 0, w, h);
  cap = { tr, scale0, sx0: p.x, sy0: p.y, sr0: p.r, w, h };
  res = null; resErr = "";
  const k = Math.min(1, 1600 / Math.max(w, h)), c2 = document.createElement("canvas"); c2.width = Math.round(w * k); c2.height = Math.round(h * k); c2.getContext("2d").drawImage(frozen, 0, 0, c2.width, c2.height);
  c2.toBlob(async blob => {
    const t = now();
    try {
      const r = await fetch(`/api/classify?cx=${tr.cx}&cy=${tr.cy}&r=${tr.r}`, { method: "POST", body: blob });
      if (!r.ok) throw new Error(r.status); const d = await r.json(); d.rtt = now() - t; res = d;
      imgs.crop = new Image(); imgs.crop.src = d.crop;
      imgs.nb = d.neighbours.map(n => { const im = new Image(); im.src = "/files/" + n.path; return im; });
      const a = document.createElement("canvas"); a.width = a.height = d.attention.length; const ac = a.getContext("2d"), id = ac.createImageData(a.width, a.height);
      d.attention.forEach((row, y) => row.forEach((v, x) => { const i = (y * a.width + x) * 4, u = Math.max(0, (v - .35) / .65); id.data[i] = 255; id.data[i + 1] = Math.round(lerp(235, 40, u)); id.data[i + 2] = 20; id.data[i + 3] = Math.round(255 * Math.pow(u, 1.2) * .85); }));
      ac.putImageData(id, 0, 0); imgs.att = a;
    } catch (e) { resErr = "server error"; }
  }, "image/jpeg", .9);
}
const LAYOUT = () => { const by = H - 124, avail = by - 20 - 260; return { by, rc: Math.min(160, (avail - 30) / 2), cy: (260 + by - 20) / 2 }; };

// ---------- screens
const hit = {};
function drawHome(t) {
  ctx.fillStyle = "#020804"; ctx.fillRect(0, 0, W, H);
  ctx.save(); ctx.strokeStyle = "rgba(0,255,65,.16)"; ctx.lineWidth = 1;
  const hy = H * .51; for (let i = -12; i <= 12; i++) { ctx.beginPath(); ctx.moveTo(W / 2 + i * 12, hy); ctx.lineTo(W / 2 + i * 70, H); ctx.stroke(); }
  for (let k = 0; k < 14; k++) { const p = ((k + (now() / 1500) % 1) / 14), y = hy + (H - hy) * p * p; ctx.globalAlpha = .6 * p + .1; ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(W, y); ctx.stroke(); }
  ctx.restore();
  const cy = H * .32, g = ctx.createRadialGradient(W / 2, cy, 10, W / 2, cy, 260); g.addColorStop(0, "rgba(0,255,65,.16)"); g.addColorStop(1, "rgba(0,255,65,0)"); ctx.fillStyle = g; ctx.fillRect(0, 0, W, H * .66);
  const cx = W / 2, pulse = (now() / 1800) % 1;
  ctx.save(); ctx.strokeStyle = G; ctx.shadowColor = G; ctx.shadowBlur = 16;
  ctx.globalAlpha = 1 - pulse; ctx.lineWidth = 1.5; ctx.beginPath(); ctx.arc(cx, cy, 70 + pulse * 60, 0, 7); ctx.stroke();
  ctx.globalAlpha = 1; ctx.lineWidth = 3; ctx.beginPath(); ctx.arc(cx, cy, 62, 0, 7); ctx.stroke();
  ctx.lineWidth = 1; ctx.setLineDash([3, 7]); ctx.beginPath(); ctx.arc(cx, cy, 78, now() / 2000, now() / 2000 + 5.2); ctx.stroke(); ctx.restore();
  glowText("₸", cx, cy + 22, 64, G, "center");
  glowText("TENGE·SCAN", W / 2, cy + 170, 28, G, "center");
  txt("Kazakhstan coin recognition", W / 2, cy + 196, 12, "#4fbf68", "center", "500"); txt("5 · 10 · 20 · 50 · 100 tenge", W / 2, cy + 216, 12, DIM, "center", "500");
  const bx = W / 2 - 120, by = H - 194, bw = 240, bh = 64, pr = .5 + .5 * Math.sin(now() / 400);
  ctx.save(); ctx.shadowColor = G; ctx.shadowBlur = 18 + pr * 14; cut(bx, by, bw, bh, 16); ctx.fillStyle = "rgba(0,255,65,.12)"; ctx.fill(); ctx.strokeStyle = G; ctx.lineWidth = 2; ctx.stroke(); ctx.restore();
  glowText("START SCAN", W / 2, by + 40, 20, G, "center");
  txt("DINOv2", W / 2, H - 54, 12, "#2f7a42", "center", "500");
  screenFrame(.7); hit.start = [bx, by, bw, bh];
}

function drawLive(t, sub) {
  ctx.fillStyle = "#000"; ctx.fillRect(0, 0, W, H);
  if (srcReady()) { const s = viewCover(), [w, h] = srcSize(); drawSource(s, W / 2, H / 2, w / 2, h / 2); }
  const tg = ctx.createLinearGradient(0, 0, 0, 120); tg.addColorStop(0, "rgba(0,8,2,.65)"); tg.addColorStop(1, "rgba(0,8,2,0)"); ctx.fillStyle = tg; ctx.fillRect(0, 0, W, 120);
  const bg = ctx.createLinearGradient(0, H - 140, 0, H); bg.addColorStop(0, "rgba(0,8,2,0)"); bg.addColorStop(1, "rgba(0,8,2,.65)"); ctx.fillStyle = bg; ctx.fillRect(0, H - 140, W, 140);
  screenFrame(); topHud(sub === "lock" ? "LOCKED" : "SEARCHING", sub !== "lock");
  ctx.save(); ctx.strokeStyle = GD; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(W / 2 - 14, H / 2); ctx.lineTo(W / 2 + 14, H / 2); ctx.moveTo(W / 2, H / 2 - 14); ctx.lineTo(W / 2, H / 2 + 14); ctx.stroke(); ctx.restore();
  const tr = track && srcReady() ? trackScreen(track, viewCover()) : null;
  if (sub === "search") {
    scanLine(t, 2.2);
    if (tr) {
      const k = ease((now() - track.born) / 700), n = now() / 1000;
      ctx.save(); ctx.strokeStyle = G; ctx.globalAlpha = Math.min(1, k * 1.5); ctx.lineWidth = 2; ctx.shadowColor = G; ctx.shadowBlur = 10;
      ctx.setLineDash([6, 8]); ctx.beginPath(); ctx.arc(tr.x, tr.y, tr.r + 4, n * 1.6, n * 1.6 + 6.283); ctx.stroke(); ctx.restore();
      const side = lerp(330, 2 * tr.r + 30, k); brackets(tr.x, tr.y, side, side, 22, G, 2.5, Math.min(1, k * 2));
      const ok = track.manual || settled(track);
      txt(ok ? `COIN?  r=${Math.round(tr.r)}px` : "FOCUSING…", tr.x, tr.y - side / 2 - 12, 11, ok ? G : WARN, "center", "600", Math.min(1, k * 2));
      if (!ok) txt("waiting for a sharp image", W / 2, H - 80, 11, WARN, "center", "500");
      if (track.n >= 4 && now() - track.stableSince > 800 && now() - track.born > 900 && ok && !track.manual) setState("lock");
    } else {
      txt("point the camera at a coin", W / 2, H - 92, 12, "#4fbf68", "center", "500");
      if (t > 4) txt("or tap on the coin", W / 2, H - 72, 11, DIM, "center", "500");
      if (camError) txt(camError, W / 2, H - 112, 11, WARN, "center", "600");
    }
  } else if (tr) {
    const side = 2 * tr.r + 30, p = ease(t / 1.0);
    brackets(tr.x, tr.y, side, side, 22, "#fff", 3, 1);
    ctx.save(); ctx.strokeStyle = G; ctx.lineWidth = 4; ctx.shadowColor = G; ctx.shadowBlur = 14; ctx.beginPath(); ctx.arc(tr.x, tr.y, tr.r + 4, -1.5708, -1.5708 + 6.283 * p); ctx.stroke(); ctx.restore();
    txt(`LOCKING  ${Math.round(p * 100)}%`, tr.x, tr.y - side / 2 - 12, 12, "#fff", "center", "700"); txt("hold still", W / 2, H - 80, 12, TXT, "center", "500");
  }
}

function frozenView(p) {           // p: 0 = as captured, 1 = result layout. Returns screen centre/radius/scale of the coin.
  const L = LAYOUT(), s1 = cap.scale0 * L.rc / cap.sr0, e = ease(p);
  return { x: lerp(cap.sx0, W / 2, e), y: lerp(cap.sy0, L.cy, e), scale: lerp(cap.scale0, s1, e), L };
}
function drawFrozen(v) {
  ctx.fillStyle = "#010402"; ctx.fillRect(0, 0, W, H);
  ctx.drawImage(frozen, v.x - cap.tr.cx * cap.w * v.scale, v.y - cap.tr.cy * cap.h * v.scale, cap.w * v.scale, cap.h * v.scale);
}

function drawZoom(t) {
  const p = Math.min(1, t / .9), v = frozenView(p), r = cap.tr.r * Math.min(cap.w, cap.h) * v.scale;
  drawFrozen(v);
  brackets(v.x, v.y, 2 * r + 30, 2 * r + 30, 22, G, 2.5, 1);
  txt("CROP 224 × 224", v.x, v.y - r - 27, 11, G, "center", "700", 1 - ease(p * 1.5));
  ctx.save(); ctx.globalAlpha = Math.max(0, .85 - t * 5); ctx.fillStyle = "#fff"; ctx.fillRect(0, 0, W, H); ctx.restore();
  screenFrame(1 - p); topHud("CAPTURED", false);
}

function chipRow(y, items) {
  ctx.font = "600 11px ui-monospace, Menlo, monospace";
  const ws = items.map(s => ctx.measureText(s).width + 20), gap = 8, total = ws.reduce((u, v) => u + v, 0) + gap * (items.length - 1);
  let x = (W - total) / 2;
  items.forEach((s, i) => { ctx.save(); ctx.strokeStyle = GD; ctx.lineWidth = 1; rr(x, y, ws[i], 24, 12); ctx.stroke(); ctx.restore(); txt(s, x + ws[i] / 2, y + 16, 11, TXT, "center", "600"); x += ws[i] + gap; });
}

function drawResult(t) {
  const L = LAYOUT(), v = frozenView(1), rc = L.rc, cx = W / 2, cy = L.cy, side = 2 * rc + 30;
  drawFrozen(v);
  const top = ctx.createLinearGradient(0, 0, 0, 260); top.addColorStop(0, "rgba(0,8,2,.92)"); top.addColorStop(1, "rgba(0,8,2,0)"); ctx.fillStyle = top; ctx.fillRect(0, 0, W, 260);
  const bot = ctx.createLinearGradient(0, H - 230, 0, H); bot.addColorStop(0, "rgba(0,8,2,0)"); bot.addColorStop(1, "rgba(0,8,2,.95)"); ctx.fillStyle = bot; ctx.fillRect(0, H - 230, W, 230);
  brackets(cx, cy, side, side, 30, G, 3);
  // leader line from the box corner to the card
  const sx = cx + side / 2, sy = cy - side / 2, ax = W - 64, ay = 226, pts = [[sx, sy], [sx, sy - 26], [ax, ay]];
  const lp = ease((t - .15) / .6), lens = [26, Math.hypot(ax - sx, ay - sy + 26)]; let rem = lp * (lens[0] + lens[1]);
  ctx.save(); ctx.strokeStyle = G; ctx.lineWidth = 1.8; ctx.shadowColor = G; ctx.shadowBlur = 8; ctx.beginPath(); ctx.moveTo(sx, sy);
  for (let i = 0; i < 2; i++) { const f = Math.max(0, Math.min(1, rem / lens[i])); rem -= lens[i]; if (f > 0) ctx.lineTo(lerp(pts[i][0], pts[i + 1][0], f), lerp(pts[i][1], pts[i + 1][1], f)); }
  ctx.stroke(); ctx.fillStyle = G; ctx.beginPath(); ctx.arc(sx, sy, 4, 0, 7); ctx.fill(); ctx.restore();
  // info card
  const ca = ease((t - .5) / .5), cxx = 24, cyy = 86 - (1 - ca) * 14, cw = W - 48, ch = 140, unsure = res && res.uncertain, col = unsure ? WARN : G;
  ctx.save(); ctx.globalAlpha = ca;
  cut(cxx, cyy, cw, ch, 18); ctx.fillStyle = "rgba(2,14,6,.86)"; ctx.fill(); ctx.shadowColor = col; ctx.shadowBlur = 14; ctx.strokeStyle = col; ctx.lineWidth = 1.8; ctx.stroke(); ctx.shadowBlur = 0;
  txt("DENOMINATION", cxx + 18, cyy + 28, 10, DIM, "left", "600");
  if (resErr) { glowText("ERROR", cxx + 18, cyy + 74, 36, "#ff4d5e", "left"); txt("tap ✕ block (top right) to retry", cxx + 18, cyy + 104, 11, TXT, "left", "500"); }
  else if (!res) { glowText("ANALYSING" + ".".repeat(1 + Math.floor(now() / 300) % 3), cxx + 18, cyy + 70, 30, G, "left"); }
  else {
    glowText(unsure ? "NOT SURE" : `${res.label} TENGE`, cxx + 18, cyy + 74, 40, col, "left");
    txt(unsure ? `most likely ${CLS[res.pred]} tenge` : "CONFIDENCE", cxx + 18, cyy + 104, 10, unsure ? WARN : DIM, "left", "600");
    const cf = res.conf * ease((t - .7) / .7);
    ctx.fillStyle = "rgba(0,255,65,.15)"; rr(cxx + 18, cyy + 112, cw - 120, 8, 4); ctx.fill(); ctx.fillStyle = col; rr(cxx + 18, cyy + 112, (cw - 120) * cf, 8, 4); ctx.fill();
    txt((cf * 100).toFixed(1) + "%", cxx + cw - 18, cyy + 120, 22, col, "right", "700");
    txt(`2nd: ${CLS[res.second]} tenge  ${(res.second_conf * 100).toFixed(1)}%`, cxx + cw - 18, cyy + 28, 10, WARN, "right", "600");
  }
  ctx.restore();
  txt("TENGE·SCAN", 28, 52, 13, G, "left", "700");
  ctx.save(); ctx.strokeStyle = GD; ctx.lineWidth = 1.5; rr(W - 62, 36, 38, 26, 8); ctx.stroke(); ctx.restore(); hit.close = [W - 62, 30, 44, 38];
  const by = L.by; ctx.save(); const ba = .5 + .5 * Math.sin(now() / 350);
  cut(24, by, W - 48, 66, 28); ctx.fillStyle = "rgba(0,255,65,.1)"; ctx.fill(); ctx.strokeStyle = G; ctx.lineWidth = 1.8; ctx.shadowColor = G; ctx.shadowBlur = 8 + ba * 8; ctx.stroke(); ctx.restore();
  glowText("MORE INFO", W / 2, by + 42, 22, G, "center"); hit.more = [24, by, W - 48, 66];
  chipRow(by + 82, res ? [`${res.timing.model_ms.toFixed(0)} ms`, res.device, res.size && res.size.ok ? `Ø ${res.size.mm.toFixed(1)} mm` : "224×224"] : ["…"]);
}

// ---------- details sheet
function section(y, title) { txt(title, 24, y, 10, DIM, "left", "700"); ctx.strokeStyle = "rgba(0,255,65,.18)"; ctx.beginPath(); ctx.moveTo(24, y + 8); ctx.lineTo(W - 24, y + 8); ctx.stroke(); }
const okImg = im => im && im.complete && im.naturalWidth > 0;
function thumb(im, x, y, s, zoom = .62) {
  if (!okImg(im)) { ctx.fillStyle = "#06150a"; ctx.fillRect(x, y, s, s); return; }
  const N = im.naturalWidth, n = N * zoom, o = (N - n) / 2; ctx.drawImage(im, o, o, n, n, x, y, s, s);
}
function browserName() {
  const u = navigator.userAgent, os = /iPhone|iPad/.test(u) ? "iOS" : /Android/.test(u) ? "Android" : /Mac/.test(u) ? "macOS" : "desktop";
  const b = /CriOS|Chrome/.test(u) ? "Chrome" : /FxiOS|Firefox/.test(u) ? "Firefox" : /Safari/.test(u) ? "Safari" : "browser"; return `${b} · ${os}`;
}
function drawSheet(p, nowMs) {
  const top = lerp(H, 0, p);
  ctx.save(); ctx.globalAlpha = .62 * p; ctx.fillStyle = "#000"; ctx.fillRect(0, 0, W, H); ctx.restore();
  ctx.save(); rr(0, top, W, H - top + 30, 26 * (1 - p)); ctx.fillStyle = "#020a04"; ctx.fill(); ctx.restore();
  hit.sheetTop = top;
  ctx.save(); rr(0, top, W, H - top, 26 * (1 - p)); ctx.clip();
  txt("ANALYSIS", 28, top + 52, 13, G, "left", "700");
  ctx.save(); ctx.strokeStyle = GD; ctx.lineWidth = 1.5; rr(W - 62, top + 36, 38, 26, 8); ctx.stroke(); ctx.restore(); hit.sheetClose = [W - 62, top + 30, 44, 38];
  ctx.save(); ctx.beginPath(); ctx.rect(0, top + 80, W, H - top - 80); ctx.clip(); ctx.translate(0, top + 80 - scrollY);
  const a = ease((nowMs - dt0) / 700); let y = 14;
  if (res) {
    const unsure = res.uncertain;
    ctx.save(); ctx.beginPath(); ctx.arc(52, y + 36, 32, 0, 7); ctx.clip(); thumb(imgs.crop, 20, y + 4, 64, .5); ctx.restore();
    ctx.save(); ctx.strokeStyle = unsure ? WARN : G; ctx.lineWidth = 2; ctx.beginPath(); ctx.arc(52, y + 36, 32, 0, 7); ctx.stroke(); ctx.restore();
    glowText(unsure ? "NOT SURE" : `${res.label} TENGE`, 100, y + 34, 24, unsure ? WARN : G, "left");
    txt(unsure ? `most likely ${CLS[res.pred]} tenge · ${(res.conf * 100).toFixed(1)}%` : `${(res.conf * 100).toFixed(1)}% confident`, 100, y + 56, 11, "#4fbf68", "left", "500");
    y += 92;
    section(y, "CLASS PROBABILITIES"); y += 28;
    res.probs.map((p, i) => [p, i]).sort((u, v) => v[0] - u[0]).forEach(([pr, i]) => {
      txt(CLS[i], 24, y + 12, 13, TXT, "left", "700"); ctx.fillStyle = "rgba(255,255,255,.07)"; rr(64, y + 2, W - 150, 14, 7); ctx.fill();
      ctx.fillStyle = CCOL[i]; rr(64, y + 2, Math.max(6, (W - 150) * pr * a), 14, 7); ctx.fill();
      txt((pr * 100 * a).toFixed(1) + "%", W - 24, y + 14, 12, TXT, "right", "600"); y += 28;
    });
    const pm = res.probs_model.indexOf(Math.max(...res.probs_model));
    if (res.size && res.size.ok) txt(`model alone: ${CLS[pm]} tenge ${(res.probs_model[pm] * 100).toFixed(0)}%  ·  with size: ${CLS[res.pred]} tenge ${(res.conf * 100).toFixed(0)}%`, 24, y + 8, 10, pm === res.pred ? DIM : WARN, "left", "500");
    else txt("5 ↔ 10 and 20 ↔ 50 are the confusions found in the paper", 24, y + 8, 10, WARN, "left", "500");
    y += 36;
    section(y, "SIZE FROM SQUARED PAPER"); y += 22;
    if (res.size && res.size.ok) {
      const s = res.size, ex = (serverInfo && serverInfo.expected_cells) || { "5": 3.59, "10": 3.96, "20": 3.80, "50": 4.70, "100": 5.25 }, lo = 3.3, hi = 5.5, ax = v => 36 + (W - 72) * (v - lo) / (hi - lo);
      ctx.strokeStyle = GD; ctx.lineWidth = 1.5; ctx.beginPath(); ctx.moveTo(36, y + 22); ctx.lineTo(W - 36, y + 22); ctx.stroke();
      CLS.forEach((c, i) => { const x = ax(ex[c]); ctx.fillStyle = CCOL[i]; ctx.beginPath(); ctx.arc(x, y + 22, 4.5, 0, 7); ctx.fill(); txt(c, x, y + (i === 2 ? 44 : 10), 10, CCOL[i], "center", "700"); });
      const mx = ax(Math.min(hi, Math.max(lo, s.cells))); ctx.fillStyle = "#fff"; ctx.beginPath(); ctx.moveTo(mx, y + 28); ctx.lineTo(mx - 6, y + 40); ctx.lineTo(mx + 6, y + 40); ctx.closePath(); ctx.fill();
      txt("cells (1 cell = 5 mm)", W - 36, y + 56, 9, DIM, "right", "500"); y += 70;
      [["coin diameter", `${s.cells.toFixed(2)} cells ≈ ${s.mm.toFixed(1)} mm`], ["cell on the photo", `${s.cell_px.toFixed(0)} px`], ["distance to the sheet", `≈ ${s.distance_cm.toFixed(0)} cm (estimate)`]]
        .forEach(([k, v]) => { txt(k, 24, y, 11, DIM, "left", "600"); txt(v, W - 24, y, 11, TXT, "right", "600"); y += 22; });
      y += 14;
    } else { txt("no squared paper found: size is not used", 24, y + 14, 11, DIM, "left", "500"); y += 44; }
    section(y, "NEAREST COINS IN THE TRAINING SET"); y += 22;
    const tw = (W - 48 - 4 * 8) / 5;
    res.neighbours.forEach((n, i) => {
      const x = 24 + i * (tw + 8), same = n.cls === res.pred; thumb(imgs.nb[i], x, y, tw); ctx.strokeStyle = same ? G : WARN; ctx.lineWidth = 1.6; ctx.strokeRect(x, y, tw, tw);
      txt(CLS[n.cls], x + tw / 2, y + tw + 15, 11, same ? G : WARN, "center", "700"); txt(n.sim.toFixed(2), x + tw / 2, y + tw + 29, 9, DIM, "center", "500");
    });
    y += tw + 46; txt(`kNN vote (k=${res.knn.k}): ${CLS[res.knn.pred]} tenge, ${res.knn.count} of ${res.knn.k}`, 24, y, 11, TXT, "left", "600"); y += 28;
    section(y, "WHERE THE MODEL LOOKS"); y += 18;
    const pw = (W - 48 - 12) / 2, hx = 24 + pw + 12;
    thumb(imgs.crop, 24, y, pw, 1); thumb(imgs.crop, hx, y, pw, 1);
    if (imgs.att) { ctx.save(); ctx.beginPath(); ctx.rect(hx, y, pw, pw); ctx.clip(); ctx.imageSmoothingEnabled = true; ctx.imageSmoothingQuality = "high"; ctx.drawImage(imgs.att, hx, y, pw, pw); ctx.restore(); }
    ctx.strokeStyle = GD; ctx.strokeRect(24, y, pw, pw); ctx.strokeRect(hx, y, pw, pw);
    y += pw + 16; txt("crop", 24 + pw / 2, y, 10, DIM, "center", "600"); txt("attention (last layer, CLS token)", hx + pw / 2, y, 9, DIM, "center", "600"); y += 30;
    section(y, "EMBEDDING MAP  (LDA, 2D)"); y += 18;
    const mw = W - 48, mh = 190;
    ctx.fillStyle = "rgba(0,255,65,.04)"; ctx.fillRect(24, y, mw, mh); ctx.strokeStyle = GD; ctx.strokeRect(24, y, mw, mh);
    res.map.points.forEach(pt => { ctx.fillStyle = CCOL[pt.c]; ctx.globalAlpha = .75; ctx.beginPath(); ctx.arc(24 + mw * pt.x, y + mh * pt.y, 2.6, 0, 7); ctx.fill(); ctx.globalAlpha = 1; });
    const qx = 24 + mw * Math.min(.96, Math.max(.04, res.map.query.x)), qy = y + mh * Math.min(.94, Math.max(.06, res.map.query.y)), pu = (nowMs / 1200) % 1;
    ctx.save(); ctx.strokeStyle = "#fff"; ctx.shadowColor = "#fff"; ctx.shadowBlur = 10; ctx.lineWidth = 1.4; ctx.globalAlpha = 1 - pu; ctx.beginPath(); ctx.arc(qx, qy, 6 + pu * 16, 0, 7); ctx.stroke(); ctx.restore();
    ctx.fillStyle = "#fff"; ctx.beginPath(); ctx.arc(qx, qy, 4.5, 0, 7); ctx.fill(); txt("your coin", qx + (qx > W - 110 ? -10 : 10), qy - 10, 10, "#fff", qx > W - 110 ? "right" : "left", "700");
    CLS.forEach((c, i) => { ctx.fillStyle = CCOL[i]; ctx.beginPath(); ctx.arc(34 + i * 54, y + mh + 18, 4, 0, 7); ctx.fill(); txt(c, 43 + i * 54, y + mh + 22, 10, TXT, "left", "600"); });
    y += mh + 48;
    section(y, "SYSTEM"); y += 28;
    const rows = [["model inference", `${res.timing.model_ms.toFixed(0)} ms`], ["server total", `${res.timing.total_ms.toFixed(0)} ms`], ["round trip (phone ↔ server)", `${res.rtt.toFixed(0)} ms`],
      ["server device", res.device], ["browser", browserName()], ["model", "DINOv2 ViT-S/14 · 22 M params"], ["input", "coin crop 224 × 224"], ["reference images", serverInfo ? String(serverInfo.n_index) : "—"]];
    if (serverInfo && serverInfo.mode === "dev") rows.push(["weights", "development (fold 1)"]);
    rows.forEach(([k, v]) => { txt(k, 24, y, 11, DIM, "left", "600"); txt(v, W - 24, y, 11, TXT, "right", "600"); y += 22; });
    y += 24;
  }
  scrollMax = Math.max(0, y - (H - top - 80));
  ctx.restore(); ctx.restore();
}

// ---------- loop and input
let last = now();
function frame(nowMs) {
  try { frameBody(nowMs); } catch (e) { console.error(e); if (!cap && (state === "zoom" || state === "result" || state === "details")) setState("search"); }
  requestAnimationFrame(frame);
}
function frameBody(nowMs) {
  const dt = Math.min(.05, (nowMs - last) / 1000); last = nowMs;
  ctx.setTransform(DPR * S, 0, 0, DPR * S, 0, 0); ctx.clearRect(0, 0, W, H);
  Object.keys(hit).forEach(k => delete hit[k]);
  const t = (nowMs - t0) / 1000;
  if (state === "search" || state === "lock") detectTick();
  if (state === "lock" && (!track || (!track.manual && (nowMs - track.last > 600 || track.hist[track.hist.length - 1] < 0.6 * track.peak || track.hist[track.hist.length - 1] < FOCUS_MIN * .7)))) setState("search");
  else if (state === "lock" && t > 1.0) { capture(); setState("zoom"); }
  else if (state === "zoom" && t > .95) setState("result");
  const tt = (nowMs - t0) / 1000;
  if (state === "home") drawHome(tt);
  else if (state === "search") drawLive(tt, "search");
  else if (state === "lock") drawLive(tt, "lock");
  else if (state === "zoom") drawZoom(tt);
  else drawResult((nowMs - resultT0) / 1000);
  const target = state === "details" ? 1 : 0;
  sheetP += (target - sheetP) * Math.min(1, dt * 11); if (Math.abs(target - sheetP) < .002) sheetP = target;
  if (state === "details" && sheetP < .05) dt0 = nowMs;
  if (sheetP > 0.001) { drawSheet(ease(sheetP), nowMs); hit.more = null; hit.close = null; }
}
requestAnimationFrame(frame);

const inRect = (x, y, r) => r && x >= r[0] && x <= r[0] + r[2] && y >= r[1] && y <= r[1] + r[3];
let drag = null;
function pos(e) { const b = cv.getBoundingClientRect(); return [(e.clientX - b.left) / S, (e.clientY - b.top) / S]; }
cv.addEventListener("pointerdown", e => { const [x, y] = pos(e); drag = { x, y, sy: scrollY, moved: false }; cv.setPointerCapture(e.pointerId); });
cv.addEventListener("pointermove", e => { if (!drag) return; const [x, y] = pos(e); if (Math.abs(y - drag.y) > 6) drag.moved = true; if (state === "details" && drag.moved) scrollY = Math.max(0, Math.min(scrollMax, drag.sy - (y - drag.y))); });
cv.addEventListener("pointerup", e => {
  const [x, y] = pos(e), d = drag; drag = null; if (!d) return;
  if (d.moved && state === "details") { if (scrollY === 0 && y - d.y > 90) setState("result"); return; }
  if (state === "home" && inRect(x, y, hit.start)) { audio(); startCamera(); setState("search"); }
  else if (state === "search" && srcReady() && y > 90 && y < H - 90) {       // manual selection: tap on the coin
    const s = viewCover(), [w, h] = srcSize();
    track = { cx: ((x - W / 2) / s + w / 2) / w, cy: ((y - H / 2) / s + h / 2) / h, r: 0.12, born: now(), stableSince: now(), n: 9, last: now(), manual: true, hist: [], peak: 0 };
    audio(); sndDetect(); setTimeout(() => { if (state === "search" && track && track.manual) setState("lock"); }, 500);
  }
  else if (state === "result" && sheetP < .05) { if (inRect(x, y, hit.more)) setState("details"); else if (inRect(x, y, hit.close)) { if (SRC && SRC.tagName === "IMG") useDemo(camError); setState("search"); } }
  else if (state === "details") { if (inRect(x, y, hit.sheetClose) || y < hit.sheetTop - 4) setState("result"); }
});
cv.addEventListener("wheel", e => { if (state === "details") { e.preventDefault(); scrollY = Math.max(0, Math.min(scrollMax, scrollY + e.deltaY)); } }, { passive: false });
