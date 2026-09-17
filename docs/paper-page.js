/* ============================================================
   Paper project page — interactions
   - Fixed background scrubbed by scroll (frame 0 at top of page,
     last frame ~halfway through the first viewport).
   - Uses a real <video> if assets/header.mp4 loads; otherwise
     renders a deterministic procedural placeholder on canvas.
   ============================================================ */
(function () {
  "use strict";

  // Fraction of the viewport height over which the background plays
  // through once. 0.5 => fully played by the time you scroll half a fold.
  // Exposed so the Tweaks panel can change it live.
  window.__scrubFraction = window.__scrubFraction || 0.5;

  var clamp = function (v, a, b) { return Math.min(b, Math.max(a, v)); };
  var lerp = function (a, b, t) { return a + (b - a) * t; };

  var stage = document.getElementById("bgStage");
  var video = document.getElementById("bgVideo");
  var canvas = document.getElementById("bgCanvas");
  var ctx = canvas.getContext("2d");

  /* ---------- scroll progress ---------- */
  function scrollProgress() {
    var vh = window.innerHeight || 800;
    var span = Math.max(1, vh * window.__scrubFraction);
    return clamp((window.scrollY || window.pageYOffset || 0) / span, 0, 1);
  }

  /* ---------- video path ---------- */
  var videoReady = false;
  function onVideoReady() {
    if (videoReady) return;
    if (!video.duration || isNaN(video.duration) || video.duration === Infinity) return;
    videoReady = true;
    stage.classList.add("has-video");
    try { video.pause(); } catch (e) {}
  }
  video.addEventListener("loadeddata", onVideoReady);
  video.addEventListener("canplay", onVideoReady);
  video.addEventListener("error", function () { /* keep canvas placeholder */ });
  // If no source resolves quickly, we simply stay on the canvas placeholder.
  try { video.load(); } catch (e) {}

  /* ---------- procedural placeholder (rotating point mesh) ---------- */
  // A clean slate-on-dark wireframe point cloud. Its rotation is a pure
  // function of scroll progress, so scroll position == frame, exactly like
  // the real video path.
  var DPR = Math.min(2, window.devicePixelRatio || 1);
  var W = 0, H = 0;
  var POINTS = [];
  var N = 150;
  (function buildSphere() {
    var golden = Math.PI * (3 - Math.sqrt(5));
    for (var i = 0; i < N; i++) {
      var y = 1 - (i / (N - 1)) * 2;
      var r = Math.sqrt(1 - y * y);
      var th = golden * i;
      POINTS.push([Math.cos(th) * r, y, Math.sin(th) * r]);
    }
  })();

  function resize() {
    W = stage.clientWidth;
    H = stage.clientHeight;
    canvas.width = Math.round(W * DPR);
    canvas.height = Math.round(H * DPR);
    ctx.setTransform(DPR, 0, 0, DPR, 0, 0);
  }

  function renderCanvas(p) {
    if (!W || !H) resize();
    // background wash
    var g = ctx.createLinearGradient(0, 0, W, H);
    g.addColorStop(0, "#0c111c");
    g.addColorStop(0.55, "#0e1524");
    g.addColorStop(1, "#0a0e17");
    ctx.fillStyle = g;
    ctx.fillRect(0, 0, W, H);

    var cx = W * 0.5, cy = H * 0.46;
    var scale = Math.min(W, H) * 0.34;
    var yaw = p * Math.PI * 1.6 + 0.4;
    var pitch = -0.45 + p * 0.5;
    var cy0 = Math.cos(yaw), sy0 = Math.sin(yaw);
    var cp0 = Math.cos(pitch), sp0 = Math.sin(pitch);

    var proj = [];
    for (var i = 0; i < POINTS.length; i++) {
      var x = POINTS[i][0], y = POINTS[i][1], z = POINTS[i][2];
      // yaw (around Y)
      var x1 = x * cy0 + z * sy0;
      var z1 = -x * sy0 + z * cy0;
      // pitch (around X)
      var y1 = y * cp0 - z1 * sp0;
      var z2 = y * sp0 + z1 * cp0;
      var depth = (z2 + 1.6) / 2.6; // ~0..1, far..near
      proj.push({
        sx: cx + x1 * scale,
        sy: cy + y1 * scale,
        d: depth
      });
    }

    // edges between near neighbours
    var thr = scale * 0.42;
    ctx.lineWidth = 1;
    for (var a = 0; a < proj.length; a++) {
      for (var b = a + 1; b < proj.length; b++) {
        var dx = proj[a].sx - proj[b].sx;
        var dy = proj[a].sy - proj[b].sy;
        var dist = Math.sqrt(dx * dx + dy * dy);
        if (dist < thr) {
          var alpha = (1 - dist / thr) * 0.5 * ((proj[a].d + proj[b].d) / 2);
          ctx.strokeStyle = "rgba(150,176,214," + alpha.toFixed(3) + ")";
          ctx.beginPath();
          ctx.moveTo(proj[a].sx, proj[a].sy);
          ctx.lineTo(proj[b].sx, proj[b].sy);
          ctx.stroke();
        }
      }
    }
    // points
    for (var k = 0; k < proj.length; k++) {
      var pt = proj[k];
      var rad = 0.6 + pt.d * 2.0;
      ctx.beginPath();
      ctx.fillStyle = "rgba(205,216,236," + (0.25 + pt.d * 0.65).toFixed(3) + ")";
      ctx.arc(pt.sx, pt.sy, rad, 0, Math.PI * 2);
      ctx.fill();
    }

    // placeholder caption (mono) — disappears once a real video is wired
    var frame = Math.round(p * 100);
    ctx.font = "11px 'IBM Plex Mono', ui-monospace, monospace";
    ctx.fillStyle = "rgba(138,151,169,0.62)";
    ctx.textAlign = "center";
    ctx.fillText(
      "PLACEHOLDER BACKGROUND · FRAME " + String(frame).padStart(3, "0") + " / 100 · drop assets/header.mp4 to replace",
      cx, H - 26
    );
  }

  /* ---------- render loop (smooth) ---------- */
  var shownTime = 0;        // smoothed video time
  var lastP = -1;
  function frame() {
    var p = scrollProgress();

    if (videoReady) {
      var target = p * video.duration;
      shownTime = lerp(shownTime, target, 0.18);
      if (Math.abs(shownTime - target) < 0.004) shownTime = target;
      // Only issue a new seek once the previous one has landed. Re-seeking
      // every frame aborts the pending seek, so nothing paints until the
      // user stops scrolling.
      if (!video.seeking && Math.abs((video.currentTime || 0) - shownTime) > 0.002) {
        try {
          if (typeof video.fastSeek === "function" && Math.abs(shownTime - target) > 0.5) {
            video.fastSeek(shownTime);
          } else {
            video.currentTime = shownTime;
          }
        } catch (e) {}
      }
    } else {
      if (p !== lastP) { renderCanvas(p); lastP = p; }
    }
    requestAnimationFrame(frame);
  }

  /* ---------- hero foot fade ---------- */
  var heroFoot = document.getElementById("heroFoot");
  function updateHeroFoot() {
    var p = scrollProgress();
    if (heroFoot) heroFoot.style.opacity = String(clamp(1 - p * 2.2, 0, 1));
  }

  /* ---------- section rail (show + active) ---------- */
  var rail = document.getElementById("rail");
  var railLinks = Array.prototype.slice.call(rail.querySelectorAll("a"));
  var ids = railLinks.map(function (a) { return a.getAttribute("href").slice(1); });
  var sections = ids.map(function (id) { return document.getElementById(id); });

  function updateRail() {
    var heroH = window.innerHeight * 0.6;
    if (rail) rail.classList.toggle("show", (window.scrollY || 0) > heroH);

    var scrollY = window.scrollY || window.pageYOffset || 0;
    var mid = scrollY + window.innerHeight * 0.35;
    var activeIdx = 0;
    for (var i = 0; i < sections.length; i++) {
      // Use the section's absolute document position. offsetTop is relative to
      // the nearest positioned ancestor (.paper), which excludes the hero —
      // mixing that with the absolute scrollY made the rail jump ahead early.
      if (sections[i]) {
        var top = sections[i].getBoundingClientRect().top + scrollY;
        if (top <= mid) activeIdx = i;
      }
    }
    for (var j = 0; j < railLinks.length; j++) {
      railLinks[j].classList.toggle("active", j === activeIdx);
    }
  }

  /* ---------- scroll wiring ---------- */
  var ticking = false;
  function onScroll() {
    if (!ticking) {
      ticking = true;
      requestAnimationFrame(function () {
        updateHeroFoot();
        updateRail();
        ticking = false;
      });
    }
  }
  window.addEventListener("scroll", onScroll, { passive: true });
  window.addEventListener("resize", function () { resize(); lastP = -1; updateRail(); });

  /* ---------- bibtex copy ---------- */
  var copyBtn = document.getElementById("copyBib");
  if (copyBtn) {
    copyBtn.addEventListener("click", function () {
      var txt = (document.getElementById("bibText").innerText || "").trim();
      var done = function () {
        copyBtn.textContent = "Copied ✓";
        copyBtn.classList.add("done");
        setTimeout(function () { copyBtn.textContent = "Copy"; copyBtn.classList.remove("done"); }, 1600);
      };
      if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(txt).then(done, done);
      } else {
        var ta = document.createElement("textarea");
        ta.value = txt; document.body.appendChild(ta); ta.select();
        try { document.execCommand("copy"); } catch (e) {}
        document.body.removeChild(ta); done();
      }
    });
  }

  /* ---------- year ---------- */
  var yr = document.getElementById("yr");
  if (yr) yr.textContent = new Date().getFullYear();

  /* ---------- go ---------- */
  resize();
  renderCanvas(0);
  updateHeroFoot();
  updateRail();
  requestAnimationFrame(frame);
})();
