/**
 * Forward / backward footwork overlay on #resultsVideo.
 * Data: ARM_ATTEMPTS_CACHE.forward_back (from arm-attempt bout scan).
 */
(function (global) {
  "use strict";

  var _enabled = false;
  var _raf = 0;
  var _byTime = null;
  var _stillThr = 0.18;

  function cache() {
    var a = global.ARM_ATTEMPTS_CACHE;
    return a && a.forward_back && Array.isArray(a.forward_back.frames)
      ? a.forward_back
      : null;
  }

  function buildIndex(fb) {
    _byTime = fb.frames.slice().sort(function (a, b) {
      return (a.t || 0) - (b.t || 0);
    });
    _stillThr = Number(fb.still_thr) || 0.18;
  }

  function sampleAt(t) {
    if (!_byTime || !_byTime.length) return null;
    var lo = 0;
    var hi = _byTime.length - 1;
    if (t <= (_byTime[0].t || 0)) return _byTime[0];
    if (t >= (_byTime[hi].t || 0)) return _byTime[hi];
    while (lo < hi) {
      var mid = (lo + hi) >> 1;
      if ((_byTime[mid].t || 0) < t) lo = mid + 1;
      else hi = mid;
    }
    var cur = _byTime[lo];
    var prev = _byTime[Math.max(0, lo - 1)];
    if (
      Math.abs((prev.t || 0) - t) <= Math.abs((cur.t || 0) - t)
    ) {
      return prev;
    }
    return cur;
  }

  function labelColor(lab) {
    if (lab === "forward") return "#3ce03c";
    if (lab === "backward") return "#3c3cf0";
    if (lab === "still") return "#b4b4b4";
    return "#787878";
  }

  function drawSpeedBar(ctx, x, y, signed, stillThr, width) {
    width = width || 110;
    var h = 10;
    var half = width / 2;
    ctx.fillStyle = "rgba(20,20,20,0.85)";
    ctx.fillRect(x, y, width, h);
    ctx.strokeStyle = "#ccc";
    ctx.beginPath();
    ctx.moveTo(x + half, y);
    ctx.lineTo(x + half, y + h);
    ctx.stroke();
    var span = Math.max(stillThr * 3, 0.5);
    var frac = Math.max(-1, Math.min(1, signed / span));
    if (Math.abs(frac) < 1e-3) return;
    var x2 = x + half + frac * half;
    ctx.fillStyle = frac > 0 ? "#3ce03c" : "#3c3cf0";
    ctx.fillRect(Math.min(x + half, x2), y + 2, Math.abs(x2 - (x + half)), h - 4);
  }

  function resizeCanvas(video, canvas) {
    var w = video.clientWidth || video.videoWidth || 640;
    var h = video.clientHeight || video.videoHeight || 360;
    if (canvas.width !== w || canvas.height !== h) {
      canvas.width = w;
      canvas.height = h;
    }
  }

  function draw() {
    var video = document.getElementById("resultsVideo");
    var canvas = document.getElementById("forwardBackCanvas");
    if (!video || !canvas) return;
    var ctx = canvas.getContext("2d");
    resizeCanvas(video, canvas);
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (!_enabled) return;
    var fb = cache();
    if (!fb) return;
    if (!_byTime) buildIndex(fb);
    var row = sampleAt(video.currentTime || 0);
    if (!row || !row.p) return;

    var colors = ["#60a5fa", "#f87171"];
    for (var p = 0; p < row.p.length; p++) {
      var person = row.p[p] || {};
      var lab = person.l || "unknown";
      var speed = Number(person.s) || 0;
      var hx = person.hx;
      var hy = person.hy;
      if (hx == null || hy == null) {
        hx = p === 0 ? 0.25 : 0.75;
        hy = 0.35;
      }
      var x = hx * canvas.width;
      var y = hy * canvas.height;
      var lc = labelColor(lab);

      // Facing (white)
      if (person.fx != null && Math.abs(person.fx) > 0.01) {
        var fl = 48;
        ctx.strokeStyle = "#f0f0f0";
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(x, y);
        ctx.lineTo(x + person.fx * fl, y);
        ctx.stroke();
        // arrow head
        var tipX = x + person.fx * fl;
        ctx.beginPath();
        ctx.moveTo(tipX, y);
        ctx.lineTo(tipX - person.fx * 10, y - 6);
        ctx.lineTo(tipX - person.fx * 10, y + 6);
        ctx.closePath();
        ctx.fillStyle = "#f0f0f0";
        ctx.fill();
      }

      // Motion (colored) when moving
      if (person.vx != null && Math.abs(speed) > _stillThr * 0.25) {
        var ml = 36 + Math.min(70, Math.abs(speed) * 36);
        ctx.strokeStyle = lc;
        ctx.lineWidth = 3;
        ctx.beginPath();
        ctx.moveTo(x, y);
        ctx.lineTo(x + person.vx * ml, y);
        ctx.stroke();
      }

      var tag = "F" + (p + 1) + " " + String(lab).toUpperCase();
      if (lab === "forward" || lab === "backward" || lab === "still") {
        tag += "  " + (speed >= 0 ? "+" : "") + speed.toFixed(2);
      }
      ctx.font = "600 13px Inter, system-ui, sans-serif";
      var tw = ctx.measureText(tag).width;
      var tx = Math.max(6, x - tw / 2);
      var ty = Math.max(18, y - 28);
      ctx.fillStyle = "rgba(0,0,0,0.75)";
      ctx.fillRect(tx - 4, ty - 14, tw + 8, 20);
      ctx.fillStyle = lc;
      ctx.fillText(tag, tx, ty);
      drawSpeedBar(ctx, tx, ty + 8, speed, _stillThr, Math.max(tw, 110));

      // small person color dot
      ctx.beginPath();
      ctx.arc(x, y, 4, 0, Math.PI * 2);
      ctx.fillStyle = colors[p % colors.length];
      ctx.fill();
    }
  }

  function tick() {
    draw();
    _raf = requestAnimationFrame(tick);
  }

  function setEnabled(on) {
    _enabled = !!on;
    var canvas = document.getElementById("forwardBackCanvas");
    if (canvas) {
      canvas.style.display = _enabled ? "block" : "none";
    }
    var btn = document.getElementById("showForwardBack");
    if (btn && btn.checked !== _enabled) btn.checked = _enabled;
    draw();
  }

  function refreshFromCache() {
    _byTime = null;
    var fb = cache();
    var wrap = document.getElementById("forwardBackToggleWrap");
    if (wrap) {
      wrap.classList.toggle("hidden", !fb || global.CLIPPER_MODE);
    }
    if (fb) buildIndex(fb);
    draw();
  }

  function initForwardBackOverlay() {
    var video = document.getElementById("resultsVideo");
    var canvas = document.getElementById("forwardBackCanvas");
    var btn = document.getElementById("showForwardBack");
    if (!video || !canvas) return;

    if (btn && !btn.dataset.bound) {
      btn.dataset.bound = "1";
      btn.addEventListener("change", function () {
        setEnabled(btn.checked);
      });
    }
    video.addEventListener("play", function () {
      if (!_raf) _raf = requestAnimationFrame(tick);
    });
    video.addEventListener("seeked", draw);
    video.addEventListener("timeupdate", draw);
    if (!_raf) _raf = requestAnimationFrame(tick);
    refreshFromCache();
    setEnabled(false);
  }

  global.initForwardBackOverlay = initForwardBackOverlay;
  global.refreshForwardBackOverlay = refreshFromCache;
  global.setForwardBackOverlayEnabled = setEnabled;

  // ---- Pre-touch spatial aggressor (same metric as app/forward_back.py) ----
  // Who advanced *more* before the light — comparative footwork, not attack.
  // Window [light-5s, light-0.5s): last 0.5s cut so the touch action does not dominate.
  var PRE_TOUCH_WINDOW_SEC = 5.0;
  var PRE_TOUCH_END_CUT_SEC = 0.5;
  var EVEN_MARGIN = 0.25;
  var STILL_FLOOR = 0.10;
  var MIN_SAMPLES = 3;

  function touchFrameFromName(touchName) {
    if (!touchName) return null;
    var m = String(touchName).match(/frame(\d+)/i);
    return m ? parseInt(m[1], 10) : null;
  }

  function personWindowStats(rows, personIdx, fps) {
    fps = fps || 30;
    var defaultDt = 1.0 / fps;
    var labels = { forward: 0, backward: 0, still: 0, unknown: 0 };
    var samples = [];
    for (var i = 0; i < rows.length; i++) {
      var people = rows[i].p || [];
      if (personIdx >= people.length) {
        labels.unknown++;
        continue;
      }
      var person = people[personIdx] || {};
      var lab = person.l || "unknown";
      if (!labels.hasOwnProperty(lab)) lab = "unknown";
      labels[lab]++;
      if (lab === "unknown") continue;
      var s = Number(person.s);
      if (isNaN(s)) continue;
      var t = Number(rows[i].t);
      if (isNaN(t)) {
        var ii = Number(rows[i].i);
        t = !isNaN(ii) ? ii / fps : samples.length * defaultDt;
      }
      samples.push({ t: t, s: s });
    }
    var known = labels.forward + labels.backward + labels.still;
    var n = samples.length;
    var meanSigned = 0;
    var netDisp = 0;
    var pathLen = 0;
    if (n) {
      var sum = 0;
      for (var j = 0; j < n; j++) sum += samples[j].s;
      meanSigned = sum / n;
    }
    if (n === 1) {
      netDisp = samples[0].s * defaultDt;
      pathLen = Math.abs(netDisp);
    } else if (n > 1) {
      for (var k = 1; k < n; k++) {
        var dt = samples[k].t - samples[k - 1].t;
        if (dt <= 1e-6 || dt > 1.0) dt = defaultDt;
        var step = 0.5 * (samples[k - 1].s + samples[k].s) * dt;
        netDisp += step;
        pathLen += Math.abs(step);
      }
    }
    return {
      samples: known,
      forward_n: labels.forward,
      backward_n: labels.backward,
      still_n: labels.still,
      unknown_n: labels.unknown,
      forward_frac: known ? Math.round((labels.forward / known) * 10000) / 10000 : 0,
      mean_signed: Math.round(meanSigned * 10000) / 10000,
      net_disp: Math.round(netDisp * 10000) / 10000,
      path_len: Math.round(pathLen * 10000) / 10000,
      advance_score: known ? Math.round((labels.forward / known) * 10000) / 10000 : 0,
      retreat_score: known ? Math.round((labels.backward / known) * 10000) / 10000 : 0,
    };
  }

  function pickSpatialAggressor(f1, f2) {
    if (f1.samples < MIN_SAMPLES && f2.samples < MIN_SAMPLES) return "unclear";
    var m1 = f1.forward_frac != null ? f1.forward_frac : f1.advance_score;
    var m2 = f2.forward_frac != null ? f2.forward_frac : f2.advance_score;
    if (m1 < STILL_FLOOR && m2 < STILL_FLOOR) return "even";
    var delta = m1 - m2;
    var scale = Math.max(m1, m2, STILL_FLOOR);
    if (Math.abs(delta) / scale <= EVEN_MARGIN) return "even";
    return delta > 0 ? "fencer1" : "fencer2";
  }

  function scorePreTouchAggressor(fb, lightFrame) {
    if (!fb || !Array.isArray(fb.frames) || lightFrame == null) return null;
    var fps = Number(fb.fps) || 30;
    var windowSec = PRE_TOUCH_WINDOW_SEC;
    var endCutSec = PRE_TOUCH_END_CUT_SEC;
    var i0 = Math.max(0, lightFrame - Math.round(windowSec * fps));
    var i1 = Math.max(i0, lightFrame - Math.round(endCutSec * fps));
    var rows = [];
    var hasIndex = false;
    for (var i = 0; i < fb.frames.length; i++) {
      if (fb.frames[i].i != null) {
        hasIndex = true;
        break;
      }
    }
    if (hasIndex) {
      for (var j = 0; j < fb.frames.length; j++) {
        var row = fb.frames[j];
        if (row.i == null) continue;
        var ii = Number(row.i);
        if (ii >= i0 && ii < i1) rows.push(row);
      }
    } else {
      var tLight = lightFrame / fps;
      var t0 = tLight - windowSec;
      var t1 = tLight - endCutSec;
      for (var k = 0; k < fb.frames.length; k++) {
        var t = Number(fb.frames[k].t) || 0;
        if (t >= t0 && t < t1) rows.push(fb.frames[k]);
      }
    }
    var f1 = personWindowStats(rows, 0, fps);
    var f2 = personWindowStats(rows, 1, fps);
    return {
      aggressor: pickSpatialAggressor(f1, f2),
      window_sec: windowSec,
      end_cut_sec: endCutSec,
      light_frame: lightFrame,
      sample_count: rows.length,
      fencer1: f1,
      fencer2: f2,
    };
  }

  function getPreTouchAggressorForTouch(touchName) {
    var a = global.ARM_ATTEMPTS_CACHE;
    if (a && a.pre_touch_aggressor && a.pre_touch_aggressor.by_touch) {
      var hit = a.pre_touch_aggressor.by_touch[touchName];
      if (hit) return hit;
    }
    var fb = cache();
    var frame = touchFrameFromName(touchName);
    if (!fb || frame == null) return null;
    return scorePreTouchAggressor(fb, frame);
  }

  function getPreTouchAggressorSummary(touchNames) {
    var a = global.ARM_ATTEMPTS_CACHE;
    if (a && a.pre_touch_aggressor && a.pre_touch_aggressor.touches_scored > 0) {
      return a.pre_touch_aggressor;
    }
    var fb = cache();
    if (!fb) return null;
    var names = touchNames || [];
    var counts = { fencer1: 0, fencer2: 0, even: 0, unclear: 0 };
    var used = 0;
    for (var i = 0; i < names.length; i++) {
      var scored = getPreTouchAggressorForTouch(names[i]);
      if (!scored) continue;
      used++;
      if (counts.hasOwnProperty(scored.aggressor)) counts[scored.aggressor]++;
    }
    if (!used) return null;
    var main = "even";
    if (counts.fencer1 > counts.fencer2) main = "fencer1";
    else if (counts.fencer2 > counts.fencer1) main = "fencer2";
    return {
      fencer1_pre_touch_aggression: counts.fencer1,
      fencer2_pre_touch_aggression: counts.fencer2,
      even: counts.even,
      unclear: counts.unclear,
      both_advancing: 0,
      neither_advancing: counts.even,
      touches_scored: used,
      main_footwork_aggressor: main,
    };
  }

  function formatFootworkAggressorLabel(key) {
    if (key === "fencer1") return "Fencer 1 advanced more";
    if (key === "fencer2") return "Fencer 2 advanced more";
    if (key === "even") return "Even (similar advance)";
    if (key === "both") return "Even (similar advance)";
    if (key === "neither") return "Even (little advance)";
    return "Unclear";
  }

  global.getPreTouchAggressorForTouch = getPreTouchAggressorForTouch;
  global.getPreTouchAggressorSummary = getPreTouchAggressorSummary;
  global.formatFootworkAggressorLabel = formatFootworkAggressorLabel;
})(window);
