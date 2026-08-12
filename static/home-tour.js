/**
 * Homepage product tour — scroll-driven walkthrough of a shared bout analysis.
 */
(function () {
  var DEFAULT_TOKEN = 'OsWPvz6JfQK1jbP9WNomr_veQqioXNgqy59Vp0a4xlw';
  var STEP_COUNT = 5;
  var reduceMotion =
    window.matchMedia &&
    window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  function esc(s) {
    return String(s == null ? '' : s)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;');
  }

  function fencerNum(touch) {
    var t = String(touch || '').toLowerCase();
    if (t.indexOf('fencer1') >= 0) return 1;
    if (t.indexOf('fencer2') >= 0) return 2;
    return 0;
  }

  function attackLabel(a) {
    var s = String(a || '').toLowerCase();
    if (s === 'other') return 'counter';
    return s || '—';
  }

  function countBy(preds, keyFn) {
    var out = {};
    preds.forEach(function (p) {
      var k = keyFn(p);
      if (!k) return;
      out[k] = (out[k] || 0) + 1;
    });
    return out;
  }

  function barRows(counts, order, colorMap) {
    var total = order.reduce(function (s, k) {
      return s + (counts[k] || 0);
    }, 0);
    if (!total) {
      return '<p class="tour-muted">No classified touches in this demo bout.</p>';
    }
    return order
      .map(function (k) {
        var n = counts[k] || 0;
        var pct = Math.round((n / total) * 100);
        var color = (colorMap && colorMap[k]) || 'var(--brand)';
        return (
          '<div class="tour-bar-row">' +
          '<span class="tour-bar-label">' +
          esc(k) +
          '</span>' +
          '<div class="tour-bar-track"><div class="tour-bar-fill" style="width:' +
          pct +
          '%;background:' +
          color +
          '"></div></div>' +
          '<span class="tour-bar-pct">' +
          pct +
          '%</span>' +
          '</div>'
        );
      })
      .join('');
  }

  /* ── Forward / back overlay (tour video) ── */
  var _fbByTime = null;
  var _fbStillThr = 0.18;
  var _fbRaf = 0;
  var _fbEnabled = false;

  function fbLabelColor(lab) {
    if (lab === 'forward') return '#3ce03c';
    if (lab === 'backward') return '#3c3cf0';
    if (lab === 'still') return '#b4b4b4';
    return '#787878';
  }

  function fbBuildIndex(fb) {
    _fbByTime = (fb.frames || []).slice().sort(function (a, b) {
      return (a.t || 0) - (b.t || 0);
    });
    _fbStillThr = Number(fb.still_thr) || 0.18;
  }

  function fbSampleAt(t) {
    if (!_fbByTime || !_fbByTime.length) return null;
    var lo = 0;
    var hi = _fbByTime.length - 1;
    if (t <= (_fbByTime[0].t || 0)) return _fbByTime[0];
    if (t >= (_fbByTime[hi].t || 0)) return _fbByTime[hi];
    while (lo < hi) {
      var mid = (lo + hi) >> 1;
      if ((_fbByTime[mid].t || 0) < t) lo = mid + 1;
      else hi = mid;
    }
    var cur = _fbByTime[lo];
    var prev = _fbByTime[Math.max(0, lo - 1)];
    if (Math.abs((prev.t || 0) - t) <= Math.abs((cur.t || 0) - t)) return prev;
    return cur;
  }

  function fbDrawSpeedBar(ctx, x, y, signed, stillThr, width) {
    width = width || 110;
    var h = 10;
    var half = width / 2;
    ctx.fillStyle = 'rgba(20,20,20,0.85)';
    ctx.fillRect(x, y, width, h);
    ctx.strokeStyle = '#ccc';
    ctx.beginPath();
    ctx.moveTo(x + half, y);
    ctx.lineTo(x + half, y + h);
    ctx.stroke();
    var span = Math.max(stillThr * 3, 0.5);
    var frac = Math.max(-1, Math.min(1, signed / span));
    if (Math.abs(frac) < 1e-3) return;
    var x2 = x + half + frac * half;
    ctx.fillStyle = frac > 0 ? '#3ce03c' : '#3c3cf0';
    ctx.fillRect(Math.min(x + half, x2), y + 2, Math.abs(x2 - (x + half)), h - 4);
  }

  function drawTourForwardBack() {
    var video = document.getElementById('tourVideo');
    var canvas = document.getElementById('tourForwardBackCanvas');
    if (!video || !canvas) return;
    var w = video.clientWidth || video.videoWidth || 640;
    var h = video.clientHeight || video.videoHeight || 360;
    if (canvas.width !== w || canvas.height !== h) {
      canvas.width = w;
      canvas.height = h;
    }
    var ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (!_fbEnabled || !_fbByTime) return;
    var row = fbSampleAt(video.currentTime || 0);
    if (!row || !row.p) return;
    var colors = ['#60a5fa', '#f87171'];
    for (var p = 0; p < row.p.length; p++) {
      var person = row.p[p] || {};
      var lab = person.l || 'unknown';
      var speed = Number(person.s) || 0;
      var hx = person.hx != null ? person.hx : p === 0 ? 0.25 : 0.75;
      var hy = person.hy != null ? person.hy : 0.35;
      var x = hx * canvas.width;
      var y = hy * canvas.height;
      var lc = fbLabelColor(lab);

      if (person.fx != null && Math.abs(person.fx) > 0.01) {
        var fl = 48;
        ctx.strokeStyle = '#f0f0f0';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(x, y);
        ctx.lineTo(x + person.fx * fl, y);
        ctx.stroke();
        var tipX = x + person.fx * fl;
        ctx.beginPath();
        ctx.moveTo(tipX, y);
        ctx.lineTo(tipX - person.fx * 10, y - 6);
        ctx.lineTo(tipX - person.fx * 10, y + 6);
        ctx.closePath();
        ctx.fillStyle = '#f0f0f0';
        ctx.fill();
      }

      if (person.vx != null && Math.abs(speed) > _fbStillThr * 0.25) {
        var ml = 36 + Math.min(70, Math.abs(speed) * 36);
        ctx.strokeStyle = lc;
        ctx.lineWidth = 3;
        ctx.beginPath();
        ctx.moveTo(x, y);
        ctx.lineTo(x + person.vx * ml, y);
        ctx.stroke();
      }

      var tag = 'F' + (p + 1) + ' ' + String(lab).toUpperCase();
      if (lab === 'forward' || lab === 'backward' || lab === 'still') {
        tag += '  ' + (speed >= 0 ? '+' : '') + speed.toFixed(2);
      }
      ctx.font = '600 12px DM Sans, system-ui, sans-serif';
      var tw = ctx.measureText(tag).width;
      var tx = Math.max(6, x - tw / 2);
      var ty = Math.max(18, y - 28);
      ctx.fillStyle = 'rgba(0,0,0,0.75)';
      ctx.fillRect(tx - 4, ty - 14, tw + 8, 20);
      ctx.fillStyle = lc;
      ctx.fillText(tag, tx, ty);
      fbDrawSpeedBar(ctx, tx, ty + 8, speed, _fbStillThr, Math.max(tw, 110));

      ctx.beginPath();
      ctx.arc(x, y, 4, 0, Math.PI * 2);
      ctx.fillStyle = colors[p % colors.length];
      ctx.fill();
    }
  }

  function tourFbTick() {
    drawTourForwardBack();
    if (_fbEnabled) _fbRaf = requestAnimationFrame(tourFbTick);
    else _fbRaf = 0;
  }

  function setTourForwardBack(on) {
    _fbEnabled = !!on;
    var canvas = document.getElementById('tourForwardBackCanvas');
    if (canvas) canvas.style.display = _fbEnabled ? 'block' : 'none';
    if (_fbEnabled && !_fbRaf) _fbRaf = requestAnimationFrame(tourFbTick);
    else if (!_fbEnabled) drawTourForwardBack();
  }

  function initTourForwardBack(fb) {
    _fbByTime = null;
    if (!fb || !Array.isArray(fb.frames) || !fb.frames.length) {
      setTourForwardBack(false);
      return;
    }
    fbBuildIndex(fb);
    var video = document.getElementById('tourVideo');
    if (video && !video.dataset.fbBound) {
      video.dataset.fbBound = '1';
      video.addEventListener('seeked', drawTourForwardBack);
      video.addEventListener('timeupdate', drawTourForwardBack);
      video.addEventListener('play', function () {
        if (_fbEnabled && !_fbRaf) _fbRaf = requestAnimationFrame(tourFbTick);
      });
    }
    setTourForwardBack(true);
  }

  function buildPanels(data, shareUrl) {
    var SHARE_URL = shareUrl || '/result';
    var r = data.results || {};
    var preds = (r.predictions || []).slice();
    var deleted = {};
    (r.deleted_touch_ids || []).forEach(function (id) {
      deleted[id] = true;
    });
    preds = preds.filter(function (p) {
      return p && p.touch && !deleted[p.touch];
    });

    var f1 = preds.filter(function (p) {
      return fencerNum(p.touch) === 1;
    });
    var f2 = preds.filter(function (p) {
      return fencerNum(p.touch) === 2;
    });

    var arm = r.arm_attempts || {};
    var forwardBack =
      arm.forward_back && Array.isArray(arm.forward_back.frames)
        ? arm.forward_back
        : null;
    // Overlay is timed to the full bout video, not the highlight reel.
    var videoUrl = forwardBack
      ? r.video_url || r.highlight_reel_url || ''
      : r.highlight_reel_url || r.video_url || '';
    var llm = r.llm_analysis || null;
    var analysis = (llm && llm.analysis) || {};
    var archName =
      analysis.archetype_name ||
      (analysis.fencer1 && analysis.fencer1.archetype_name) ||
      (analysis.fencer2 && analysis.fencer2.archetype_name) ||
      '';
    var archStyle =
      analysis.archetype_style ||
      (analysis.fencer1 && analysis.fencer1.archetype_style) ||
      (analysis.fencer2 && analysis.fencer2.archetype_style) ||
      '';
    var rationale =
      analysis.rationale ||
      (analysis.fencer1 && analysis.fencer1.rationale) ||
      (analysis.fencer2 && analysis.fencer2.rationale) ||
      '';
    var drills = (analysis.practice_suggestions || []).slice(0, 3);

    var locCounts = countBy(preds, function (p) {
      return p.prediction ? String(p.prediction).toLowerCase() : null;
    });
    var atkCounts = countBy(preds, function (p) {
      return p.attack_prediction
        ? attackLabel(p.attack_prediction)
        : null;
    });

    var touchChips = preds
      .slice(0, 10)
      .map(function (p, i) {
        var n = fencerNum(p.touch);
        var cls = n === 1 ? 'f1' : n === 2 ? 'f2' : '';
        var label = p.touch_ref || 'T' + (i + 1);
        return (
          '<span class="tour-chip ' +
          cls +
          '">' +
          esc(label) +
          (p.prediction ? ' · ' + esc(p.prediction) : '') +
          '</span>'
        );
      })
      .join('');

    var drillHtml = drills.length
      ? drills
          .map(function (d, i) {
            return (
              '<div class="tour-drill">' +
              '<p class="tour-drill-rank">#' +
              (i + 1) +
              '</p>' +
              '<div>' +
              '<p class="tour-drill-title">' +
              esc(d.title || 'Drill') +
              '</p>' +
              (d.why
                ? '<p class="tour-drill-why">' + esc(d.why) + '</p>'
                : '') +
              '</div></div>'
            );
          })
          .join('')
      : '<p class="tour-muted">Coaching drills appear here after style analysis.</p>';

    var videoHtml = videoUrl
      ? '<div class="tour-video-wrap">' +
        '<video id="tourVideo" class="tour-video" muted playsinline loop preload="metadata" src="' +
        esc(videoUrl) +
        '"></video>' +
        (forwardBack
          ? '<canvas id="tourForwardBackCanvas" class="tour-fb-canvas"></canvas>'
          : '') +
        '</div>'
      : '';

    return {
      videoUrl: videoUrl,
      forwardBack: forwardBack,
      panels: [
        {
          title: 'Every touch, marked',
          body:
            'Lights become a scored timeline — who hit, where, and how the attack was shaped.',
          html:
            '<div class="tour-score">' +
            '<div><span class="tour-score-n f1">' +
            f1.length +
            '</span><span class="tour-score-l">Fencer 1</span></div>' +
            '<div class="tour-score-sep">–</div>' +
            '<div><span class="tour-score-n f2">' +
            f2.length +
            '</span><span class="tour-score-l">Fencer 2</span></div>' +
            '</div>' +
            videoHtml +
            '<div class="tour-chip-row">' +
            touchChips +
            (preds.length > 10
              ? '<span class="tour-chip more">+' +
                (preds.length - 10) +
                ' more</span>'
              : '') +
            '</div>',
        },
        {
          title: 'Your fencing style',
          body:
            'Thresholds lock a consistent archetype from this bout — then coaching explains why in plain language.',
          html:
            '<div class="tour-arch">' +
            '<p class="tour-eyebrow">Style</p>' +
            '<h3>' +
            esc(archName || 'Archetype ready') +
            '</h3>' +
            '<p class="tour-arch-style">' +
            esc(
              archStyle ||
                'Upload a bout to reveal whether you fence as a Duelist, Sniper, Bolt, and more.'
            ) +
            '</p>' +
            (rationale
              ? '<div class="tour-arch-obs">' +
                '<p class="tour-eyebrow">fight observations</p>' +
                '<p class="tour-arch-why">' +
                esc(rationale) +
                '</p></div>'
              : '') +
            '</div>',
        },
        {
          title: "Top 3 coach's drills",
          body:
            'Practice notes ranked for this bout — not a generic tip sheet.',
          html: '<div class="tour-drills">' + drillHtml + '</div>',
        },
        {
          title: 'Patterns you can train',
          body:
            'Target zones and attack mix from the same video — so review stops being guesswork.',
          html:
            '<div class="tour-charts">' +
            '<div><p class="tour-chart-h">Touch locations</p>' +
            barRows(
              locCounts,
              ['chest', 'abdomen', 'arm', 'leg'],
              {
                chest: '#f87171',
                abdomen: '#fb923c',
                arm: '#60a5fa',
                leg: '#34d399',
              }
            ) +
            '</div>' +
            '<div><p class="tour-chart-h">Attack types</p>' +
            barRows(
              atkCounts,
              ['lunge', 'fleche', 'counter'],
              {
                lunge: '#a78bfa',
                fleche: '#22d3ee',
                counter: '#fbbf24',
              }
            ) +
            '</div></div>',
        },
        {
          title: 'Open the full analysis',
          body:
            'Explore every touch, 3D pose, and coaching note from this demo bout — or upload your own.',
          html:
            '<div class="tour-cta-panel">' +
            '<p class="tour-cta-lead">This walkthrough uses a real shared analysis.</p>' +
            '<a class="btn-primary tour-open-btn" href="' +
            esc(SHARE_URL) +
            '">Open full analysis</a>' +
            '<a class="btn-secondary tour-analyze-btn" href="/demo">Analyze my fencing</a>' +
            '</div>',
        },
      ],
    };
  }

  function setStep(root, step) {
    root.dataset.step = String(step);
    root.querySelectorAll('[data-tour-step]').forEach(function (el) {
      var i = parseInt(el.getAttribute('data-tour-step'), 10);
      el.classList.toggle('is-active', i === step);
    });
    root.querySelectorAll('.tour-dot').forEach(function (el, i) {
      el.classList.toggle('is-active', i === step);
    });
    var video = document.getElementById('tourVideo') || root.querySelector('.tour-video');
    if (video) {
      if (step === 0) {
        setTourForwardBack(!!_fbByTime);
        var p = video.play();
        if (p && p.catch) p.catch(function () {});
      } else {
        setTourForwardBack(false);
        try {
          video.pause();
        } catch (e) {}
      }
    }
  }

  function progressFor(track) {
    var rect = track.getBoundingClientRect();
    var vh = window.innerHeight || 1;
    // When track top hits viewport top, progress 0; when bottom would leave, progress 1.
    var scrollable = Math.max(1, rect.height - vh);
    var raw = -rect.top / scrollable;
    return Math.max(0, Math.min(1, raw));
  }

  function initTour(root, built) {
    var copyHost = root.querySelector('[data-tour-copy]');
    var stageHost = root.querySelector('[data-tour-stage]');
    var dotsHost = root.querySelector('[data-tour-dots]');
    if (!copyHost || !stageHost) return;

    copyHost.innerHTML = built.panels
      .map(function (p, i) {
        return (
          '<div class="tour-copy-step" data-tour-step="' +
          i +
          '">' +
          '<p class="tour-kicker">Step ' +
          (i + 1) +
          ' / ' +
          STEP_COUNT +
          '</p>' +
          '<h2>' +
          esc(p.title) +
          '</h2>' +
          '<p>' +
          esc(p.body) +
          '</p>' +
          '</div>'
        );
      })
      .join('');

    stageHost.innerHTML = built.panels
      .map(function (p, i) {
        return (
          '<div class="tour-panel" data-tour-step="' +
          i +
          '">' +
          p.html +
          '</div>'
        );
      })
      .join('');

    if (dotsHost) {
      dotsHost.innerHTML = built.panels
        .map(function (_, i) {
          return '<button type="button" class="tour-dot" aria-label="Go to step ' + (i + 1) + '" data-dot="' + i + '"></button>';
        })
        .join('');
      dotsHost.querySelectorAll('[data-dot]').forEach(function (btn) {
        btn.addEventListener('click', function () {
          var i = parseInt(btn.getAttribute('data-dot'), 10);
          var track = root.querySelector('.tour-track');
          if (!track) return;
          var rect = track.getBoundingClientRect();
          var top = window.scrollY + rect.top;
          var scrollable = Math.max(1, track.offsetHeight - window.innerHeight);
          var y = top + (i / Math.max(1, STEP_COUNT - 1)) * scrollable;
          window.scrollTo({ top: y, behavior: reduceMotion ? 'auto' : 'smooth' });
        });
      });
    }

    var track = root.querySelector('.tour-track');
    var lastStep = -1;
    function onScroll() {
      if (!track) return;
      var p = progressFor(track);
      var step = Math.min(
        STEP_COUNT - 1,
        Math.floor(p * STEP_COUNT + 1e-6)
      );
      if (p >= 0.999) step = STEP_COUNT - 1;
      if (step !== lastStep) {
        lastStep = step;
        setStep(root, step);
      }
      root.style.setProperty('--tour-progress', String(p));
    }

    setStep(root, 0);
    onScroll();
    window.addEventListener('scroll', onScroll, { passive: true });
    window.addEventListener('resize', onScroll);
    if (built.forwardBack) initTourForwardBack(built.forwardBack);
  }

  function showError(root, msg, shareUrl) {
    var stage = root.querySelector('[data-tour-stage]');
    var copy = root.querySelector('[data-tour-copy]');
    if (copy) {
      copy.innerHTML =
        '<div class="tour-copy-step is-active">' +
        '<p class="tour-kicker">Demo</p>' +
        '<h2>See a real bout breakdown</h2>' +
        '<p>' +
        esc(msg || 'Demo analysis is temporarily unavailable.') +
        '</p>' +
        '<a class="btn-primary" href="/demo">Analyze my fencing</a>' +
        '</div>';
    }
    if (stage) {
      stage.innerHTML =
        '<div class="tour-panel is-active tour-cta-panel">' +
        '<a class="btn-secondary" href="' +
        esc(shareUrl || '/result') +
        '">Try opening the shared analysis</a>' +
        '</div>';
    }
  }

  async function boot() {
    var root = document.getElementById('product-tour');
    if (!root) return;
    var token = (
      (root.getAttribute('data-demo-share') || '') ||
      window.DEMO_SHARE_TOKEN ||
      DEFAULT_TOKEN
    ).trim();
    var shareUrl = '/result?share=' + encodeURIComponent(token);
    try {
      var res = await fetch(
        '/api/queue/shared-results/' +
          encodeURIComponent(token) +
          '?lite=1',
        { credentials: 'same-origin' }
      );
      var data = await res.json();
      if (!res.ok || !data.success || !data.results) {
        throw new Error((data && data.error) || 'Could not load demo bout');
      }
      var built = buildPanels(data, shareUrl);
      initTour(root, built);
      root.classList.add('is-ready');
    } catch (e) {
      showError(root, e.message || String(e), shareUrl);
      root.classList.add('is-ready');
    }
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', boot);
  } else {
    boot();
  }
})();
