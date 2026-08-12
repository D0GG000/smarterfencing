/**
 * Per-touch clip playback (normal speed → slow-mo loop) and focus-mode helpers.
 * Expects #resultsVideo, #touchClipControls, #playTouchClipBtn, #touchClipPhase.
 */
(function (global) {
    'use strict';

    const BEFORE_SEC = 1.0;
    const AFTER_SEC = 0.5;
    const SLOW_MO_RATE = 0.25;
    const END_EPS = 0.04;

    function computeTouchClip(touchFrame, fps, videoDuration) {
        const rate = fps && fps > 0 ? fps : 30;
        const time = touchFrame / rate;
        let clipStart = Math.max(0, time - BEFORE_SEC);
        let clipEnd = time + AFTER_SEC;
        if (videoDuration && Number.isFinite(videoDuration) && videoDuration > 0) {
            clipEnd = Math.min(clipEnd, videoDuration);
            clipStart = Math.min(clipStart, Math.max(0, clipEnd - 0.05));
        }
        return { clipStart, clipEnd };
    }

    function seekVideoToSeconds(video, targetSeconds, onDone) {
        if (!video || !video.src) {
            if (onDone) onDone();
            return;
        }
        const t = Math.max(0, targetSeconds);
        let programmatic = false;

        const applySeek = () => {
            try {
                programmatic = true;
                if (video.seekable && video.seekable.length > 0) {
                    const lo = video.seekable.start(0);
                    const hi = video.seekable.end(video.seekable.length - 1);
                    video.currentTime = Math.min(Math.max(t, lo), hi);
                } else {
                    video.currentTime = t;
                }
            } catch (e) {
                /* ignore */
            }
            requestAnimationFrame(() => {
                programmatic = false;
                if (onDone) onDone();
            });
        };

        const run = () => requestAnimationFrame(applySeek);
        if (video.readyState >= 1) run();
        else video.addEventListener('loadedmetadata', run, { once: true });

        return () => programmatic;
    }

    function findTouchIndex(allResults, touchId) {
        if (!touchId || !Array.isArray(allResults)) return -1;
        return allResults.findIndex((r) => r && r.touch === touchId);
    }

    class TouchClipPlayer {
        constructor() {
            this.video = null;
            this.clipStart = 0;
            this.clipEnd = 0;
            this.phase = 'idle';
            this.active = false;
            this._programmaticSeek = false;
            this._mutedBeforeClip = null;
            this._onTimeUpdate = this._onTimeUpdate.bind(this);
            this._onSeeking = this._onSeeking.bind(this);
            this._onPlay = this._onPlay.bind(this);
            this._onPause = this._onPause.bind(this);
        }

        _els() {
            return {
                controls: document.getElementById('touchClipControls'),
                playBtn: document.getElementById('playTouchClipBtn'),
                phase: document.getElementById('touchClipPhase'),
            };
        }

        _setPhaseLabel(text) {
            const { phase } = this._els();
            if (phase) phase.textContent = text || '';
        }

        _setPlayButton(playing) {
            const { playBtn } = this._els();
            if (!playBtn) return;
            playBtn.textContent = playing ? 'Stop' : 'Play touch';
            playBtn.setAttribute('aria-pressed', playing ? 'true' : 'false');
        }

        _showControls(show) {
            const { controls } = this._els();
            if (controls) controls.classList.toggle('hidden', !show);
        }

        _restoreAudio() {
            if (!this.video || this._mutedBeforeClip === null) return;
            this.video.muted = this._mutedBeforeClip;
            this._mutedBeforeClip = null;
        }

        stop() {
            if (this.video) {
                this.video.removeEventListener('timeupdate', this._onTimeUpdate);
                this.video.removeEventListener('seeking', this._onSeeking);
                this.video.removeEventListener('play', this._onPlay);
                this.video.removeEventListener('pause', this._onPause);
                this.video.playbackRate = 1;
                this._restoreAudio();
            }
            this.active = false;
            this.phase = 'idle';
            this._programmaticSeek = false;
            this._setPlayButton(false);
            this._setPhaseLabel('');
        }

        prepare(touchFrame, fps) {
            const video = document.getElementById('resultsVideo');
            this.video = video;
            if (!video || !video.src) {
                this._showControls(false);
                return;
            }
            const dur = video.duration;
            const clip = computeTouchClip(
                touchFrame,
                fps,
                Number.isFinite(dur) ? dur : null
            );
            this.clipStart = clip.clipStart;
            this.clipEnd = clip.clipEnd;
            this.stop();
            this._showControls(true);
            seekVideoToSeconds(video, this.clipStart);
        }

        _runProgrammaticSeek(targetSeconds, onDone) {
            if (!this.video) {
                if (onDone) onDone();
                return;
            }
            this._programmaticSeek = true;
            seekVideoToSeconds(this.video, targetSeconds, () => {
                requestAnimationFrame(() => {
                    this._programmaticSeek = false;
                    if (onDone) onDone();
                });
            });
        }

        play() {
            const video = this.video || document.getElementById('resultsVideo');
            if (!video || !video.src) return;
            this.video = video;

            this.stop();
            this.active = true;
            this.phase = 'normal';
            video.playbackRate = 1;
            this._mutedBeforeClip = video.muted;

            video.addEventListener('timeupdate', this._onTimeUpdate);
            video.addEventListener('seeking', this._onSeeking);
            video.addEventListener('play', this._onPlay);
            video.addEventListener('pause', this._onPause);

            this._setPlayButton(true);
            this._setPhaseLabel('Normal speed');

            this._runProgrammaticSeek(this.clipStart, () => {
                video.play().catch(() => {
                    this.stop();
                });
            });
        }

        toggle() {
            if (this.active) this.stop();
            else this.play();
        }

        _onPlay() {
            this._setPlayButton(true);
        }

        _onPause() {
            if (!this.active || this._programmaticSeek) return;
            this.stop();
        }

        _onSeeking() {
            if (!this.active || this._programmaticSeek) return;
            this.stop();
        }

        _onTimeUpdate() {
            if (!this.active || !this.video) return;
            if (this.video.currentTime < this.clipEnd - END_EPS) return;

            if (this.phase === 'normal') {
                this.phase = 'slowmo';
                this.video.playbackRate = SLOW_MO_RATE;
                this.video.muted = true;
                this._setPhaseLabel('Slow motion (0.25×, muted)');
                this._runProgrammaticSeek(this.clipStart, () => {
                    this.video.play().catch(() => this.stop());
                });
                return;
            }

            if (this.phase === 'slowmo') {
                this._runProgrammaticSeek(this.clipStart, () => {
                    this.video.play().catch(() => this.stop());
                });
            }
        }
    }

    const player = new TouchClipPlayer();

    function applyFocusMode(enabled, opts) {
        opts = opts || {};
        document.body.classList.toggle('touch-focus-mode', !!enabled);
        const banner = document.getElementById('touchFocusBanner');
        if (banner) {
            banner.classList.toggle('hidden', !enabled);
            if (enabled) {
                banner.textContent = opts.readOnly
                    ? 'Shared touch · view only'
                    : 'Single touch view';
            }
        }
        const guestBanner = document.getElementById('guestViewBanner');
        if (guestBanner && enabled && opts.readOnly) {
            guestBanner.classList.add('hidden');
        }
    }

    function readTouchParamFromUrl() {
        try {
            return new URLSearchParams(window.location.search).get('touch') || '';
        } catch (e) {
            return '';
        }
    }

    function frameNumFromTouchKey(touchKey) {
        if (!touchKey) return 0;
        const m = String(touchKey).match(/frame(\d+)/i);
        return m ? parseInt(m[1], 10) : 0;
    }

    function jobTouchPrefix(jobId) {
        const raw = String(jobId || 'unknown').toUpperCase();
        return raw.length >= 6 ? raw.slice(0, 6) : raw.padEnd(6, '0').slice(0, 6);
    }

    function ensureTouchRefs(predictions, jobId) {
        if (!Array.isArray(predictions) || !jobId) return predictions;
        const prefix = jobTouchPrefix(jobId);
        const used = new Set();
        predictions.forEach((p) => {
            const ref = p && p.touch_ref;
            if (!ref) return;
            const m = String(ref).toUpperCase().match(new RegExp(`^${prefix}-(\\d+)$`));
            if (m) used.add(parseInt(m[1], 10));
        });
        const missing = predictions.filter((p) => p && p.touch && !p.touch_ref);
        missing.sort((a, b) => frameNumFromTouchKey(a.touch) - frameNumFromTouchKey(b.touch));
        let n = 1;
        missing.forEach((p) => {
            while (used.has(n)) n += 1;
            p.touch_ref = `${prefix}-${String(n).padStart(2, '0')}`;
            used.add(n);
            n += 1;
        });
        return predictions;
    }

    function formatTouchRefLabel(pred) {
        return pred && pred.touch_ref ? pred.touch_ref : '';
    }

    function selectInitialTouch(parsed, selectTouchFn) {
        const touchId = global._pendingTouchFocusId || readTouchParamFromUrl();
        if (touchId && parsed && parsed.length) {
            const found = parsed.find((p) => p.touch === touchId);
            if (found) {
                selectTouchFn(found.idx);
                applyFocusMode(true, { readOnly: !!global.resultsReadOnly });
                return;
            }
        }
        if (parsed && parsed.length) selectTouchFn(parsed[0].idx);
    }

    function updateTouchShareEditor(canShare) {
        const copyBtn = document.getElementById('copyTouchLinkBtn');
        const userBtn = document.getElementById('shareTouchUserBtn');
        const communityBtn = document.getElementById('postTouchCommunityBtn');
        if (copyBtn) copyBtn.classList.toggle('hidden', !canShare);
        if (userBtn) userBtn.classList.toggle('hidden', !canShare);
        if (communityBtn) communityBtn.classList.toggle('hidden', !canShare);
    }

    function _shareFallbackHost() {
        const host = document.getElementById('touchShareFallbackHost');
        return host || document.getElementById('touchClipControls');
    }

    function _clearShareFallback() {
        const host = document.getElementById('touchShareFallbackHost');
        if (host) {
            host.replaceChildren();
            host.classList.add('hidden');
        }
    }

    async function copyTouchShareLink({ touchId, jobId, apiFetch }) {
        const statusEl = document.getElementById('touchShareStatus');
        const btn = document.getElementById('copyTouchLinkBtn');
        if (!touchId) {
            if (statusEl) statusEl.textContent = 'Select a touch first.';
            return;
        }
        const prev = btn ? btn.textContent : '';
        if (btn) {
            btn.disabled = true;
            btn.textContent = 'Creating…';
        }
        if (statusEl) statusEl.textContent = '';
        try {
            let shareUrl;
            const params = new URLSearchParams(window.location.search);
            const existingShare = params.get('share');
            if (existingShare) {
                const u = new URL(window.location.href);
                u.searchParams.set('touch', touchId);
                u.searchParams.delete('job_id');
                u.searchParams.delete('video');
                u.searchParams.delete('data');
                shareUrl = u.toString();
            } else {
                if (!jobId) throw new Error('Sign in and save results to create a share link.');
                const res = await apiFetch(
                    `/api/queue/jobs/${encodeURIComponent(jobId)}/share`,
                    {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({}),
                    }
                );
                const data = await res.json();
                if (!data.success) throw new Error(data.error || 'Could not create link');
                const u = new URL(data.share_url);
                u.searchParams.set('touch', touchId);
                shareUrl = u.toString();
            }
            const copied = await (global.copyTextToClipboard
                ? global.copyTextToClipboard(shareUrl)
                : Promise.resolve(false));
            const fallbackHost = _shareFallbackHost();
            if (copied) {
                if (statusEl) statusEl.textContent = 'Link copied!';
                _clearShareFallback();
            } else {
                if (global.showManualCopyField && fallbackHost) {
                    if (fallbackHost.id === 'touchShareFallbackHost') {
                        fallbackHost.classList.remove('hidden');
                    }
                    global.showManualCopyField(fallbackHost, shareUrl);
                }
                if (statusEl) {
                    statusEl.textContent = 'Select the link below and copy it.';
                }
            }
            setTimeout(() => {
                if (statusEl) statusEl.textContent = '';
            }, 2500);
        } catch (e) {
            if (statusEl) statusEl.textContent = e.message || String(e);
        } finally {
            if (btn) {
                btn.disabled = false;
                btn.textContent = prev || 'Copy touch link';
            }
        }
    }

    global.TouchClip = {
        BEFORE_SEC,
        AFTER_SEC,
        SLOW_MO_RATE,
        computeTouchClip,
        seekVideoToSeconds,
        findTouchIndex,
        player,
        applyFocusMode,
        readTouchParamFromUrl,
        ensureTouchRefs,
        formatTouchRefLabel,
        frameNumFromTouchKey,
        jobTouchPrefix,
        selectInitialTouch,
        updateTouchShareEditor,
        copyTouchShareLink,
        onTouchSelected(touchFrame, fps) {
            player.prepare(touchFrame, fps);
        },
        playSelectedTouchClip() {
            player.toggle();
        },
        stopPlayback() {
            player.stop();
        },
    };
})(window);
