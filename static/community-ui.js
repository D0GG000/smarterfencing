/**
 * Modal UI: post an analysis, touch clip, or highlight reel to the public Community Hub.
 */
(function (global) {
    'use strict';

    let modalEl = null;
    let pendingContext = null;

    function ensureModal() {
        if (modalEl) return modalEl;
        modalEl = document.createElement('div');
        modalEl.id = 'communityShareModal';
        modalEl.className = 'hidden fixed inset-0 z-[200] flex items-center justify-center p-4';
        modalEl.innerHTML = `
            <div class="absolute inset-0 bg-black/70" data-close-modal></div>
            <div class="relative w-full max-w-md bg-gray-800 border border-gray-600 rounded-xl shadow-xl p-5 text-white">
                <h3 class="text-lg font-semibold mb-1">Post to Community Hub</h3>
                <p id="communityShareKindLine" class="text-xs text-emerald-300/90 mb-1"></p>
                <p id="communityShareRefLine" class="text-xs font-mono text-sky-400/90 mb-2 hidden"></p>
                <p class="text-xs text-gray-400 mb-4">This is public. Anyone can view it and comment on the touch.</p>
                <label class="text-xs text-gray-400 block mb-1" for="communityShareCaption">Caption</label>
                <textarea id="communityShareCaption" rows="3" maxlength="4000" class="w-full bg-gray-900 border border-gray-600 rounded-lg px-3 py-2 text-sm resize-y mb-4" placeholder="What happened on this touch? Ask for feedback or show off."></textarea>
                <p id="communityShareStatus" class="text-xs text-red-400 min-h-[1rem] mb-3"></p>
                <div class="flex flex-wrap gap-2 justify-end">
                    <button type="button" class="px-3 py-1.5 rounded-lg text-sm bg-gray-700 hover:bg-gray-600" data-close-modal>Cancel</button>
                    <button type="button" id="communityShareSubmit" class="px-3 py-1.5 rounded-lg text-sm bg-emerald-600 hover:bg-emerald-500 font-medium">Post publicly</button>
                </div>
            </div>
        `;
        document.body.appendChild(modalEl);
        modalEl.querySelectorAll('[data-close-modal]').forEach((el) => {
            el.addEventListener('click', closeModal);
        });
        document.getElementById('communityShareSubmit').addEventListener('click', submitModal);
        return modalEl;
    }

    function closeModal() {
        if (modalEl) modalEl.classList.add('hidden');
    }

    function kindLabel(kind) {
        if (kind === 'highlight_reel') return 'Sharing: highlight reel';
        if (kind === 'analysis') return 'Sharing: full bout analysis';
        return 'Sharing: a single touch';
    }

    function openModal(ctx) {
        pendingContext = ctx || {};
        ensureModal();
        const kindLine = document.getElementById('communityShareKindLine');
        if (kindLine) kindLine.textContent = kindLabel(pendingContext.kind);
        const refLine = document.getElementById('communityShareRefLine');
        if (refLine) {
            if (pendingContext.touchRef) {
                refLine.textContent = `Touch ID: ${pendingContext.touchRef}`;
                refLine.classList.remove('hidden');
            } else {
                refLine.textContent = '';
                refLine.classList.add('hidden');
            }
        }
        document.getElementById('communityShareCaption').value = '';
        document.getElementById('communityShareStatus').textContent = '';
        modalEl.classList.remove('hidden');
        document.getElementById('communityShareCaption').focus();
    }

    async function submitModal() {
        const statusEl = document.getElementById('communityShareStatus');
        const btn = document.getElementById('communityShareSubmit');
        if (!pendingContext) return;

        const caption = (document.getElementById('communityShareCaption').value || '').trim();

        btn.disabled = true;
        statusEl.textContent = 'Posting…';
        statusEl.className = 'text-xs text-gray-400 min-h-[1rem] mb-3';

        try {
            const res = await pendingContext.apiFetch('/api/community/posts', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    job_id: pendingContext.jobId,
                    touch_id: pendingContext.touchId || null,
                    kind: pendingContext.kind,
                    caption,
                }),
            });
            const data = await res.json();
            if (!data.success) {
                if (data.needs_username) {
                    statusEl.textContent = data.error || 'Set a username first.';
                    statusEl.className = 'text-xs text-amber-400 min-h-[1rem] mb-3';
                    setTimeout(() => { window.location.href = '/profile'; }, 1200);
                    return;
                }
                throw new Error((data.error || 'Post failed') + (data.detail ? ` (${data.detail})` : ''));
            }
            closeModal();
            if (pendingContext.onSuccess) pendingContext.onSuccess(data);
            else {
                const go = confirm('Posted to the Community Hub! View it now?');
                if (go && data.post_url) window.location.href = data.post_url;
            }
        } catch (e) {
            statusEl.textContent = e.message || String(e);
            statusEl.className = 'text-xs text-red-400 min-h-[1rem] mb-3';
        } finally {
            btn.disabled = false;
        }
    }

    global.CommunityShare = {
        open: openModal,
        close: closeModal,
    };
})(window);
