/**
 * Modal UI: share a touch or full analysis with another user by email/username.
 */
(function (global) {
    'use strict';

    let modalEl = null;

    function ensureModal() {
        if (modalEl) return modalEl;
        modalEl = document.createElement('div');
        modalEl.id = 'touchUserShareModal';
        modalEl.className = 'hidden fixed inset-0 z-[200] flex items-center justify-center p-4';
        modalEl.innerHTML = `
            <div class="absolute inset-0 bg-black/70" data-close-modal></div>
            <div class="relative w-full max-w-md bg-gray-800 border border-gray-600 rounded-xl shadow-xl p-5 text-white">
                <h3 id="touchUserShareTitle" class="text-lg font-semibold mb-1">Share touch with user</h3>
                <p id="touchUserShareKindLine" class="text-xs text-violet-300/90 mb-1 hidden"></p>
                <p id="touchUserShareRefLine" class="text-xs font-mono text-sky-400/90 mb-2 hidden"></p>
                <p id="touchUserShareHint" class="text-xs text-gray-400 mb-4">Recipient must have a SmarterFencing account. They will see this touch and your comment in Messages.</p>
                <label class="text-xs text-gray-400 block mb-1" for="touchShareRecipientEmail">Recipient email or username</label>
                <input type="text" id="touchShareRecipientEmail" class="w-full bg-gray-900 border border-gray-600 rounded-lg px-3 py-2 text-sm mb-3" placeholder="coach@example.com or coach_name" autocomplete="username">
                <label class="text-xs text-gray-400 block mb-1" for="touchShareComment">Your comment</label>
                <textarea id="touchShareComment" rows="3" maxlength="4000" class="w-full bg-gray-900 border border-gray-600 rounded-lg px-3 py-2 text-sm resize-y mb-4" placeholder="Can you look at this attack?"></textarea>
                <p id="touchUserShareStatus" class="text-xs text-red-400 min-h-[1rem] mb-3"></p>
                <div class="flex flex-wrap gap-2 justify-end">
                    <button type="button" class="px-3 py-1.5 rounded-lg text-sm bg-gray-700 hover:bg-gray-600" data-close-modal>Cancel</button>
                    <button type="button" id="touchUserShareSubmit" class="px-3 py-1.5 rounded-lg text-sm bg-indigo-600 hover:bg-indigo-500 font-medium">Send</button>
                </div>
            </div>
        `;
        document.body.appendChild(modalEl);
        modalEl.querySelectorAll('[data-close-modal]').forEach((el) => {
            el.addEventListener('click', closeModal);
        });
        document.getElementById('touchUserShareSubmit').addEventListener('click', submitModal);
        return modalEl;
    }

    function closeModal() {
        if (modalEl) modalEl.classList.add('hidden');
    }

    let pendingContext = null;

    function isAnalysisShare(ctx) {
        return (ctx && ctx.kind) === 'analysis';
    }

    function openModal(ctx) {
        pendingContext = ctx || {};
        ensureModal();
        const analysis = isAnalysisShare(pendingContext);
        const title = document.getElementById('touchUserShareTitle');
        if (title) {
            title.textContent = analysis ? 'Share analysis with user' : 'Share touch with user';
        }
        const kindLine = document.getElementById('touchUserShareKindLine');
        if (kindLine) {
            if (analysis) {
                kindLine.textContent = 'Sharing: full bout analysis';
                kindLine.classList.remove('hidden');
            } else {
                kindLine.textContent = '';
                kindLine.classList.add('hidden');
            }
        }
        const refLine = document.getElementById('touchUserShareRefLine');
        if (refLine) {
            if (!analysis && pendingContext.touchRef) {
                refLine.textContent = `Touch ID: ${pendingContext.touchRef}`;
                refLine.classList.remove('hidden');
            } else {
                refLine.textContent = '';
                refLine.classList.add('hidden');
            }
        }
        const hint = document.getElementById('touchUserShareHint');
        if (hint) {
            hint.textContent = analysis
                ? 'Recipient must have a SmarterFencing account. They will see the full analysis and your comment in Messages.'
                : 'Recipient must have a SmarterFencing account. They will see this touch and your comment in Messages.';
        }
        const comment = document.getElementById('touchShareComment');
        if (comment) {
            comment.placeholder = analysis
                ? 'Can you review this bout with me?'
                : 'Can you look at this attack?';
        }
        document.getElementById('touchShareRecipientEmail').value = '';
        document.getElementById('touchShareComment').value = '';
        document.getElementById('touchUserShareStatus').textContent = '';
        modalEl.classList.remove('hidden');
        document.getElementById('touchShareRecipientEmail').focus();
    }

    async function submitModal() {
        const statusEl = document.getElementById('touchUserShareStatus');
        const btn = document.getElementById('touchUserShareSubmit');
        if (!pendingContext) return;

        const recipient = (document.getElementById('touchShareRecipientEmail').value || '').trim();
        const comment = (document.getElementById('touchShareComment').value || '').trim();
        if (!recipient) {
            statusEl.textContent = 'Enter recipient email or username.';
            return;
        }
        if (!comment) {
            statusEl.textContent = 'Add a comment for your coach.';
            return;
        }

        btn.disabled = true;
        statusEl.textContent = 'Sending…';
        statusEl.className = 'text-xs text-gray-400 min-h-[1rem] mb-3';

        const analysis = isAnalysisShare(pendingContext);
        const body = {
            job_id: pendingContext.jobId,
            recipients: [recipient],
            comment,
            kind: analysis ? 'analysis' : 'touch',
        };
        if (!analysis) {
            body.touch_id = pendingContext.touchId;
        }

        try {
            const res = await pendingContext.apiFetch('/api/touch-shares', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(body),
            });
            const data = await res.json();
            if (!data.success) {
                if (data.needs_username) {
                    statusEl.textContent = data.error || 'Set a username first.';
                    statusEl.className = 'text-xs text-amber-400 min-h-[1rem] mb-3';
                    setTimeout(() => {
                        window.location.href = '/profile';
                    }, 1200);
                    return;
                }
                throw new Error(data.error || 'Send failed');
            }
            closeModal();
            if (pendingContext.onSuccess) pendingContext.onSuccess(data);
            else {
                alert(
                    analysis
                        ? 'Analysis shared! They can view it in Messages.'
                        : 'Touch shared! They can view it in Messages.'
                );
            }
        } catch (e) {
            statusEl.textContent = e.message || String(e);
            statusEl.className = 'text-xs text-red-400 min-h-[1rem] mb-3';
        } finally {
            btn.disabled = false;
        }
    }

    global.TouchUserShare = {
        open: openModal,
        close: closeModal,
    };
})(window);
