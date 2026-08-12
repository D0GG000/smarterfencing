(() => {
  function initHeader() {
    const drawer = document.getElementById('drawer');
    const scrim = document.getElementById('scrim');
    const openBtn = document.querySelector('button.nav-toggle[aria-controls="drawer"]');
    const headerEl = document.querySelector('body > header');
    const spacerEl = document.querySelector('.header-spacer');

    if (!drawer || !scrim || !openBtn) {
      console.warn('[header] Missing elements', { drawer, scrim, openBtn });
      return;
    }

    function openDrawer() {
      drawer.classList.add('open');
      drawer.setAttribute('aria-hidden', 'false');
      scrim.hidden = false;
      openBtn.setAttribute('aria-expanded', 'true');
      document.body.style.overflow = 'hidden';
    }

    function closeDrawer() {
      drawer.classList.remove('open');
      drawer.setAttribute('aria-hidden', 'true');
      scrim.hidden = true;
      openBtn.setAttribute('aria-expanded', 'false');
      document.body.style.overflow = '';
    }

    // Toggle button
    openBtn.addEventListener('click', (e) => {
      e.preventDefault();
      drawer.classList.contains('open') ? closeDrawer() : openDrawer();
    });

    // Scrim closes drawer
    scrim.addEventListener('click', closeDrawer);

    // Close via Esc
    window.addEventListener('keydown', (e) => {
      if (e.key === 'Escape' && drawer.classList.contains('open')) closeDrawer();
    });

    // Keep header spacer correct
    function syncHeader() {
      const h = headerEl?.offsetHeight || 72;
      document.documentElement.style.setProperty('--header-h', h + 'px');
      if (spacerEl) spacerEl.style.height = h + 'px';
    }
    window.addEventListener('load', syncHeader);
    window.addEventListener('resize', syncHeader);
    syncHeader();

    // Close button inside drawer
    drawer.querySelectorAll('[data-close]').forEach((btn) => {
      btn.addEventListener('click', (e) => {
        if (btn.tagName === 'BUTTON') {
          e.preventDefault();
          closeDrawer();
        }
      });
    });

    // Robust anchor handling inside the drawer
    document.addEventListener('click', (e) => {
      const anchor = e.target.closest('#drawer a');
      if (!anchor) return; // not a drawer anchor

      const href = anchor.getAttribute('href') || '';
      console.log('[header] drawer anchor click path', { href });

      // Always close when a drawer link is clicked
      closeDrawer();

      if (href.startsWith('#')) {
        e.preventDefault();
        const id = href.slice(1);
        const target = document.getElementById(id);
        if (target) {
          const headerH = parseInt(getComputedStyle(document.documentElement)
            .getPropertyValue('--header-h') || '0', 10);
          const y = target.getBoundingClientRect().top + window.pageYOffset - headerH;
          window.scrollTo({ top: y, behavior: 'smooth' });
        } else {
          window.location.hash = id;
        }
      } else {
        // Non-hash links navigate normally after drawer closes
      }
    });
  }

  document.addEventListener('DOMContentLoaded', initHeader);

  /**
   * Copy text to clipboard with execCommand fallback (Safari, denied permissions, etc.).
   * @returns {Promise<boolean>} true when copy likely succeeded
   */
  async function copyTextToClipboard(text) {
    if (!text) return false;
    try {
      if (navigator.clipboard && typeof navigator.clipboard.writeText === 'function') {
        await navigator.clipboard.writeText(text);
        return true;
      }
    } catch (e) {
      /* try fallback */
    }
    try {
      const ta = document.createElement('textarea');
      ta.value = text;
      ta.setAttribute('readonly', '');
      ta.style.cssText = 'position:fixed;left:-9999px;top:0;opacity:0';
      document.body.appendChild(ta);
      ta.focus();
      ta.select();
      ta.setSelectionRange(0, text.length);
      const ok = document.execCommand('copy');
      document.body.removeChild(ta);
      return !!ok;
    } catch (e2) {
      return false;
    }
  }

  /**
   * Show a read-only field the user can select and copy manually.
   */
  function showManualCopyField(container, url) {
    if (!container || !url) return;
    let box = container.querySelector('[data-manual-copy]');
    if (!box) {
      box = document.createElement('div');
      box.setAttribute('data-manual-copy', '1');
      box.className = 'mt-2';
      container.appendChild(box);
    }
    box.replaceChildren();
    const hint = document.createElement('p');
    hint.className = 'text-xs text-amber-400/90 mb-1';
    hint.textContent = 'Select the link below and copy (⌘C / Ctrl+C):';
    const input = document.createElement('input');
    input.type = 'text';
    input.readOnly = true;
    input.value = url;
    input.className = 'w-full text-xs bg-gray-800 border border-gray-600 rounded px-2 py-1.5 text-gray-200';
    const selectAll = () => {
      input.focus();
      input.select();
    };
    input.addEventListener('focus', selectAll);
    input.addEventListener('click', selectAll);
    box.append(hint, input);
    selectAll();
  }

  window.copyTextToClipboard = copyTextToClipboard;
  window.showManualCopyField = showManualCopyField;
})();

