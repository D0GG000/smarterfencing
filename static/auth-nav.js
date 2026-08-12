(() => {
  function esc(s) {
    if (!s) return "";
    const d = document.createElement("div");
    d.textContent = s;
    return d.innerHTML;
  }

  function updateMessagesUnread(count) {
    const link = document.getElementById("navMessagesLink");
    const badge = document.getElementById("navMessagesUnread");
    if (!link || !badge) return;

    const n = Number(count) || 0;
    if (n > 0) {
      link.classList.add("has-unread");
      badge.hidden = false;
      badge.removeAttribute("aria-hidden");
      badge.textContent = n > 99 ? "99+" : String(n);
      link.setAttribute(
        "aria-label",
        n === 1 ? "Messages, 1 unread" : `Messages, ${n} unread`
      );
    } else {
      link.classList.remove("has-unread");
      badge.hidden = true;
      badge.setAttribute("aria-hidden", "true");
      badge.textContent = "";
      link.removeAttribute("aria-label");
    }
  }

  async function refreshNavMessagesUnread() {
    try {
      const r = await fetch("/api/touch-shares/unread-count", {
        credentials: "same-origin",
      });
      if (!r.ok) {
        updateMessagesUnread(0);
        return;
      }
      const data = await r.json();
      updateMessagesUnread(data.success ? data.unread : 0);
    } catch (e) {
      updateMessagesUnread(0);
    }
  }

  function renderAuthUI(d) {
    const headerSlot = document.getElementById("headerAuthSlot");
    const drawerSlot = document.getElementById("drawerAuthSlot");

    if (d.logged_in && d.user) {
      const label =
        d.user.username
          ? `@${d.user.username}`
          : d.user.display_name || d.user.email || d.user.id || "Account";
      if (headerSlot) {
        headerSlot.innerHTML = `<span class="header-auth-user">${esc(
          label
        )}</span> · <a href="/auth/logout">Log out</a>`;
      }
      if (drawerSlot) {
        drawerSlot.innerHTML = `<div class="drawer-user">${esc(
          label
        )}</div><a href="/auth/logout" data-close>Log out</a>`;
      }
      refreshNavMessagesUnread();
    } else {
      updateMessagesUnread(0);
      if (headerSlot) {
        headerSlot.innerHTML = `
          <span class="header-auth-signin">
            <span class="header-auth-label">Sign in</span>
            <span class="header-auth-sep" aria-hidden="true">·</span>
            <a href="/auth/google">With Google</a>
            <span class="header-auth-sep" aria-hidden="true">·</span>
            <a href="/demo#sign-in">With email</a>
          </span>`;
      }
      if (drawerSlot) {
        drawerSlot.innerHTML = `
          <span>Sign in:</span>
          <a href="/auth/google" data-close>Google</a>
          ·
          <a href="/demo#sign-in" data-close>Email</a>`;
      }
    }
  }

  window.refreshNavMessagesUnread = refreshNavMessagesUnread;

  document.addEventListener("DOMContentLoaded", async () => {
    const headerSlot = document.getElementById("headerAuthSlot");
    const drawerSlot = document.getElementById("drawerAuthSlot");
    if (!headerSlot && !drawerSlot) return;
    try {
      const r = await fetch("/api/auth/me", { credentials: "same-origin" });
      const d = await r.json();
      renderAuthUI(d);
    } catch (e) {
      updateMessagesUnread(0);
      if (headerSlot) headerSlot.innerHTML = "";
      if (drawerSlot) drawerSlot.innerHTML = "";
    }
  });
})();
