(function () {
  function copyText(text) {
    if (navigator.clipboard && navigator.clipboard.writeText) {
      return navigator.clipboard.writeText(text);
    }
    // Fallback
    const ta = document.createElement('textarea');
    ta.value = text;
    ta.style.position = 'fixed';
    ta.style.top = '-9999px';
    document.body.appendChild(ta);
    ta.focus();
    ta.select();
    try { document.execCommand('copy'); } catch (e) {}
    document.body.removeChild(ta);
    return Promise.resolve();
  }

  function makeButton(pre, code) {
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'code-copy-button';
    btn.setAttribute('aria-label', 'Copy code');
    btn.textContent = 'Copy';
    btn.addEventListener('click', function (e) {
      e.stopPropagation();
      copyText(code.textContent).then(() => {
        const prev = btn.textContent;
        btn.textContent = 'Copied!';
        btn.classList.add('copied');
        setTimeout(() => {
          btn.textContent = prev;
          btn.classList.remove('copied');
        }, 1200);
      });
    });
    pre.appendChild(btn);
  }

  document.addEventListener('DOMContentLoaded', function () {
    document.querySelectorAll('pre > code').forEach(function (code) {
      const pre = code.parentElement;
      if (!pre.querySelector('.code-copy-button')) {
        makeButton(pre, code);
      }
    });
    // (plain citation uses fenced code, handled above)
  });
})();
