// Typewriter effect for the brand title in the sidebar
document.addEventListener('DOMContentLoaded', function () {
  var el = document.querySelector('.book-menu .book-menu-content h2.book-brand a span');
  if (!el) return;

  var fullText = (el.dataset && el.dataset.text) ? el.dataset.text : (el.textContent || '').trim();
  if (!fullText) return;

  // Prepare element for typing
  el.textContent = '';
  el.classList.add('brand-typed');

  var i = 0;
  // Human-like cadence: a bit faster 165–265ms per char
  var minDelay = 165;   // ms
  var maxDelay = 265;   // ms
  var startDelay = 400; // initial delay before typing

  function nextDelay(ch) {
    var d = minDelay + Math.random() * (maxDelay - minDelay);
    if (/[\.!?]/.test(ch)) d += 260;    // longer pause on end punctuation
    else if (/[,:;]/.test(ch)) d += 155; // medium pause on minor punctuation
    if (ch === ' ') d += 70;             // bigger pause between words
    return Math.round(d);
  }

  function typeNext() {
    if (i < fullText.length) {
      el.textContent += fullText.charAt(i);
      i += 1;
      window.setTimeout(typeNext, nextDelay(fullText.charAt(i-1)));
    } else {
      el.classList.add('brand-typed-done');
    }
  }

  window.setTimeout(typeNext, startDelay);
});
