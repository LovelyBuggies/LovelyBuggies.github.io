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
  var speed = 50;   // ms per character
  var startDelay = 150; // initial delay before typing (halved)

  function typeNext() {
    if (i < fullText.length) {
      el.textContent += fullText.charAt(i);
      i += 1;
      window.setTimeout(typeNext, speed);
    } else {
      el.classList.add('brand-typed-done');
    }
  }

  window.setTimeout(typeNext, startDelay);
});
