// Preserve links to the old /types/?t=… archives.
const typeHeading = document.querySelector('[data-type-heading]');
if (typeHeading) {
  const requested = new URLSearchParams(location.search).get('t')?.toLowerCase();
  const filter = requested === 'overview' ? 'review' : requested;
  const entries = Array.from(document.querySelectorAll('[data-post-type]'));
  if (entries.some(entry => entry.dataset.postType === filter)) {
    typeHeading.textContent = typeHeading.dataset[filter] || filter;
    entries.forEach(entry => { entry.hidden = entry.dataset.postType !== filter; });
  }
}

// Translated headings share IDs, so readers can switch languages at the same section.
const languageSwitch = document.querySelector('.language-switch');
if (languageSwitch) {
  const translatedPage = new URL(languageSwitch.href);
  if (typeHeading && location.search) translatedPage.search = location.search;
  function updateLanguageLink() {
    translatedPage.hash = location.hash;
    languageSwitch.href = translatedPage.href;
  }
  updateLanguageLink();
  window.addEventListener('hashchange', updateLanguageLink);
}
