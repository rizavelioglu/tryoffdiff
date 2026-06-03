// No jQuery — vanilla DOM ready.
document.addEventListener('DOMContentLoaded', function () {
  // Initialize the teaser carousel if bulma-carousel is loaded.
  if (typeof bulmaCarousel !== 'undefined') {
    bulmaCarousel.attach('.teaser-carousel', {
      slidesToScroll: 1,
      slidesToShow: 1,
      loop: true,
      infinite: true,
      autoplay: true,
      autoplaySpeed: 10000
    });
  }

  // BibTeX copy-to-clipboard button.
  var copyBtn = document.getElementById('copy-bibtex-btn');
  var bibtexEl = document.getElementById('bibtex-code');
  if (copyBtn && bibtexEl) {
    copyBtn.addEventListener('click', function () {
      var label = document.getElementById('copy-bibtex-text');
      var done = function () {
        if (!label) return;
        label.textContent = 'Copied!';
        setTimeout(function () { label.textContent = 'Copy'; }, 1200);
      };
      var fail = function () {
        if (!label) return;
        label.textContent = 'Copy failed';
        setTimeout(function () { label.textContent = 'Copy'; }, 1500);
      };
      if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(bibtexEl.textContent).then(done, fail);
      } else {
        // Legacy fallback.
        try {
          var range = document.createRange();
          range.selectNode(bibtexEl);
          var sel = window.getSelection();
          sel.removeAllRanges();
          sel.addRange(range);
          document.execCommand('copy');
          sel.removeAllRanges();
          done();
        } catch (e) { fail(); }
      }
    });
  }
});
