// Shared header behaviour for the main pages: language switch and current-page highlight.
(function () {
  var root = document.documentElement;
  var saved = null;
  // 'oerf-site-language' is the older pages' key; every switch writes both, so the latest choice wins.
  try { saved = localStorage.getItem('oerf-site-language') || localStorage.getItem('oerf-lang'); } catch (e) {}
  root.setAttribute('data-lang', saved === 'en' ? 'en' : 'zh');
  root.setAttribute('lang', saved === 'en' ? 'en' : 'zh-CN');

  document.addEventListener('DOMContentLoaded', function () {
    var page = location.pathname.split('/').pop() || 'index.html';
    document.querySelectorAll('.nav a').forEach(function (a) {
      if (a.getAttribute('href').split('/').pop() === page) a.setAttribute('aria-current', 'page');
    });
    var button = document.querySelector('.lang-btn');
    if (button) button.addEventListener('click', function () {
      var next = root.getAttribute('data-lang') === 'zh' ? 'en' : 'zh';
      root.setAttribute('data-lang', next);
      root.setAttribute('lang', next === 'zh' ? 'zh-CN' : 'en');
      try { localStorage.setItem('oerf-lang', next); localStorage.setItem('oerf-site-language', next); } catch (e) {}
    });
  });
})();
