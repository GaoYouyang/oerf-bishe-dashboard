(() => {
  'use strict';

  const STORAGE_KEY = 'oerf-site-language';
  const SUPPORTED = new Set(['zh', 'en']);
  const root = document.documentElement;

  function normalize(value) {
    const language = String(value || '').toLowerCase();
    if (language.startsWith('en')) return 'en';
    if (language.startsWith('zh')) return 'zh';
    return null;
  }

  function storedLanguage() {
    try {
      return normalize(window.localStorage.getItem(STORAGE_KEY));
    } catch (_error) {
      return null;
    }
  }

  function requestedLanguage() {
    const queryLanguage = normalize(new URLSearchParams(window.location.search).get('lang'));
    return queryLanguage || storedLanguage() || 'zh';
  }

  function translatedValue(element, prefix, language) {
    const preferred = element.getAttribute(`${prefix}-${language}`);
    if (preferred !== null) return preferred;
    return element.getAttribute(`${prefix}-zh`);
  }

  function translateElement(element, language) {
    if (element.hasAttribute('data-i18n-zh') || element.hasAttribute('data-i18n-en')) {
      const value = translatedValue(element, 'data-i18n', language);
      if (value !== null) element.textContent = value;
    }
    if (element.hasAttribute('data-i18n-html-zh') || element.hasAttribute('data-i18n-html-en')) {
      const value = translatedValue(element, 'data-i18n-html', language);
      if (value !== null) element.innerHTML = value;
    }
    for (const attribute of ['placeholder', 'aria-label', 'title', 'alt', 'value']) {
      const prefix = `data-i18n-${attribute}`;
      if (!element.hasAttribute(`${prefix}-zh`) && !element.hasAttribute(`${prefix}-en`)) continue;
      const value = translatedValue(element, prefix, language);
      if (value !== null) element.setAttribute(attribute, value);
    }
  }

  function translateSubtree(container, language = currentLanguage()) {
    if (!(container instanceof Element || container instanceof Document)) return;
    if (container instanceof Element) translateElement(container, language);
    container.querySelectorAll([
      '[data-i18n-zh]',
      '[data-i18n-en]',
      '[data-i18n-html-zh]',
      '[data-i18n-html-en]',
      '[data-i18n-placeholder-zh]',
      '[data-i18n-placeholder-en]',
      '[data-i18n-aria-label-zh]',
      '[data-i18n-aria-label-en]',
      '[data-i18n-title-zh]',
      '[data-i18n-title-en]',
      '[data-i18n-alt-zh]',
      '[data-i18n-alt-en]',
      '[data-i18n-value-zh]',
      '[data-i18n-value-en]'
    ].join(',')).forEach(element => translateElement(element, language));
  }

  function currentLanguage() {
    return SUPPORTED.has(root.dataset.siteLanguage) ? root.dataset.siteLanguage : 'zh';
  }

  function updateDocumentMetadata(language) {
    const title = root.getAttribute(`data-title-${language}`) || root.getAttribute('data-title-zh');
    if (title) document.title = title;
    document.querySelectorAll('meta[data-content-zh], meta[data-content-en]').forEach(meta => {
      const value = meta.getAttribute(`data-content-${language}`) || meta.getAttribute('data-content-zh');
      if (value !== null) meta.setAttribute('content', value);
    });
  }

  function updateControls(language) {
    document.querySelectorAll('[data-site-language-option]').forEach(button => {
      const active = button.dataset.siteLanguageOption === language;
      button.setAttribute('aria-pressed', String(active));
      button.setAttribute('tabindex', active ? '0' : '-1');
    });
    const note = document.querySelector('[data-site-language-page-note]');
    if (note) {
      const translatedCount = document.querySelectorAll('[data-i18n-en], [data-i18n-html-en]').length;
      note.hidden = language !== 'en' || translatedCount > 0;
    }
  }

  function setLanguage(language, options = {}) {
    const normalized = normalize(language) || 'zh';
    root.dataset.siteLanguage = normalized;
    root.lang = normalized === 'en' ? 'en' : 'zh-CN';
    if (options.persist !== false) {
      try {
        window.localStorage.setItem(STORAGE_KEY, normalized);
      } catch (_error) {
        // The language still works for this page when storage is unavailable.
      }
    }
    translateSubtree(document, normalized);
    updateDocumentMetadata(normalized);
    updateControls(normalized);
    window.dispatchEvent(new CustomEvent('oerf:languagechange', { detail: { language: normalized } }));
  }

  function createSwitcher() {
    if (document.querySelector('[data-site-language-switcher]')) return;
    const bar = document.createElement('div');
    bar.className = 'site-language-bar';
    bar.dataset.siteLanguageSwitcher = '';
    bar.innerHTML = `
      <p class="site-language-page-note" data-site-language-page-note hidden>This archived page has not yet been translated; the Chinese source is preserved.</p>
      <div class="site-language-switcher" role="group" aria-label="Language / 语言">
        <span class="site-language-mark" aria-hidden="true">文 / A</span>
        <button class="site-language-option" type="button" lang="zh-CN" data-site-language-option="zh">中文</button>
        <button class="site-language-option" type="button" lang="en" data-site-language-option="en">EN</button>
      </div>`;
    const host = document.querySelector('header') || document.body;
    host.insertBefore(bar, host.firstChild);
    bar.querySelectorAll('[data-site-language-option]').forEach(button => {
      button.addEventListener('click', () => setLanguage(button.dataset.siteLanguageOption));
      button.addEventListener('keydown', event => {
        if (!['ArrowLeft', 'ArrowRight'].includes(event.key)) return;
        event.preventDefault();
        const next = button.dataset.siteLanguageOption === 'zh' ? 'en' : 'zh';
        const target = bar.querySelector(`[data-site-language-option="${next}"]`);
        setLanguage(next);
        target.focus();
      });
    });
  }

  function initialize() {
    createSwitcher();
    setLanguage(requestedLanguage(), { persist: true });
  }

  window.OERFLanguage = Object.freeze({
    get: currentLanguage,
    set: setLanguage,
    translateSubtree
  });

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initialize, { once: true });
  } else {
    initialize();
  }
})();
