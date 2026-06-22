/* Vritti AI Labs landing animations.
   Design rule: every element is fully visible in CSS by default. GSAP layers
   polish on top. Performance-first: no constant per-frame work tied to scroll,
   no continuous backround-position animations on big text, scroll handlers
   throttled to rAF, hover effects skipped on touch. */

(function () {
  const reduce = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  const hasHover = window.matchMedia('(hover: hover)').matches;
  if (typeof gsap === 'undefined') return;
  if (typeof ScrollTrigger !== 'undefined') gsap.registerPlugin(ScrollTrigger);

  /* ---------- Scroll progress bar (rAF-throttled) ---------- */
  const bar = document.getElementById('scrollProgress');
  if (bar) {
    let ticking = false;
    const update = () => {
      const h = document.documentElement;
      const max = h.scrollHeight - h.clientHeight;
      bar.style.width = max > 0 ? (h.scrollTop / max * 100) + '%' : '0%';
      ticking = false;
    };
    window.addEventListener('scroll', () => {
      if (!ticking) { requestAnimationFrame(update); ticking = true; }
    }, { passive: true });
    update();
  }

  /* ---------- Hero entry: slide up (one-shot) ---------- */
  if (!reduce) {
    const heroItems = Array.from(document.querySelectorAll('.hero .reveal'))
      .sort((a, b) => +a.dataset.reveal - +b.dataset.reveal);
    gsap.from(heroItems, { y: 16, duration: 0.6, ease: 'power3.out', stagger: 0.07, immediateRender: true });
  }

  /* ---------- Floating orbs (slow yoyo, off the main scroll path) ---------- */
  if (!reduce) {
    gsap.to('.orb-a', { x: 40, y: -25, duration: 16, repeat: -1, yoyo: true, ease: 'sine.inOut' });
    gsap.to('.orb-b', { x: -32, y: 22, duration: 20, repeat: -1, yoyo: true, ease: 'sine.inOut' });
  }

  /* The continuous background-position gradient sweep on accent text was
     repainting a big text block every frame. Replaced with a one-shot
     entry reveal in CSS (static gradient stays). */

  /* ---------- One-shot reveal-on-scroll for sections ---------- */
  if (!reduce && typeof ScrollTrigger !== 'undefined') {
    document.querySelectorAll('.reveal-on-scroll').forEach(el => {
      const delay = parseFloat(el.dataset.delay || '0');
      gsap.from(el, {
        y: 24, duration: 0.55, ease: 'power3.out', delay,
        scrollTrigger: { trigger: el, start: 'top 88%', toggleActions: 'play none none none', once: true },
      });
    });
  }

  /* ---------- Animated counters (one-shot, viewport entry) ---------- */
  function formatNum(value, format) {
    const v = Math.round(value);
    if (format === 'compact') {
      if (v >= 1_000_000) return (v / 1_000_000).toFixed(v % 1_000_000 === 0 ? 0 : 1) + 'M';
      if (v >= 1_000)     return (v / 1_000).toFixed(v % 1_000 === 0 ? 0 : 1) + 'K';
      return v.toString();
    }
    return v.toLocaleString();
  }
  document.querySelectorAll('.counter').forEach(el => {
    const from = parseFloat(el.dataset.from || '0');
    const to   = parseFloat(el.dataset.to   || '0');
    const fmt  = el.dataset.format || 'int';
    if (reduce || typeof ScrollTrigger === 'undefined') { el.textContent = formatNum(to, fmt); return; }
    el.textContent = formatNum(to, fmt);
    const state = { v: from };
    ScrollTrigger.create({
      trigger: el, start: 'top 85%', once: true,
      onEnter: () => {
        el.textContent = formatNum(from, fmt);
        gsap.to(state, {
          v: to, duration: 1.4, ease: 'power2.out',
          onUpdate: () => { el.textContent = formatNum(state.v, fmt); },
          onComplete: () => { el.textContent = formatNum(to, fmt); },
        });
      },
    });
  });

  /* Hero background parallax was a scrub-on-scroll handler that ran every
     scroll tick and forced layout reads. Dropped entirely. The static orbs
     and grid look fine without it. */

  /* ---------- Smooth-scroll anchors ---------- */
  document.querySelectorAll('a[href^="#"]').forEach(a => {
    a.addEventListener('click', e => {
      const id = a.getAttribute('href').slice(1);
      if (!id) return;
      const target = document.getElementById(id);
      if (!target) return;
      e.preventDefault();
      const y = target.getBoundingClientRect().top + window.pageYOffset - 56;
      window.scrollTo({ top: y, behavior: reduce ? 'auto' : 'smooth' });
      history.replaceState(null, '', '#' + id);
    });
  });

  /* ---------- Lesson card tilt: hover-only, off touch ---------- */
  if (!reduce && hasHover) {
    document.querySelectorAll('.lcard').forEach(card => {
      let raf = 0;
      card.addEventListener('pointermove', e => {
        if (raf) return;
        raf = requestAnimationFrame(() => {
          const r = card.getBoundingClientRect();
          const px = (e.clientX - r.left) / r.width;
          const py = (e.clientY - r.top) / r.height;
          gsap.to(card, { rotateY: (px - 0.5) * 3, rotateX: (0.5 - py) * 3, duration: 0.25, ease: 'power2.out', transformPerspective: 800 });
          raf = 0;
        });
      });
      card.addEventListener('pointerleave', () => {
        gsap.to(card, { rotateX: 0, rotateY: 0, duration: 0.35, ease: 'power3.out' });
      });
    });
  }

  /* ---------- Demo typewriter (one-shot, on viewport entry) ---------- */
  const demo = document.getElementById('demoTyped');
  const trigger = document.getElementById('demoOutput');
  if (demo && trigger) {
    const text = window.__SLM_DEMO_TEXT__ || '';
    if (!text) { demo.textContent = '(demo unavailable)'; }
    else if (reduce || typeof ScrollTrigger === 'undefined') { demo.textContent = text; }
    else {
      demo.textContent = text;
      ScrollTrigger.create({
        trigger, start: 'top 75%', once: true,
        onEnter: () => {
          demo.textContent = '';
          const obj = { i: 0 };
          gsap.to(obj, {
            i: text.length, duration: Math.min(5, 0.018 * text.length), ease: 'none',
            onUpdate: () => { demo.textContent = text.slice(0, Math.floor(obj.i)) + '▍'; },
            onComplete: () => { demo.textContent = text; },
          });
        },
      });
    }
  }

  /* ---------- Sticky nav scrolled border (rAF-throttled) ---------- */
  const nav = document.querySelector('.nav');
  if (nav) {
    let navTicking = false;
    const update = () => {
      nav.classList.toggle('scrolled', window.scrollY > 8);
      navTicking = false;
    };
    window.addEventListener('scroll', () => {
      if (!navTicking) { requestAnimationFrame(update); navTicking = true; }
    }, { passive: true });
    update();
  }
})();
