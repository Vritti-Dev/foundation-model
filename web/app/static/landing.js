/* Vritti AI Labs landing animations.
   Design rule: every element is fully visible in CSS by default. GSAP layers
   POLISH on top (entry slide, parallax, gradient sweep, counter, tilt). If
   GSAP fails to load or animations are reduced, the page still looks right. */

(function () {
  const reduce = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  if (typeof gsap === 'undefined') return;
  if (typeof ScrollTrigger !== 'undefined') gsap.registerPlugin(ScrollTrigger);

  /* ---------- 1. Scroll progress bar ---------- */
  const bar = document.getElementById('scrollProgress');
  if (bar) {
    const onScroll = () => {
      const h = document.documentElement;
      const max = h.scrollHeight - h.clientHeight;
      bar.style.width = max > 0 ? (h.scrollTop / max * 100) + '%' : '0%';
    };
    window.addEventListener('scroll', onScroll, { passive: true });
    onScroll();
  }

  /* ---------- 2. Hero entry: slide up from a slight offset. Opacity stays
       visible the whole time, so the screenshot or a slow phone never blanks. */
  if (!reduce) {
    const heroItems = Array.from(document.querySelectorAll('.hero .reveal'))
      .sort((a, b) => +a.dataset.reveal - +b.dataset.reveal);
    gsap.from(heroItems, {
      y: 18, duration: 0.7, ease: 'power3.out',
      stagger: 0.08, immediateRender: true,
    });
  }

  /* ---------- 3. Floating background orbs ---------- */
  if (!reduce) {
    gsap.to('.orb-a', { x: 50, y: -30, duration: 14, repeat: -1, yoyo: true, ease: 'sine.inOut' });
    gsap.to('.orb-b', { x: -40, y: 30,  duration: 18, repeat: -1, yoyo: true, ease: 'sine.inOut' });
  }

  /* ---------- 4. Gradient sweep on accent text ---------- */
  if (!reduce) {
    document.querySelectorAll('.gradient-sweep').forEach(el => {
      gsap.fromTo(el,
        { backgroundPositionX: '0%' },
        { backgroundPositionX: '200%', duration: 3.5, repeat: -1, ease: 'none' });
    });
  }

  /* ---------- 5. Subtle entry for each scroll section ---------- */
  if (!reduce && typeof ScrollTrigger !== 'undefined') {
    document.querySelectorAll('.reveal-on-scroll').forEach(el => {
      const delay = parseFloat(el.dataset.delay || '0');
      gsap.from(el, {
        y: 28, duration: 0.7, ease: 'power3.out', delay,
        scrollTrigger: { trigger: el, start: 'top 88%', toggleActions: 'play none none none' },
      });
    });
  }

  /* ---------- 6. Animated counters ---------- */
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
    if (reduce || typeof ScrollTrigger === 'undefined') {
      el.textContent = formatNum(to, fmt); return;
    }
    el.textContent = formatNum(to, fmt);   // final value as default
    const state = { v: from };
    ScrollTrigger.create({
      trigger: el, start: 'top 85%', once: true,
      onEnter: () => {
        el.textContent = formatNum(from, fmt);
        gsap.to(state, {
          v: to, duration: 1.6, ease: 'power2.out',
          onUpdate: () => { el.textContent = formatNum(state.v, fmt); },
          onComplete: () => { el.textContent = formatNum(to, fmt); },
        });
      },
    });
  });

  /* ---------- 7. Hero background parallax ---------- */
  if (!reduce && typeof ScrollTrigger !== 'undefined') {
    gsap.to('.hero-bg', {
      yPercent: 18, ease: 'none',
      scrollTrigger: { trigger: '.hero', start: 'top top', end: 'bottom top', scrub: true },
    });
  }

  /* ---------- 8. Smooth-scroll anchors ---------- */
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

  /* ---------- 9. Lesson card hover tilt ---------- */
  if (!reduce) {
    document.querySelectorAll('.lcard').forEach(card => {
      card.addEventListener('pointermove', e => {
        const r = card.getBoundingClientRect();
        const px = (e.clientX - r.left) / r.width;
        const py = (e.clientY - r.top) / r.height;
        gsap.to(card, { rotateY: (px - 0.5) * 4, rotateX: (0.5 - py) * 4, duration: 0.25, ease: 'power2.out', transformPerspective: 800 });
      });
      card.addEventListener('pointerleave', () => {
        gsap.to(card, { rotateX: 0, rotateY: 0, duration: 0.45, ease: 'power3.out' });
      });
    });
  }

  /* ---------- 10. Typewriter for demo block ---------- */
  const demo = document.getElementById('demoTyped');
  const trigger = document.getElementById('demoOutput');
  if (demo && trigger) {
    const text = window.__SLM_DEMO_TEXT__ || '';
    if (!text) {
      demo.textContent = '(demo unavailable)';
    } else if (reduce || typeof ScrollTrigger === 'undefined') {
      demo.textContent = text;
    } else {
      demo.textContent = text;  // visible by default
      ScrollTrigger.create({
        trigger, start: 'top 75%', once: true,
        onEnter: () => {
          demo.textContent = '';
          const obj = { i: 0 };
          gsap.to(obj, {
            i: text.length, duration: Math.min(6, 0.022 * text.length), ease: 'none',
            onUpdate: () => { demo.textContent = text.slice(0, Math.floor(obj.i)) + '▍'; },
            onComplete: () => { demo.textContent = text; },
          });
        },
      });
    }
  }

  /* ---------- 11. Sticky nav border on scroll ---------- */
  const nav = document.querySelector('.nav');
  if (nav) {
    const update = () => nav.classList.toggle('scrolled', window.scrollY > 8);
    window.addEventListener('scroll', update, { passive: true });
    update();
  }
})();
