/* Vritti AI Labs landing — minimal, GSAP-free.
   Replaces the GSAP+ScrollTrigger stack with native APIs:
   IntersectionObserver for reveals + counters + typewriter, CSS for orbs,
   native scroll-behavior: smooth for anchors. ~2KB instead of ~85KB. */

(function () {
  const reduce = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  /* --- Reveal-on-scroll: IntersectionObserver adds .in-view; CSS does the rest --- */
  if (!reduce && 'IntersectionObserver' in window) {
    const io = new IntersectionObserver(entries => {
      for (const e of entries) {
        if (e.isIntersecting) {
          e.target.classList.add('in-view');
          io.unobserve(e.target);
        }
      }
    }, { rootMargin: '0px 0px -10% 0px', threshold: 0.05 });
    document.querySelectorAll('.reveal-on-scroll').forEach(el => io.observe(el));
  } else {
    document.querySelectorAll('.reveal-on-scroll').forEach(el => el.classList.add('in-view'));
  }

  /* --- Counters: tick up once on viewport entry --- */
  function formatNum(value, format) {
    const v = Math.round(value);
    if (format === 'compact') {
      if (v >= 1_000_000) return (v / 1_000_000).toFixed(v % 1_000_000 === 0 ? 0 : 1) + 'M';
      if (v >= 1_000)     return (v / 1_000).toFixed(v % 1_000 === 0 ? 0 : 1) + 'K';
      return v.toString();
    }
    return v.toLocaleString();
  }
  function animateCounter(el) {
    const from = parseFloat(el.dataset.from || '0');
    const to   = parseFloat(el.dataset.to   || '0');
    const fmt  = el.dataset.format || 'int';
    if (reduce) { el.textContent = formatNum(to, fmt); return; }
    const start = performance.now();
    const dur = 1200;
    const step = (t) => {
      const p = Math.min(1, (t - start) / dur);
      const eased = 1 - Math.pow(1 - p, 3);
      el.textContent = formatNum(from + (to - from) * eased, fmt);
      if (p < 1) requestAnimationFrame(step);
      else el.textContent = formatNum(to, fmt);
    };
    requestAnimationFrame(step);
  }
  const counters = document.querySelectorAll('.counter');
  if (counters.length && 'IntersectionObserver' in window) {
    const io = new IntersectionObserver(entries => {
      for (const e of entries) {
        if (e.isIntersecting) {
          animateCounter(e.target);
          io.unobserve(e.target);
        }
      }
    }, { threshold: 0.4 });
    counters.forEach(el => {
      el.textContent = formatNum(parseFloat(el.dataset.to || '0'), el.dataset.format || 'int');
      io.observe(el);
    });
  } else {
    counters.forEach(el => animateCounter(el));
  }

  /* --- Demo typewriter: one-shot on viewport entry --- */
  const demo = document.getElementById('demoTyped');
  const demoTrigger = document.getElementById('demoOutput');
  if (demo && demoTrigger) {
    const text = window.__SLM_DEMO_TEXT__ || '';
    demo.textContent = text;
    if (text && !reduce && 'IntersectionObserver' in window) {
      const io = new IntersectionObserver(entries => {
        for (const e of entries) {
          if (e.isIntersecting) {
            io.disconnect();
            demo.textContent = '';
            let i = 0;
            const total = text.length;
            const stepDur = Math.max(8, Math.min(22, 4500 / total));
            const tick = () => {
              i = Math.min(total, i + 1);
              demo.textContent = text.slice(0, i) + (i < total ? '▍' : '');
              if (i < total) setTimeout(tick, stepDur);
            };
            tick();
          }
        }
      }, { threshold: 0.3 });
      io.observe(demoTrigger);
    }
  }

  /* --- Scroll progress bar (rAF-throttled) --- */
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

  /* --- Sticky nav: add .scrolled when past 8px (IntersectionObserver on a sentinel) --- */
  const nav = document.querySelector('.nav');
  if (nav) {
    // A 1px sentinel at the very top. When it leaves the viewport, the nav is scrolled.
    const sentinel = document.createElement('div');
    sentinel.style.cssText = 'position:absolute;top:0;left:0;width:1px;height:8px;pointer-events:none';
    document.body.prepend(sentinel);
    if ('IntersectionObserver' in window) {
      const io = new IntersectionObserver(([e]) => {
        nav.classList.toggle('scrolled', !e.isIntersecting);
      }, { threshold: 0 });
      io.observe(sentinel);
    }
  }
})();
