/*
 * Step-by-step animation player.
 *
 * Markup (see _templates/animation.html):
 *   <div class="stepper" data-interval="2600">
 *     <div class="st-stage"><svg>… elements with data-step="n" …</svg></div>
 *     <ol class="st-steps"><li><h3>Title</h3><p>Caption</p></li> …</ol>
 *   </div>
 *
 * An element with data-step="n" appears at step n and stays. Add
 * data-until="m" to hide it again after step m. Paths/lines with class
 * "draw" are stroked in when they appear.
 */
(function () {
  "use strict";

  const ICONS = {
    restart: '<path d="M3 12a9 9 0 1 0 3-6.7L3 8"/><path d="M3 3v5h5"/>',
    prev: '<path d="m15 18-6-6 6-6"/>',
    next: '<path d="m9 18 6-6-6-6"/>',
    play: '<path d="M7 4.5v15l12-7.5z"/>',
    pause: '<path d="M8 4v16M16 4v16"/>',
    full: '<path d="M8 3H5a2 2 0 0 0-2 2v3M21 8V5a2 2 0 0 0-2-2h-3M3 16v3a2 2 0 0 0 2 2h3M16 21h3a2 2 0 0 0 2-2v-3"/>',
  };

  // Interface text, picked from the page's <html lang>.
  const STRINGS = {
    en: {
      restart: "Restart", prev: "Previous", next: "Next", play: "Play", pause: "Pause",
      replay: "Replay", stepMode: "Step-by-step", full: "Full screen",
      goTo: (n) => "Go to step " + n, count: (n, t) => "Step " + n + " of " + t,
      hint: "<kbd>Space</kbd> play or pause, <kbd>←</kbd> <kbd>→</kbd> previous or next step, " +
        "<kbd>S</kbd> step-by-step, <kbd>R</kbd> restart, <kbd>F</kbd> full screen",
    },
    pt: {
      restart: "Reiniciar", prev: "Anterior", next: "Próximo", play: "Reproduzir", pause: "Pausar",
      replay: "Repetir", stepMode: "Passo a passo", full: "Tela cheia",
      goTo: (n) => "Ir para o passo " + n, count: (n, t) => "Passo " + n + " de " + t,
      hint: "<kbd>Espaço</kbd> reproduzir ou pausar, <kbd>←</kbd> <kbd>→</kbd> passo anterior ou próximo, " +
        "<kbd>S</kbd> passo a passo, <kbd>R</kbd> reiniciar, <kbd>F</kbd> tela cheia",
    },
  };
  const T = STRINGS[(document.documentElement.lang || "en").slice(0, 2).toLowerCase()] || STRINGS.en;

  function icon(name) {
    return '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" ' +
      'stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">' + ICONS[name] + "</svg>";
  }

  function button(cls, iconName, label) {
    const b = document.createElement("button");
    b.type = "button";
    b.className = "st-btn " + cls;
    b.innerHTML = icon(iconName) + (label ? '<span class="st-label">' + label + "</span>" : "");
    if (!label) b.setAttribute("aria-label", iconName === "full" ? T.full : iconName);
    return b;
  }

  function addArrowMarker(svg) {
    if (svg.querySelector("#st-arrow")) return;
    const ns = "http://www.w3.org/2000/svg";
    let defs = svg.querySelector("defs");
    if (!defs) {
      defs = document.createElementNS(ns, "defs");
      svg.prepend(defs);
    }
    defs.insertAdjacentHTML("beforeend",
      '<marker id="st-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" ' +
      'orient="auto-start-reverse"><path d="M0 0L10 5L0 10z" fill="var(--d-stroke)"/></marker>');
  }

  function Stepper(root) {
    const stage = root.querySelector(".st-stage");
    const list = root.querySelector(".st-steps");
    const steps = Array.from(list.children).map((li) => li.innerHTML);
    const total = steps.length;
    const interval = Number(root.dataset.interval) || 2600;
    const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    list.remove();

    stage.querySelectorAll("svg").forEach(addArrowMarker);

    const items = Array.from(stage.querySelectorAll("[data-step]")).map((el) => ({
      el,
      from: Number(el.dataset.step),
      until: el.dataset.until ? Number(el.dataset.until) : Infinity,
    }));

    // Prepare "draw" strokes so they can be revealed with a dash offset.
    stage.querySelectorAll(".draw").forEach((p) => {
      if (typeof p.getTotalLength !== "function") return;
      const len = Math.ceil(p.getTotalLength());
      p.style.strokeDasharray = len + " " + len;
      p.style.strokeDashoffset = len;
      p.dataset.len = len;
    });

    // ---- build the layout ----
    const main = document.createElement("div");
    main.className = "st-main";
    const caption = document.createElement("div");
    caption.className = "st-caption";
    caption.setAttribute("aria-live", "polite");
    root.prepend(main);
    main.append(stage, caption);

    const bar = document.createElement("div");
    bar.className = "st-bar";
    const progress = document.createElement("div");
    progress.className = "st-progress";
    const segs = steps.map((_, i) => {
      const s = document.createElement("button");
      s.type = "button";
      s.setAttribute("aria-label", T.goTo(i + 1));
      s.addEventListener("click", () => { pause(); go(i + 1); });
      progress.append(s);
      return s;
    });

    const controls = document.createElement("div");
    controls.className = "st-controls";
    const bRestart = button("restart", "restart", T.restart);
    const bPrev = button("", "prev", T.prev);
    const bMain = button("primary", "play", T.play);
    const bNext = button("", "next", T.next);
    bNext.innerHTML = '<span class="st-label">' + T.next + "</span>" + icon("next");
    const bFull = button("icon-only", "full");
    const toggle = document.createElement("label");
    toggle.className = "st-toggle";
    toggle.innerHTML = '<input type="checkbox"><span class="st-switch"></span><span>' + T.stepMode + "</span>";
    const toggleInput = toggle.querySelector("input");
    const count = document.createElement("span");
    count.className = "st-count";
    controls.append(bRestart, bPrev, bMain, bNext, bFull, toggle, count);

    const hint = document.createElement("div");
    hint.className = "st-hint";
    hint.innerHTML = T.hint;

    bar.append(progress, controls, hint);
    root.append(bar);

    // ---- state ----
    let step = 1;
    let playing = false;
    let timer = null;
    let started = false;

    function render(animateCaption) {
      for (const it of items) {
        const on = step >= it.from && step <= it.until;
        it.el.classList.toggle("is-on", on);
        it.el.classList.toggle("is-current", on && it.from === step);
        const strokes = it.el.classList.contains("draw") ? [it.el] : it.el.querySelectorAll(".draw");
        strokes.forEach((p) => { if (p.dataset.len) p.style.strokeDashoffset = on ? 0 : p.dataset.len; });
      }
      caption.innerHTML = steps[step - 1];
      if (animateCaption && !reduceMotion) {
        caption.classList.remove("is-changing");
        void caption.offsetWidth; // restart the CSS animation
        caption.classList.add("is-changing");
      }
      segs.forEach((s, i) => s.classList.toggle("is-done", i < step));
      bPrev.disabled = step <= 1;
      bNext.disabled = step >= total;
      count.textContent = T.count(step, total);
      root.dataset.at = step; // lets CSS style a drawing per step: .stepper[data-at="2"] …

      const atEnd = step >= total && !playing;
      const label = playing ? T.pause : atEnd ? T.replay : T.play;
      bMain.innerHTML = icon(playing ? "pause" : atEnd ? "restart" : "play") + '<span class="st-label">' + label + "</span>";
      bMain.setAttribute("aria-label", label);
    }

    function go(n) {
      const next = Math.max(1, Math.min(total, n));
      const changed = next !== step;
      step = next;
      render(changed);
    }

    function schedule() {
      clearTimeout(timer);
      if (!playing) return;
      timer = setTimeout(() => {
        if (step >= total) { pause(); return; }
        go(step + 1);
        if (step >= total) pause(); else schedule();
      }, interval);
    }

    function play() {
      if (step >= total) go(1);
      playing = true;
      render(false);
      schedule();
    }

    function pause() {
      playing = false;
      clearTimeout(timer);
      render(false);
    }

    function togglePlay() { playing ? pause() : play(); }

    function restart() {
      pause();
      go(1);
      if (!toggleInput.checked) play();
    }

    function setStepMode(on) {
      toggleInput.checked = on;
      if (on) pause();
    }

    function fullscreen() {
      if (document.fullscreenElement) document.exitFullscreen();
      else if (root.requestFullscreen) root.requestFullscreen();
    }

    bRestart.addEventListener("click", restart);
    bPrev.addEventListener("click", () => { pause(); go(step - 1); });
    bNext.addEventListener("click", () => { pause(); go(step + 1); });
    bMain.addEventListener("click", togglePlay);
    bFull.addEventListener("click", fullscreen);
    toggleInput.addEventListener("change", () => setStepMode(toggleInput.checked));

    document.addEventListener("keydown", (e) => {
      if (e.metaKey || e.ctrlKey || e.altKey) return;
      if (root.closest("[hidden]")) return; // another tab's player is showing
      const k = e.key.toLowerCase();
      if (k === " ") { e.preventDefault(); togglePlay(); }
      else if (k === "arrowright") { e.preventDefault(); pause(); go(step + 1); }
      else if (k === "arrowleft") { e.preventDefault(); pause(); go(step - 1); }
      else if (k === "s") setStepMode(!toggleInput.checked);
      else if (k === "r") restart();
      else if (k === "f") fullscreen();
    });

    render(false);

    // Start playing the first time the animation scrolls into view (or its
    // slide is shown), and pause whenever it leaves the screen.
    const autoplay = root.dataset.autoplay !== "false" && !reduceMotion;
    if ("IntersectionObserver" in window) {
      new IntersectionObserver((entries) => {
        for (const entry of entries) {
          if (entry.isIntersecting && !started) {
            started = true;
            // Don't autoplay if the viewer already moved on (e.g. with the keys).
            if (autoplay && !toggleInput.checked && step === 1) play();
          } else if (!entry.isIntersecting && playing) {
            pause();
          }
        }
      }, { threshold: 0.4 }).observe(root);
    }
  }

  function init() { document.querySelectorAll(".stepper").forEach(Stepper); }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
  else init();
})();
