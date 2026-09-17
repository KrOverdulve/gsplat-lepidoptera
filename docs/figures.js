/* ============================================================
   Interactive figures for the Method / Results sections
   - .compare        before/after drag slider (pointer + keyboard)
   - [data-swap]     segmented control swapping a slider's image pair
   - [data-pair]     segmented control swapping a side-by-side pair
   - .passes         step through the three rendering passes
   - #ventral        GT / ours / baseline viewer with thumbnails
   - table.sortable  click a header to sort, click again to reset
   ============================================================ */
(function () {
  "use strict";

  var $$ = function (sel, root) { return Array.prototype.slice.call((root || document).querySelectorAll(sel)); };
  var clamp = function (v, a, b) { return Math.min(b, Math.max(a, v)); };

  function selectIn(group, btn) {
    $$("button", group).forEach(function (b) {
      var on = b === btn;
      b.classList.toggle("on", on);
      if (b.hasAttribute("aria-selected")) b.setAttribute("aria-selected", on ? "true" : "false");
    });
  }

  /* ---------- before/after slider ---------- */
  $$(".compare").forEach(function (el) {
    var pos = parseFloat(el.getAttribute("data-start")) || 50;
    var knob = document.createElement("span");
    knob.className = "knob";
    knob.innerHTML = '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="m9 6-6 6 6 6M15 6l6 6-6 6"/></svg>';
    el.appendChild(knob);
    el.tabIndex = 0;
    el.setAttribute("role", "slider");
    el.setAttribute("aria-label", "Comparison slider");
    el.setAttribute("aria-valuemin", "0");
    el.setAttribute("aria-valuemax", "100");

    var lblL = el.querySelector(".cmp-lbl.l"), lblR = el.querySelector(".cmp-lbl.r");
    function set(p) {
      pos = clamp(p, 0, 100);
      el.style.setProperty("--pos", pos + "%");
      el.setAttribute("aria-valuenow", String(Math.round(pos)));
      if (lblL) lblL.style.opacity = pos < 18 ? "0" : "1";
      if (lblR) lblR.style.opacity = pos > 82 ? "0" : "1";
    }
    function fromEvent(e) {
      var r = el.getBoundingClientRect();
      set(((e.clientX - r.left) / r.width) * 100);
    }
    var dragging = false;
    el.addEventListener("pointerdown", function (e) {
      dragging = true;
      try { el.setPointerCapture(e.pointerId); } catch (err) {}
      fromEvent(e);
    });
    el.addEventListener("pointermove", function (e) {
      if (dragging) fromEvent(e);
    });
    var stop = function () { dragging = false; };
    el.addEventListener("pointerup", stop);
    el.addEventListener("pointercancel", stop);
    el.addEventListener("keydown", function (e) {
      var step = e.shiftKey ? 10 : 2;
      if (e.key === "ArrowLeft") { set(pos - step); e.preventDefault(); }
      else if (e.key === "ArrowRight") { set(pos + step); e.preventDefault(); }
      else if (e.key === "Home") { set(0); e.preventDefault(); }
      else if (e.key === "End") { set(100); e.preventDefault(); }
    });
    set(pos);
  });

  /* ---------- swap image pair inside a slider ---------- */
  $$("[data-swap]").forEach(function (group) {
    var target = document.getElementById(group.getAttribute("data-swap"));
    if (!target) return;
    var a = target.querySelector(".cmp-a"), b = target.querySelector(".cmp-b");
    $$("button", group).forEach(function (btn) {
      btn.addEventListener("click", function () {
        selectIn(group, btn);
        a.src = btn.getAttribute("data-a");
        b.src = btn.getAttribute("data-b");
      });
    });
  });

  /* ---------- swap a side-by-side pair ---------- */
  $$("[data-pair]").forEach(function (group) {
    var target = document.getElementById(group.getAttribute("data-pair"));
    if (!target) return;
    var imgs = $$("img", target);
    $$("button", group).forEach(function (btn) {
      btn.addEventListener("click", function () {
        selectIn(group, btn);
        imgs[0].src = btn.getAttribute("data-a");
        imgs[1].src = btn.getAttribute("data-b");
      });
    });
  });

  /* ---------- rendering passes stepper ---------- */
  $$(".passes").forEach(function (box) {
    var imgs = $$(".passes-stage img", box);
    var group = box.querySelector(".seg");
    $$("button", group).forEach(function (btn) {
      btn.addEventListener("click", function () {
        var i = parseInt(btn.getAttribute("data-i"), 10);
        selectIn(group, btn);
        imgs.forEach(function (im, j) { im.classList.toggle("on", j === i); });
      });
    });
  });

  /* ---------- ventral viewer ---------- */
  var ventral = document.getElementById("ventral");
  if (ventral) {
    var vt = ventral.querySelector(".v-thumbs");
    var panels = {
      gt: ventral.querySelector('img[data-k="gt"]'),
      ours: ventral.querySelector('img[data-k="ours"]'),
      base: ventral.querySelector('img[data-k="base"]')
    };
    $$("button", vt).forEach(function (btn) {
      btn.addEventListener("click", function () {
        var i = btn.getAttribute("data-i");
        selectIn(vt, btn);
        Object.keys(panels).forEach(function (k) {
          panels[k].src = "assets/figs/ventral_" + k + "_" + i + ".webp";
        });
      });
    });
  }

  /* ---------- sortable tables ---------- */
  $$("table.sortable").forEach(function (table) {
    var tbody = table.tBodies[0];
    var heads = $$("thead th", table);
    var active = -1;

    function restore() {
      $$("tr", tbody)
        .sort(function (a, b) { return a.getAttribute("data-o") - b.getAttribute("data-o"); })
        .forEach(function (r) { tbody.appendChild(r); });
    }

    heads.forEach(function (th, col) {
      var kind = th.getAttribute("data-k");
      if (!kind) return;
      th.tabIndex = 0;
      th.title = kind === "num" ? "Sort by this metric (best first)" : "Restore original order";
      function go() {
        heads.forEach(function (h) { h.classList.remove("sorted"); });
        if (kind !== "num" || active === col) { active = -1; restore(); return; }
        active = col;
        th.classList.add("sorted");
        var desc = th.getAttribute("data-dir") === "desc";
        $$("tr", tbody)
          .sort(function (a, b) {
            var x = parseFloat(a.cells[col].textContent), y = parseFloat(b.cells[col].textContent);
            return desc ? y - x : x - y;
          })
          .forEach(function (r) { tbody.appendChild(r); });
      }
      th.addEventListener("click", go);
      th.addEventListener("keydown", function (e) {
        if (e.key === "Enter" || e.key === " ") { e.preventDefault(); go(); }
      });
    });
  });
})();

/* ---------- click-to-load SuperSplat viewer ---------- */
(function () {
  "use strict";
  var box = document.getElementById("splat");
  if (!box) return;
  var btn = box.querySelector(".splat-poster");
  btn.addEventListener("click", function () {
    var f = document.createElement("iframe");
    f.src = box.getAttribute("data-src");
    f.title = "Interactive 3D reconstruction (SuperSplat viewer)";
    f.allow = "fullscreen; xr-spatial-tracking";
    f.setAttribute("allowfullscreen", "");
    box.appendChild(f);
    btn.remove();
    f.focus();
  });
})();
