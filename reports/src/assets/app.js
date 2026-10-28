/* Hub de relatórios — popup LaTeX, nav dropdown, regenerar/retreinar. */
(function () {
  "use strict";

  var TEMPLATES = {};
  var dlg, el = {};

  // ---------------- templates embutidos (fonte: reports/src/latexfmt.py) ----
  function loadTemplates() {
    var tag = document.getElementById("latex-templates");
    if (tag) { try { TEMPLATES = JSON.parse(tag.textContent); } catch (e) {} }
  }

  // ---------------- popup LaTeX ----------------
  function ensureDialog() {
    if (dlg) return;
    dlg = document.createElement("dialog");
    dlg.className = "latex-dlg";
    dlg.innerHTML =
      '<div class="dlg-head"><h3>Copiar para LaTeX</h3>' +
      '<button class="close" type="button">✕ fechar</button></div>' +
      '<div class="dlg-body">' +
      '<div class="fmt-tabs"></div>' +
      '<div class="dlg-fields">' +
      '  <div><label>\\label (prefixado fig:/tab:)</label>' +
      '    <input class="mono f-label" type="text"></div>' +
      '  <div><label>Legenda (\\caption)</label>' +
      '    <input class="f-caption" type="text"></div>' +
      '  <div class="f-file-wrap"><label>Arquivo (\\includegraphics)</label>' +
      '    <input class="mono f-file" type="text"></div>' +
      '  <div class="f-width-wrap"><label>Largura (fração de \\textwidth/\\linewidth)</label>' +
      '    <input class="f-width" type="text" inputmode="decimal"></div>' +
      '  <div><label>Posição flutuante</label>' +
      '    <select class="f-place"><option>h</option><option>h!</option>' +
      '    <option>t</option><option>t!</option><option>htbp</option>' +
      '    <option>p</option></select></div>' +
      '</div>' +
      '<div class="snippet"><button class="copy" type="button">Copiar</button>' +
      '<pre></pre></div>' +
      '<p style="font-size:11px;color:#888;margin:8px 0 0">' +
      'Cole no preâmbulo do artigo os pacotes necessários. ' +
      'O \\label já segue a convenção fig:/tab:.</p></div>';
    document.body.appendChild(dlg);
    el.tabs = dlg.querySelector(".fmt-tabs");
    el.label = dlg.querySelector(".f-label");
    el.caption = dlg.querySelector(".f-caption");
    el.file = dlg.querySelector(".f-file");
    el.fileWrap = dlg.querySelector(".f-file-wrap");
    el.width = dlg.querySelector(".f-width");
    el.widthWrap = dlg.querySelector(".f-width-wrap");
    el.place = dlg.querySelector(".f-place");
    el.pre = dlg.querySelector("pre");
    el.copy = dlg.querySelector(".copy");
    dlg.querySelector(".close").addEventListener("click", function () {
      dlg.close();
    });
    el.copy.addEventListener("click", function () { copySnippet(); });
    ["input", "change"].forEach(function (ev) {
      dlg.querySelector(".dlg-fields").addEventListener(ev, refresh);
    });
  }

  var current = null; // {kind, format, ins}

  function openPopup(ins) {
    ensureDialog();
    loadTemplates();
    current = { ins: ins, kind: ins.dataset.kind,
                formats: (ins.dataset.formats || "generic").split("|"),
                format: null };
    var fmt = current.formats[0];
    current.format = TEMPLATES[fmt] ? fmt : Object.keys(TEMPLATES)[0];
    el.tabs.innerHTML = "";
    current.formats.forEach(function (f) {
      if (!TEMPLATES[f]) return;
      var b = document.createElement("button");
      b.type = "button";
      b.textContent = TEMPLATES[f].name || f;
      b.dataset.fmt = f;
      b.addEventListener("click", function () {
        current.format = f;
        markTab(); refresh();
      });
      el.tabs.appendChild(b);
    });
    markTab();
    el.label.value = ins.dataset.label || "";
    el.caption.value = ins.dataset.caption || "";
    el.file.value = ins.dataset.file || "";
    el.width.value = ins.dataset.width || "0.8";
    el.place.value = ins.dataset.placement || "h";
    var isFig = current.kind === "figure";
    el.fileWrap.style.display = isFig ? "" : "none";
    el.widthWrap.style.display = isFig ? "" : "none";
    refresh();
    dlg.showModal();
  }

  function markTab() {
    el.tabs.querySelectorAll("button").forEach(function (b) {
      b.classList.toggle("active", b.dataset.fmt === current.format);
    });
  }

  function snippetText() {
    var t = TEMPLATES[current.format];
    if (!t) return "% templates não disponíveis";
    var tpl = t[current.kind] || "";
    var label = el.label.value.trim();
    var prefix = current.kind === "figure" ? "fig:" : "tab:";
    if (label && label.indexOf(prefix) !== 0) label = prefix + label;
    var body = {};
    try { body = JSON.parse(current.ins.dataset.body || "{}"); } catch (e) {}
    var width = String(el.width.value || "0.8").replace(",", ".");
    var vals = {
      file: el.file.value, caption: el.caption.value, label: label,
      width: width, placement: el.place.value,
      body: body[current.format] || ""
    };
    return tpl.replace(/\{(\w+)\}/g, function (m, k) {
      return (vals[k] !== undefined ? vals[k] : m);
    });
  }

  function refresh() {
    if (!dlg || !dlg.open) return;
    el.pre.textContent = snippetText();
  }

  function copySnippet() {
    var text = snippetText();
    function done() {
      el.copy.textContent = "Copiado ✓";
      setTimeout(function () { el.copy.textContent = "Copiar"; }, 1600);
    }
    if (navigator.clipboard && navigator.clipboard.writeText) {
      navigator.clipboard.writeText(text).then(done, fallback);
    } else { fallback(); }
    function fallback() {
      var ta = document.createElement("textarea");
      ta.value = text; document.body.appendChild(ta); ta.select();
      try { document.execCommand("copy"); done(); } catch (e) {}
      document.body.removeChild(ta);
    }
  }

  // ---------------- nav dropdown ----------------
  function wireNav() {
    document.querySelectorAll(".drop > button").forEach(function (btn) {
      btn.addEventListener("click", function (ev) {
        ev.stopPropagation();
        var d = btn.parentElement;
        var was = d.classList.contains("open");
        document.querySelectorAll(".drop.open").forEach(function (o) {
          o.classList.remove("open");
        });
        d.classList.toggle("open", !was);
      });
    });
    document.addEventListener("click", function () {
      document.querySelectorAll(".drop.open").forEach(function (o) {
        o.classList.remove("open");
      });
    });
  }

  // ---------------- acoes regenerar / retreinar / rebuild ----------------
  function pagePath() {
    var parts = location.pathname.replace(/\/index\.html$/, "")
      .split("/").filter(Boolean);
    return parts.slice(-2).join("/"); // <categoria>/<nome>
  }

  function setStatus(msg, cls) {
    var s = document.querySelector(".job-status");
    if (!s) return;
    s.textContent = msg || "";
    s.className = "job-status " + (cls || "");
  }

  function pollJob(catname, btn, statusSel) {
    var n = 0;
    var timer = setInterval(function () {
      fetch("/api/job/" + catname).then(function (r) { return r.json(); })
        .then(function (st) {
          if (st.running) {
            setStatus("executando… (" + (st.secs | 0) + "s)", "");
          } else {
            clearInterval(timer);
            btn.disabled = false;
            if (st.ok) {
              setStatus("concluído em " + (st.secs | 0) + "s — recarregando…",
                        "ok");
              setTimeout(function () { location.reload(); }, 900);
            } else {
              setStatus("falhou — veja o log na pasta do relatório", "err");
            }
          }
        }).catch(function () {
          clearInterval(timer); btn.disabled = false;
          setStatus("servidor indisponível", "err");
        });
      if (++n > 60 * 240) { clearInterval(timer); btn.disabled = false; }
    }, 1000);
  }

  function post(catname, action, btn) {
    btn.disabled = true;
    setStatus("enviando…", "");
    fetch("/api/" + action + "/" + catname, { method: "POST" })
      .then(function (r) { return r.json().then(function (j) {
        return { ok: r.ok, json: j };
      }); })
      .then(function (res) {
        if (!res.ok) {
          btn.disabled = false;
          setStatus(res.json && res.json.error || "erro", "err");
          return;
        }
        if (action === "rebuild") {
          setStatus("reconstruído — recarregando…", "ok");
          setTimeout(function () { location.reload(); }, 600);
          return;
        }
        if (res.json.async) { pollJob(catname, btn); }
        else { pollJob(catname, btn); }
      })
      .catch(function () {
        btn.disabled = false;
        setStatus("sem servidor — rode: python reports/src/server.py", "err");
      });
  }

  function wireActions() {
    var catname = pagePath();
    var regen = document.querySelector(".act-regen");
    var retrain = document.querySelector(".act-retrain");
    var rebuild = document.querySelector(".act-rebuild");
    if (regen) regen.addEventListener("click", function () {
      post(catname, "regenerate", regen);
    });
    if (retrain) {
      if (retrain.dataset.enabled !== "1") {
        retrain.title = "Retreino não configurado para este relatório";
        retrain.disabled = true;
      } else {
        retrain.addEventListener("click", function () {
          if (!confirm("Retreinar pode demorar horas (GPU). Continuar?")) return;
          post(catname, "retrain", retrain);
        });
      }
    }
    if (rebuild) rebuild.addEventListener("click", function () {
      post(catname, "rebuild", rebuild);
    });
    var hubRebuild = document.querySelector(".act-rebuild-hub");
    if (hubRebuild) hubRebuild.addEventListener("click", function () {
      post("_", "rebuild", hubRebuild);
    });
  }

  // ---------------- lightbox (ampliar imagem) ----------------
  var lb = null;

  function ensureLightbox() {
    if (lb) return;
    lb = document.createElement("dialog");
    lb.className = "lightbox";
    lb.innerHTML = '<img alt="visualização ampliada">';
    document.body.appendChild(lb);
    // fecha quando o clique cai FORA da imagem (fundo/área do dialog)
    lb.addEventListener("click", function (ev) {
      if (ev.target !== lb.querySelector("img")) lb.close();
    });
  }

  function openLightbox(img) {
    ensureLightbox();
    var big = lb.querySelector("img");
    big.src = img.currentSrc || img.src;
    if (typeof lb.showModal === "function") lb.showModal();
    else lb.setAttribute("open", "");
  }

  function wireLightbox() {
    document.querySelectorAll(".ins-fig img").forEach(function (img) {
      img.addEventListener("click", function () { openLightbox(img); });
    });
  }

  // ---------------- sidebar: metadados vivos ----------------
  function esc(s) {
    return String(s == null ? "" : s)
      .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  }

  function renderMeta(data) {
    var box = document.getElementById("meta-live");
    if (!box) return;
    var job = data.job || {};
    var html = '<div class="toc-title">execução</div>';
    var state, cls;
    if (job.running) {
      state = "▶ rodando (" + (job.secs | 0) + "s)";
      cls = "run";
    } else if (job.ok === true) {
      state = "✓ último job ok";
      cls = "ok";
    } else if (job.ok === false) {
      state = "✗ último job falhou";
      cls = "err";
    } else {
      state = "• ocioso";
      cls = "idle";
    }
    html += '<div class="chip ' + cls + '">' + state + "</div>";
    html += "<dl class='meta-rows'>";
    html += "<div><dt>criado</dt><dd>" + esc(fmtTs(data.created)) + "</dd></div>";
    html += "<div><dt>regenerado</dt><dd>" + esc(fmtTs(data.regenerated)) +
      "</dd></div>";
    html += "<div><dt>favorito</dt><dd>" +
      (data.favorite ? "★ sim" : "não") + "</dd></div>";
    html += "<div><dt>gerado</dt><dd>" + esc(data.generated || "—") + "</dd></div>";
    html += "<div><dt>início do job</dt><dd>" +
      (job.ts ? new Date(job.ts * 1000).toLocaleString("pt-BR") : "—") +
      "</dd></div>";
    html += "<div><dt>regenerar</dt><dd>" +
      (data.has_regenerate ? "sim" : "não configurado") + "</dd></div>";
    html += "<div><dt>retreino</dt><dd>" +
      (data.has_retrain ? "sim" : "não configurado") + "</dd></div>";
    (data.info || []).forEach(function (par) {
      html += "<div><dt>" + esc(par[0]) + "</dt><dd>" + esc(par[1]) +
        "</dd></div>";
    });
    html += "</dl>";
    if (data.log_tail && data.log_tail.length) {
      html += '<div class="toc-title">progresso (log)</div><pre class="log">' +
        esc(data.log_tail.join("\n")) + "</pre>";
    }
    box.innerHTML = html;
  }

  function pollMeta() {
    if (!window.REPORT_PATH) return;
    var timer = null;
    function refresh() {
      fetch("/api/meta/" + window.REPORT_PATH)
        .then(function (r) { return r.ok ? r.json() : null; })
        .then(function (d) { if (d) renderMeta(d); })
        .catch(function () {});
    }
    refresh();
    timer = setInterval(refresh, 5000);
  }

  // ---------------- sidebar: TOC ativa por scroll ----------------
  function wireToc() {
    var links = Array.prototype.slice.call(
      document.querySelectorAll(".side-left a[href^='#']"));
    if (!links.length) return;
    var map = {};
    links.forEach(function (a) {
      var id = a.getAttribute("href").slice(1);
      var el = document.getElementById(id);
      if (el) map[id] = a;
    });
    var ids = Object.keys(map);
    function onScroll() {
      var current = ids[0];
      ids.forEach(function (id) {
        if (document.getElementById(id).getBoundingClientRect().top <= 120) {
          current = id;
        }
      });
      links.forEach(function (a) { a.classList.remove("active"); });
      if (map[current]) map[current].classList.add("active");
    }
    document.addEventListener("scroll", onScroll, { passive: true });
    onScroll();
  }

  // ---------------- estado: favoritos, renomear, datas ----------------
  var LS_KEY = "hubstate.v1";
  var LOCAL = {};     // {path: {title?, favorite?}} — fallback sem servidor
  var PATHS = {};     // {path: {title, href, cat}} — índice dos relatórios
  var SERVER_OK = false;

  function loadLocal() {
    try { LOCAL = JSON.parse(localStorage.getItem(LS_KEY) || "{}") || {}; }
    catch (e) { LOCAL = {}; }
  }
  function saveLocal() {
    try { localStorage.setItem(LS_KEY, JSON.stringify(LOCAL)); } catch (e) {}
  }
  function entry(path) {
    if (!LOCAL[path]) LOCAL[path] = {};
    return LOCAL[path];
  }
  function fmtTs(ts) {
    if (!ts) return "—";
    var d = new Date(ts * 1000), p = function (n) {
      return (n < 10 ? "0" : "") + n;
    };
    return p(d.getDate()) + "/" + p(d.getMonth() + 1) + "/" + d.getFullYear() +
      " " + p(d.getHours()) + ":" + p(d.getMinutes());
  }

  function syncServer() {
    fetch("/api/state").then(function (r) { return r.ok ? r.json() : null; })
      .then(function (d) {
        if (!d || !d.reports) return;
        SERVER_OK = true;
        Object.keys(d.reports).forEach(function (path) {
          var st = d.reports[path] || {};
          var e = entry(path);
          if (typeof st.title === "string" && st.title) e.title = st.title;
          e.favorite = !!st.favorite;
          if (st.created) e.created = st.created;
          if (st.regenerated) e.regenerated = st.regenerated;
        });
        saveLocal();
        applyState();
      }).catch(function () { /* sem servidor: fica o localStorage */ });
  }

  function postState(path, action, body) {
    return fetch("/api/" + action + "/" + path, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body || {})
    }).then(function (r) { return r.json().then(function (j) {
      return { ok: r.ok, json: j };
    }); });
  }

  function displayTitle(path, fallback) {
    var e = LOCAL[path];
    return (e && e.title) || fallback || path;
  }

  function applyState() {
    // nav: estrela + título nos dropdowns
    document.querySelectorAll(".drop .menu a[data-path]").forEach(function (a) {
      var path = a.dataset.path, e = LOCAL[path] || {};
      var t = a.querySelector(".nav-title");
      if (t && e.title) t.textContent = e.title;
      var star = a.querySelector(".nav-star");
      if (star) star.style.display = e.favorite ? "inline" : "none";
    });
    // hub: cartões
    document.querySelectorAll(".hub-card[data-path]").forEach(function (card) {
      var path = card.dataset.path, e = LOCAL[path] || {};
      var link = card.querySelector(".card-title");
      var fallback = PATHS[path] && PATHS[path].title;
      PATHS[path] = {
        title: e.title || (link ? link.textContent : fallback || path),
        href: link ? link.getAttribute("href") : "../../" + path + "/",
        cat: path.split("/")[0]
      };
      if (link && e.title) link.textContent = e.title;
      card.classList.toggle("fav", !!e.favorite);
      var fav = card.querySelector(".act-fav");
      if (fav) fav.textContent = e.favorite ? "★ favorito" : "☆ favorito";
    });
    renderFavs();
  }

  function renderFavs() {
    var box = document.getElementById("hub-favs");
    if (!box) return;
    var list = box.querySelector(".fav-list");
    var favs = Object.keys(PATHS).filter(function (p) {
      return LOCAL[p] && LOCAL[p].favorite;
    }).sort(function (a, b) {
      return displayTitle(a).localeCompare(displayTitle(b), "pt-BR");
    });
    box.hidden = !favs.length;
    list.innerHTML = favs.map(function (p) {
      return '<li><a href="' + PATHS[p].href + '">' +
        esc(displayTitle(p)) + "</a>" +
        '<span class="cat">' + esc(PATHS[p].cat) + "</span></li>";
    }).join("");
  }

  function toggleFav(path, btn) {
    var e = entry(path);
    e.favorite = !e.favorite;
    saveLocal();
    applyState();
    if (btn) btn.textContent = e.favorite ? "★ favorito" : "☆ favorito";
    postState(path, "favorite", { favorite: e.favorite })
      .then(function (res) {
        if (res.ok && typeof res.json.favorite === "boolean") {
          e.favorite = res.json.favorite; saveLocal(); applyState();
        }
      }).catch(function () { /* sem servidor: só localStorage */ });
  }

  function renameReport(path, btn) {
    var current = displayTitle(path, (window.REPORT_TITLE || ""));
    var title = prompt("Novo título do relatório:", current);
    if (title === null) return;
    title = title.trim();
    if (title === current) return;
    if (title) entry(path).title = title;
    else delete entry(path).title;
    saveLocal();
    applyState();
    applyPageTitle(title);
    postState(path, "rename", { title: title }).then(function (res) {
      if (res.ok && res.json.title) {
        entry(path).title = res.json.title; saveLocal(); applyState();
        applyPageTitle(res.json.title);
      }
    }).catch(function () { /* sem servidor: só localStorage */ });
  }

  function applyPageTitle(title) {
    if (!title) return;
    document.title = title + " — Wind Forecast";
    var h1 = document.querySelector(".page-head h1");
    if (h1) h1.textContent = title;
  }

  function wireStateActions() {
    document.addEventListener("click", function (ev) {
      var fav = ev.target.closest(".act-fav");
      if (fav && fav.dataset.path) { toggleFav(fav.dataset.path, fav); return; }
      var ren = ev.target.closest(".act-rename");
      if (ren && ren.dataset.path) renameReport(ren.dataset.path, ren);
    });
    // botões no topo das páginas de relatório
    if (window.REPORT_PATH) {
      var hublink = document.querySelector(".topbar .hublink");
      var path = window.REPORT_PATH;
      if (hublink) {
        var fav = document.createElement("button");
        fav.type = "button"; fav.className = "btn act-fav";
        fav.dataset.path = path; fav.textContent = "☆ favorito";
        var ren = document.createElement("button");
        ren.type = "button"; ren.className = "btn act-rename";
        ren.dataset.path = path; ren.textContent = "✏ renomear";
        hublink.parentElement.insertBefore(fav, hublink);
        hublink.parentElement.insertBefore(ren, hublink);
      }
      var e = LOCAL[path];
      if (e && e.favorite && document.querySelector(".act-fav"))
        document.querySelector(".act-fav").textContent = "★ favorito";
    }
  }

  // ---------------- init ----------------
  document.addEventListener("DOMContentLoaded", function () {
    loadTemplates();
    loadLocal();
    // índice de relatórios (títulos assados no build) para a lista de favoritos
    document.querySelectorAll(".hub-card[data-path]").forEach(function (card) {
      var link = card.querySelector(".card-title");
      PATHS[card.dataset.path] = {
        title: link ? link.textContent : card.dataset.path,
        href: link ? link.getAttribute("href") : "../../" + card.dataset.path + "/",
        cat: card.dataset.path.split("/")[0]
      };
    });
    document.querySelectorAll(".ins .btn-latex").forEach(function (btn) {
      btn.addEventListener("click", function () {
        openPopup(btn.closest(".ins"));
      });
    });
    wireLightbox();
    wireNav();
    wireActions();
    wireToc();
    wireStateActions();
    applyState();
    pollMeta();
    syncServer();
  });
})();
