"""Construcao do site estatico dos relatorios (reports/_site/).

Descobre os relatorios em reports/<categoria>/<nome>/ (report.py com build()
ou report.md/meta.json estatico), renderiza cada pagina com a shell padrao
(topbar com dropdowns por categoria, popup LaTeX, acoes de regenerar) e gera
o hub (home) consolidado em _site/index.html — sempre atualizado a cada build.

CLI:
    python reports/src/builder.py build [--serve [PORTA]]
"""

from __future__ import annotations

import importlib.util
import json
import re
import shutil
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from reports.src import components as C  # noqa: E402
from reports.src import latexfmt  # noqa: E402

REPORTS_DIR = PROJECT_ROOT / "reports"
SITE_DIR = REPORTS_DIR / "_site"
ASSETS_SRC = Path(__file__).resolve().parent / "assets"
HUB_TITLE = "Wind Speed Forecasting — Hub de Relatórios"

SKIP_DIRS = {"src", "_site", "__pycache__"}

# ---------------------------------------------------------------------------
# Estado persistente (reports/state.json) — NAO editar a mao: o servidor e o
# builder gravam aqui criado/regenerado/favorito/titulo de cada relatorio.
# ---------------------------------------------------------------------------

STATE_PATH = REPORTS_DIR / "state.json"
_state: dict = {}
_state_loaded = False


def _load_state() -> None:
    global _state, _state_loaded
    _state_loaded = True
    if not STATE_PATH.is_file():
        _state = {}
        return
    try:
        _state = json.loads(STATE_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        _state = {}


def state_for(key: str) -> dict:
    if not _state_loaded:
        _load_state()
    return _state.setdefault(key, {})


def state_all() -> dict:
    if not _state_loaded:
        _load_state()
    return _state


def state_save() -> None:
    if not _state_loaded:
        _load_state()
    try:
        STATE_PATH.write_text(
            json.dumps(_state, ensure_ascii=False, indent=1, sort_keys=True),
            encoding="utf-8")
    except OSError as error:
        print(f"aviso: nao salvei {STATE_PATH}: {error}", file=sys.stderr)


def state_update(key: str, **fields) -> None:
    """Carimba campos de um relatorio (regenerated, favorite, title...)."""
    state_for(key).update(fields)
    state_save()


def fmt_date(ts) -> str:
    """Carimbo curto pt-BR; None/0 -> em dash."""
    if not ts:
        return "—"
    return time.strftime("%d/%m/%Y %H:%M", time.localtime(ts))


# ---------------------------------------------------------------------------
# Descoberta
# ---------------------------------------------------------------------------

class Report:
    def __init__(self, category: str, name: str, path: Path):
        self.category = category
        self.name = name
        self.path = path              # pasta do relatorio
        self.kind = "md"              # py | md | html
        self.module = None
        self.meta: dict = {}
        self.full_doc = False         # html standalone (documento completo)
        self.key = f"{category}/{name}"
        self.state = state_for(self.key)
        self._load()
        self._ensure_state()

    def _load(self) -> None:
        py = self.path / "report.py"
        md = self.path / "report.md"
        html = self.path / "report.html"
        meta_file = self.path / "meta.json"
        if meta_file.is_file():
            try:
                self.meta = json.loads(
                    meta_file.read_text(encoding="utf-8"))
            except Exception:
                self.meta = {}
        if py.is_file():
            self.kind = "py"
            self._load_module(py)
        elif md.is_file():
            self.kind = "md"
        elif html.is_file():
            self.kind = "html"
            head = html.read_text(encoding="utf-8",
                                  errors="replace")[:64].lstrip().lower()
            self.full_doc = head.startswith("<!doctype") \
                or head.startswith("<html")
        if not self.meta.get("title"):
            if self.kind == "md":
                self.meta["title"] = _first_heading(md) or self.name
            else:
                self.meta["title"] = self.name

    def _ensure_state(self) -> None:
        """Garante created/regenerated no estado (primeira vez que visto)."""
        if "created" in self.state:
            return
        main = self.path / f"report.{self.kind}"
        try:
            self.state["created"] = main.stat().st_mtime
        except OSError:
            self.state["created"] = self.path.stat().st_mtime
        self.state.setdefault("regenerated", None)
        self.state.setdefault("favorite", False)
        state_save()

    def _load_module(self, py: Path) -> None:
        modname = f"reports_gen_{self.category}_{self.name}".replace(
            "-", "_").replace(".", "_")
        try:
            spec = importlib.util.spec_from_file_location(modname, py)
            module = importlib.util.module_from_spec(spec)
            sys.modules[modname] = module
            spec.loader.exec_module(module)
            self.module = module
            self.meta.update(getattr(module, "REPORT", {}))
        except Exception as error:  # noqa: BLE001 - relatorio quebrado nao
            # pode derrubar o site inteiro
            print(f"aviso: falha ao carregar {py}: {error}", file=sys.stderr)
            self.meta["error"] = str(error)

    @property
    def title(self) -> str:
        return self.state.get("title") or self.meta.get("title", self.name)

    @property
    def favorite(self) -> bool:
        return bool(self.state.get("favorite"))

    @property
    def created(self):
        return self.state.get("created")

    @property
    def regenerated(self):
        return self.state.get("regenerated")

    @property
    def description(self) -> str:
        return self.meta.get("description", "")

    @property
    def order(self):
        return self.meta.get("order", self.name)

    def render_body(self) -> str:
        if self.kind == "py" and self.module is not None \
                and hasattr(self.module, "build"):
            ctx = SimpleCtx(self)
            return str(self.module.build(ctx))
        if self.kind == "md":
            return f'<div class="markdown">' \
                   f'{C.markdown_file(self.path / "report.md")}</div>'
        if self.kind == "html":
            return (self.path / "report.html").read_text(
                encoding="utf-8", errors="replace")
        return (f'<section><h2>Indisponível</h2><p>Este relatório não '
                f'pôde ser carregado: {C.esc(self.meta.get("error", "?"))}'
                f'</p></section>')


class SimpleCtx:
    """Contexto passado a build() dos report.py."""

    def __init__(self, report: Report):
        self.report = report
        self.category = report.category
        self.name = report.name
        self.dir = report.path                 # reports/<cat>/<nome>/
        self.root = PROJECT_ROOT
        # caminho para os arquivos do relatorio ja publicados no _site
        self.published = SITE_DIR / report.category / report.name


def discover() -> list[Report]:
    reports: list[Report] = []
    if not REPORTS_DIR.is_dir():
        return reports
    for cat_dir in sorted(p for p in REPORTS_DIR.iterdir() if p.is_dir()):
        if cat_dir.name in SKIP_DIRS:
            continue
        for rep_dir in sorted(p for p in cat_dir.iterdir() if p.is_dir()):
            if rep_dir.name in SKIP_DIRS:
                continue
            if not any((rep_dir / f).is_file() for f in
                       ("report.py", "report.md", "report.html",
                        "meta.json")):
                continue
            reports.append(Report(cat_dir.name, rep_dir.name, rep_dir))
    return reports


def _first_heading(md: Path) -> str | None:
    try:
        for line in md.read_text(encoding="utf-8",
                                 errors="replace").splitlines():
            if line.startswith("# "):
                return C.esc(line[2:].strip())
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Shell / nav / hub
# ---------------------------------------------------------------------------

def build_nav(reports: list[Report], active: str = "") -> str:
    by_cat: dict[str, list[Report]] = {}
    for rep in reports:
        by_cat.setdefault(rep.category, []).append(rep)
    parts = ['<nav class="dropdowns">']
    for cat in sorted(by_cat):
        reps = sorted(by_cat[cat], key=lambda r: str(r.order))
        active_cls = " open" if cat == active.split("/")[0] else ""
        items = ""
        for rep in reps:
            bold = ' style="font-weight:700"' \
                if f"{rep.category}/{rep.name}" == active else ""
            star = (f' <span class="nav-star" style="display:'
                    f'{"inline" if rep.favorite else "none"}">★</span>')
            items += (f'<a href="../../{rep.category}/{rep.name}/"'
                      f' data-path="{rep.category}/{rep.name}"{bold}>'
                      f'<span class="nav-title">{C.esc(rep.title)}</span>'
                      f'{star}</a>')
        parts.append(
            f'<div class="drop{active_cls}">'
            f'<button type="button">{C.esc(cat)} ▾</button>'
            f'<div class="menu"><div class="cat">{C.esc(cat)}</div>'
            f'{items}</div></div>')
    parts.append("</nav>")
    return "".join(parts)


# ---------------------------------------------------------------------------
# Sidebars: TOC (esquerda) e metadados (direita)
# ---------------------------------------------------------------------------

def _truncate(text: str, n: int = 46) -> str:
    text = re.sub(r"<[^>]+>", "", text)
    return text if len(text) <= n else text[: n - 1] + "…"


def build_toc(body: str) -> str:
    """Índice de seções + figuras/tabelas a partir do HTML renderizado."""
    items = []
    pattern = re.compile(
        r'<section id="sec-([^"]+)"><h2>(.*?)</h2>|'
        r'<div class="ins (ins-fig|ins-tab)" id="([^"]+)"[^>]*'
        r'data-caption="([^"]*)"',
        re.S)
    # varre em ordem: seções abrem grupo; inseríveis entram no grupo atual
    events = []
    for m in pattern.finditer(body):
        if m.group(1):
            events.append((m.start(), "sec", m.group(1),
                           _truncate(m.group(2))))
        else:
            kind = "fig" if m.group(3) == "ins-fig" else "tab"
            events.append((m.start(), kind, m.group(4),
                           _truncate(m.group(5) or m.group(4))))
    for _, kind, anchor, text in events:
        if kind == "sec":
            if items:
                items.append("</ul></li>")
            items.append(f'<li class="toc-sec"><a href="#sec-{C.esc(anchor)}">'
                         f'{text}</a><ul class="toc-sub">')
        else:
            icon = "▣" if kind == "tab" else "🖼"
            items.append(f'<li><a href="#{C.esc(anchor)}">{icon} '
                         f'{text}</a></li>')
    if items:
        items.append("</ul></li>")
    if not items:
        return ""
    return ('<div class="toc-title">nesta página</div><ul class="toc">'
            + "".join(items) + "</ul>")


def build_meta_panel(rep: Report) -> str:
    """Painel estático de metadados (o vivo entra via /api/meta)."""
    rows = [("Categoria", rep.category), ("Gerador", f"{rep.kind} report.py"
             if rep.kind == "py" else rep.kind),
            ("Criado", fmt_date(rep.created)),
            ("Regenerado", fmt_date(rep.regenerated)),
            ("Favorito", "★ sim" if rep.favorite else "não")]
    for key, label in (("source_dir", "Fonte"), ("models_dir", "Modelos")):
        if rep.meta.get(key):
            rows.append((label, f"<code>{C.esc(rep.meta[key])}</code>"))
    for key, label in (("regenerate", "Regenerar"), ("retrain", "Retreino")):
        argv = rep.meta.get(key)
        rows.append((label, "configurado" if argv else
                     "— (não configurado)"))
    for label, value in rep.meta.get("info", []):
        rows.append((C.esc(str(label)), str(value)))
    dl = "".join(f"<div class='meta-row'><span>{k}</span>"
                 f"<b>{v}</b></div>" for k, v in rows)
    return (f'<div class="toc-title">metadados</div>'
            f'<div class="meta-list">{dl}</div>'
            f'<div class="meta-live" id="meta-live"></div>')


def page_shell(title: str, body: str, nav_html: str, reports: list[Report],
               active: str = "", rep: Report | None = None) -> str:
    templates = latexfmt.templates_json()
    if rep is not None:
        toc = build_toc(body)
        meta = build_meta_panel(rep)
        inner = (f'<aside class="side side-left">{toc or ""}</aside>'
                 f'<main class="content">{body}</main>'
                 f'<aside class="side side-right">{meta}</aside>')
        layout = f'<div class="layout">{inner}</div>'
        catname = f'{rep.category}/{rep.name}'
    else:
        layout = f'<div class="layout hub-layout">{body}</div>'
        catname = ""
    return f"""<!doctype html>
<html lang="pt-BR"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{C.esc(title)}</title>
<link rel="stylesheet" href="../../assets/theme.css">
<script id="latex-templates" type="application/json">{templates}</script>
</head><body>
<div class="topbar">
  <a class="brand" href="../../">⚡ Wind Forecast<small>hub de relatórios</small></a>
  {nav_html}
  <span class="spacer"></span>
  <a class="hublink" href="../../">⌂ hub</a>
</div>
<div class="container{' wide' if rep is not None else ''}">
{layout}
</div>
<footer class="foot">gerado por reports/src/builder.py · {time.strftime('%d/%m/%Y %H:%M')}</footer>
<script src="../../assets/app.js"></script>
<script>window.REPORT_PATH = "{catname}";window.REPORT_TITLE = {json.dumps(rep.title, ensure_ascii=False) if rep is not None else '""'};</script>
</body></html>"""


def last_modified(report: "Report") -> float:
    """Mtime mais recente dos arquivos-fonte do relatório.

    Varre report.* / meta.json / figures (ignora __pycache__ e o site
    gerado) — captura regenerações que reescrevem figuras, coisa que o
    mtime da pasta sozinho não vê.
    """
    newest = 0.0
    try:
        for item in report.path.rglob("*"):
            if item.is_dir() or "__pycache__" in item.parts:
                continue
            newest = max(newest, item.stat().st_mtime)
    except OSError:
        pass
    return newest or report.path.stat().st_mtime


def rel_time(ts: float) -> str:
    """Carimbo relativo compacto: 'hoje 14:32', 'ontem', 'há 3 dias'."""
    import datetime

    dt = datetime.datetime.fromtimestamp(ts)
    today = datetime.date.today()
    days = (today - dt.date()).days
    if days == 0:
        return f"hoje {dt:%H:%M}"
    if days == 1:
        return f"ontem {dt:%H:%M}"
    if 1 < days <= 30:
        return f"há {days} dias"
    return f"{dt:%d/%m/%Y}"


def hub_page(reports: list[Report]) -> str:
    by_cat: dict[str, list[Report]] = {}
    for rep in reports:
        by_cat.setdefault(rep.category, []).append(rep)
    body = [f'<div class="hub-hero"><h1>{HUB_TITLE}</h1>'
            f'<p>Relatórios padronizados do projeto (LiDAR, treinamento, '
            f'resultados, imputação, protocolo). Todo inserível tem botão '
            f'<strong>LaTeX</strong> para copiar nos formatos '
            f'<kbd>steam</kbd> / <kbd>sbsr</kbd> / <kbd>genérico</kbd>. '
            f'Para os botões de regenerar/retreinar funcionarem, rode '
            f'<kbd>python reports/src/server.py</kbd>.</p></div>']

    # Favoritos (assado no build; o app.js re-renderiza em tempo real)
    favs = sorted((r for r in reports if r.favorite), key=lambda r: r.title)
    fav_items = "".join(
        f'<li><a href="{rep.category}/{rep.name}/">{C.esc(rep.title)}</a>'
        f'<span class="cat">{C.esc(rep.category)}</span></li>'
        for rep in favs)
    body.append(
        '<section class="hub-favs" id="hub-favs"'
        + ("" if favs else ' hidden') + '><h2>★ Favoritos</h2>'
        f'<ul class="fav-list">{fav_items}</ul></section>')

    recent = sorted(reports, key=last_modified, reverse=True)[:6]
    recent_items = "".join(
        f'<li><a href="{rep.category}/{rep.name}/">{C.esc(rep.title)}</a>'
        f'<span class="when">{C.esc(rel_time(last_modified(rep)))}</span>'
        f'<span class="cat">{C.esc(rep.category)}</span></li>'
        for rep in recent)
    body.append(
        '<section class="hub-recent"><h2>Últimas atualizações</h2>'
        f'<ul class="recent-list">{recent_items}</ul></section>')
    cat_desc = {
        "treinamento": "Treino dos modelos sob o protocolo de reamostragem",
        "lidar": "Resultados por protocolo de split (serie LiDAR)",
        "resultados": "Relatórios de resultados e estudos de modelos",
        "imputacao": "Preenchimento de lacunas e benchmarks WFO",
        "protocolo": "Protocolos, vazamentos e roteiros metodológicos",
    }
    for cat in sorted(by_cat):
        reps = sorted(by_cat[cat], key=lambda r: str(r.order))
        body.append(f'<h2 class="hub-cat">{C.esc(cat)}'
                    + (f' <small style="text-transform:none;letter-spacing:0">'
                       f'— {C.esc(cat_desc[cat])}</small>' if cat in cat_desc
                       else "") + "</h2>")
        cards = []
        for rep in reps:
            dates = (f'criado {fmt_date(rep.created)[:10]}'
                     f' · regenerado {fmt_date(rep.regenerated)[:10]}'
                     f' · atualizado {C.esc(rel_time(last_modified(rep)))}')
            actions = [f'<a class="btn" href="../../{rep.category}/'
                       f'{rep.name}/">abrir</a>',
                       f'<button type="button" class="btn act-fav" '
                       f'data-path="{rep.key}">'
                       f'{"★" if rep.favorite else "☆"} favorito</button>',
                       f'<button type="button" class="btn act-rename" '
                       f'data-path="{rep.key}">✏ renomear</button>']
            if rep.kind == "py":
                actions.append('<span class="meta">regenerar pela página '
                               '(servidor ativo)</span>')
            error = f'<div class="meta" style="color:#b03030">erro: ' \
                    f'{C.esc(rep.meta.get("error", ""))}</div>' \
                if rep.meta.get("error") else ""
            fav_cls = " fav" if rep.favorite else ""
            cards.append(
                f'<div class="hub-card{fav_cls}" data-path="{rep.key}">'
                f'<h3><a href="../../{rep.category}/{rep.name}/" '
                f'class="card-title">{C.esc(rep.title)}</a></h3>'
                f'<div class="desc">{C.esc(rep.description)}</div>{error}'
                f'<div class="meta">{rep.category}/{rep.name} · {dates}</div>'
                f'<div class="actions">{"".join(actions)}</div></div>')
        body.append(f'<div class="hub-grid">{"".join(cards)}</div>')
    body_html = "\n".join(body)
    templates = latexfmt.templates_json()
    return f"""<!doctype html>
<html lang="pt-BR"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{HUB_TITLE}</title>
<link rel="stylesheet" href="assets/theme.css">
<script id="latex-templates" type="application/json">{templates}</script>
</head><body>
<div class="topbar">
  <span class="brand">⚡ Wind Forecast<small>hub de relatórios</small></span>
  <span class="spacer"></span>
  <button class="btn act-rebuild-hub" type="button">⟳ Atualizar hub</button>
</div>
<div class="container">
{body_html}
</div>
<footer class="foot">gerado em {time.strftime('%d/%m/%Y %H:%M')} · sempre atualizado a cada build · <code>python reports/src/server.py</code></footer>
<script src="assets/app.js"></script>
</body></html>"""


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

def _copy_report_files(rep: Report, dest: Path):
    """Copia tudo que a pagina usa (figuras, csv...) para o _site."""
    dest.mkdir(parents=True, exist_ok=True)
    for item in rep.path.iterdir():
        if item.name in ("report.py", "report.md", "report.html",
                         "meta.json", "retrain.log", "job.json",
                         "regenerate.log", "__pycache__"):
            continue
        target = dest / item.name
        try:
            if item.is_dir():
                if target.exists():
                    shutil.rmtree(target)
                shutil.copytree(item, target)
            else:
                shutil.copy2(item, target)
        except Exception as error:  # noqa: BLE001
            print(f"aviso: nao copiei {item}: {error}", file=sys.stderr)


def build_site() -> list[Report]:
    reports = discover()
    if SITE_DIR.exists():
        shutil.rmtree(SITE_DIR)
    SITE_DIR.mkdir(parents=True)
    shutil.copytree(ASSETS_SRC, SITE_DIR / "assets", dirs_exist_ok=True)
    nav = build_nav(reports)
    for rep in reports:
        dest = SITE_DIR / rep.category / rep.name
        # renderiza ANTES: build() pode gerar/copiar figuras novas na pasta
        body = rep.render_body()
        _copy_report_files(rep, dest)
        if rep.kind == "html" and rep.full_doc:
            # documento standalone (relatorios auto-contidos migrados de
            # data/results): publica como veio, sem a shell do hub
            html = (rep.path / "report.html").read_text(
                encoding="utf-8", errors="replace")
        else:
            html = page_shell(rep.title, body, nav, reports,
                              active=f"{rep.category}/{rep.name}", rep=rep)
        (dest / "index.html").write_text(html, encoding="utf-8")
        print(f"  ok: {rep.category}/{rep.name} ({rep.kind})")
    (SITE_DIR / "index.html").write_text(hub_page(reports),
                                         encoding="utf-8")
    print(f"hub: {SITE_DIR / 'index.html'} ({len(reports)} relatórios)")
    return reports


def run_standalone(report_path: str | Path):
    """Chamado por `python reports/<cat>/<nome>/report.py`."""
    build_site()


def main() -> int:
    cmd = sys.argv[1] if len(sys.argv) > 1 else "build"
    if cmd == "build":
        build_site()
        return 0
    if cmd == "serve":
        from reports.src import server
        port = int(sys.argv[2]) if len(sys.argv) > 2 else 8765
        return server.serve(port)
    sys.exit(__doc__)


if __name__ == "__main__":
    sys.exit(main())
