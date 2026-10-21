"""Componentes HTML padrao dos relatorios (reports/src).

Todo inserivel reutilizavel em artigo/slides (figura, tabela) e gerado com
data-attributes que alimentam o popup "Copiar LaTeX" (reports/src/assets/
app.js): formatos steam/sbsr/generic, label com prefixo fig:/tab:, caption
e largura editaveis no popup.

Uso tipico dentro de reports/<categoria>/<nome>/report.py:

    from reports.src import components as C

    def build(ctx):
        parts = [C.page_header(...), C.section(...,
                 C.figure("figures/x.png", "Legenda...", label="fig:x"))]
        return "".join(parts)
"""

from __future__ import annotations

import html
import json
from pathlib import Path

import markdown as md_lib

from . import latexfmt


def esc(text) -> str:
    return html.escape(str(text), quote=True)


def _json_attr(obj) -> str:
    return esc(json.dumps(obj, ensure_ascii=False))


# ---------------------------------------------------------------------------
# Inseriveis com popup LaTeX
# ---------------------------------------------------------------------------

def figure(src: str, caption: str, *, label: str | None = None,
           width: float = 0.8, placement: str = "t",
           formats=latexfmt.DEFAULT_FORMATS, alt: str = "") -> str:
    """Figura com botao de exportacao LaTeX. `src` e relativo a pagina."""
    label = label or f"fig:{latexfmt.slug(Path(src).stem)}"
    anchor = label.replace(":", "-")
    return (
        f'<div class="ins ins-fig" id="{esc(anchor)}" data-kind="figure" '
        f'data-file="{esc(src)}" data-caption="{esc(caption)}" '
        f'data-label="{esc(label)}" data-width="{width:g}" '
        f'data-placement="{esc(placement)}" '
        f'data-formats="{esc("|".join(formats))}" data-body="{{}}">'
        f'<figure><img src="{esc(src)}" alt="{esc(alt or caption)}">'
        f'<figcaption>{esc(caption)} '
        f'<code class="lbl">{esc(label)}</code></figcaption></figure>'
        f'<button type="button" class="btn-latex">LaTeX</button></div>'
    )


def table(columns, rows, caption: str, *, label: str | None = None,
          align=None, formats=latexfmt.DEFAULT_FORMATS,
          placement: str = "h", highlight_rows=(), note: str = "",
          first_col_bold: bool = True) -> str:
    """Tabela HTML + corpo tabular pre-renderizado por formato (data-body)."""
    label = label or f"tab:{latexfmt.slug(caption)}"
    anchor = label.replace(":", "-")
    bodies = {fmt: latexfmt.tabular_body(columns, rows, fmt, align)
              for fmt in formats}
    head = "".join(f"<th>{esc(c)}</th>" for c in columns)
    body_rows = []
    for i, row in enumerate(rows):
        cls = ' class="hl"' if i in highlight_rows else ""
        cells = []
        for j, cell in enumerate(row):
            cell_html = esc(cell)
            if j == 0 and first_col_bold:
                cell_html = f"<strong>{cell_html}</strong>"
            cells.append(f"<td>{cell_html}</td>")
        body_rows.append(f"<tr{cls}>{''.join(cells)}</tr>")
    note_html = f'<p class="tab-note">{esc(note)}</p>' if note else ""
    return (
        f'<div class="ins ins-tab" id="{esc(anchor)}" data-kind="table" '
        f'data-caption="{esc(caption)}" data-label="{esc(label)}" '
        f'data-placement="{esc(placement)}" '
        f'data-formats="{esc("|".join(formats))}" '
        f'data-body="{_json_attr(bodies)}">'
        f'<table><thead><tr>{head}</tr></thead>'
        f'<tbody>{"".join(body_rows)}</tbody></table>{note_html}'
        f'<button type="button" class="btn-latex">LaTeX</button></div>'
    )


def df_table(df, caption: str, **kwargs) -> str:
    """Conveniencia: DataFrame -> table()."""
    return table(list(df.columns), df.astype(object).values.tolist(),
                 caption, **kwargs)


# ---------------------------------------------------------------------------
# Blocos simples (sem export LaTeX)
# ---------------------------------------------------------------------------

def cards(items) -> str:
    """Cartoes de metrica: [(valor, rotulo), ...]"""
    return '<div class="cards">' + "".join(
        f'<div class="card"><div class="num">{esc(v)}</div>'
        f'<div class="lab">{esc(l)}</div></div>' for v, l in items) + "</div>"


def section(title: str, body: str, *, sub: str = "", sec_id: str = "",
            sub_after: bool = False) -> str:
    if not sec_id:
        sec_id = latexfmt.slug(title)
    anchor = f' id="sec-{esc(sec_id)}"' if sec_id else ""
    sub_html = f'<div class="sub">{sub}</div>' if sub else ""
    if sub_after:
        return (f"<section{anchor}><h2>{esc(title)}</h2>"
                f"{body}{sub_html}</section>")
    return (f"<section{anchor}><h2>{esc(title)}</h2>{sub_html}"
            f"{body}</section>")


def grid(html_items: str) -> str:
    return f'<div class="fig-grid">{html_items}</div>'


def note(text: str) -> str:
    return f'<p class="note">{text}</p>'


def page_header(title: str, subtitle: str = "", badges=()) -> str:
    badge_html = "".join(f'<span class="badge">{esc(b)}</span>'
                         for b in badges)
    sub_html = f"<p>{esc(subtitle)}</p>" if subtitle else ""
    return (f'<div class="page-head"><h1>{esc(title)}</h1>{sub_html}'
            f'<div class="badges">{badge_html}</div>'
            f'<div class="page-actions">'
            f'<button type="button" class="btn act-regen"'
            f' title="Reprocessa dados e reconstrói esta página">'
            f'↻ Regenerar</button>'
            f'<button type="button" class="btn act-retrain"'
            f' title="Retreina os modelos e depois regenera">'
            f'▶ Regenerar com retreino</button>'
            f'<button type="button" class="btn act-rebuild"'
            f' title="Só reconstrói a página a partir dos dados atuais">'
            f'⟳ Reconstruir página</button>'
            f'<span class="job-status"></span></div></div>')


def markdown_file(path: Path) -> str:
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    return md_lib.markdown(text, extensions=["tables", "fenced_code"])
