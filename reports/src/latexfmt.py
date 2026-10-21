"""Snippets LaTeX exportáveis dos relatórios (formatos steam, sbsr, generic).

Fonte única usada tanto pelo Python (preview inicial) quanto pelo popup JS
(`templates_json`): os templates usam placeholders {file}, {width},
{caption}, {label}, {placement} e {body}.

Convenções copiadas dos projetos reais:
- steam  (classe imc/UNIFEI): figure*/table com caption ACIMA da tabela,
  includegraphics em \\textwidth, caminho com prefixo `figs/`.
- sbsr   (Simpósio Brasileiro de Sensoriamento Remoto): figure/table [h]
  com \\begin{center}, largura em \\linewidth, tabela com bordas |c| e
  \\hline em todas as linhas, caption ABAIXO da tabela.
- generic: artigo padrão (booktabs-less), caption da tabela acima.
"""

from __future__ import annotations

import json
import re
import unicodedata

DEFAULT_FORMATS = ("steam", "sbsr", "generic")

FIGURE_TEMPLATES = {
    "generic": (
        "\\begin{{figure}}[{placement}]\n"
        "\\centering\n"
        "\\includegraphics[width={width}\\textwidth]{{{file}}}\n"
        "\\caption{{{caption}}}\n"
        "\\label{{{label}}}\n"
        "\\end{{figure}}"
    ),
    "steam": (
        "\\begin{{figure*}}[{placement}]\n"
        "    \\begin{{center}}\n"
        "    \\includegraphics[width={width}\\textwidth]{{{file}}}\n"
        "    \\caption{{{caption}}}\n"
        "    \\label{{{label}}}\n"
        "    \\end{{center}}\n"
        "\\end{{figure*}}"
    ),
    "sbsr": (
        "\\begin{{figure}}[{placement}] % use [t] para forçar a figura ao topo\n"
        "    \\begin{{center}}\n"
        "    \\includegraphics[width={width}\\linewidth]{{{file}}}\n"
        "    \\caption{{{caption}}}\n"
        "    \\label{{{label}}}\n"
        "    \\end{{center}}\n"
        "\\end{{figure}}"
    ),
}

TABLE_TEMPLATES = {
    "generic": (
        "\\begin{{table}}[{placement}]\n"
        "\\centering\n"
        "\\caption{{{caption}}}\n"
        "\\label{{{label}}}\n"
        "{body}\n"
        "\\end{{table}}"
    ),
    "steam": (
        "\\begin{{table}}[{placement}]\n"
        "\\scriptsize\n"
        "\\setlength{{\\tabcolsep}}{{3.5pt}}\n"
        "\\centering\n"
        "\\caption{{{caption}}}\n"
        "\\label{{{label}}}\n"
        "{body}\n"
        "\\end{{table}}"
    ),
    "sbsr": (
        "\\begin{{table}}[{placement}] % use [t] para forçar a tabela ao topo\n"
        "    \\centering\n"
        "{body}\n"
        "    \\caption{{{caption}}}\n"
        "    \\label{{{label}}}\n"
        "\\end{{table}}"
    ),
}

FORMAT_LABELS = {
    "steam": "STEAM (imc/UNIFEI)",
    "sbsr": "SBSR",
    "generic": "Artigo genérico",
}


def slug(text: str) -> str:
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = re.sub(r"[^a-zA-Z0-9]+", "_", text).strip("_").lower()
    return text or "item"


def _latex_escape(text: str) -> str:
    return (text.replace("\\", r"\textbackslash{}")
                .replace("&", r"\&").replace("%", r"\%")
                .replace("#", r"\#").replace("_", r"\_")
                .replace("{", r"\{").replace("}", r"\}")
                .replace("~", r"\textasciitilde{}")
                .replace("^", r"\textasciicircum{}"))


def tabular_body(columns, rows, fmt: str, align=None) -> str:
    """Corpo \\begin{tabular}...\\end{tabular} no estilo do formato."""
    n = len(columns)
    align = align or ("l" + "c" * (n - 1))
    if fmt in ("steam", "sbsr"):
        spec = "|" + "|".join(align[:n]) + "|"
        lines = [f"\\begin{{tabular}}{{{spec}}}", "\\hline"]
        header = " & ".join(f"\\textbf{{{_latex_escape(str(c))}}}"
                            for c in columns)
        lines.append(header + " \\\\ \\hline")
        for row in rows:
            cells = " & ".join(_latex_escape(str(c)) for c in row)
            lines.append(cells + " \\\\ \\hline")
        lines.append("\\end{tabular}")
        body = "\n".join(lines)
        if fmt == "steam":
            body = body.replace("\\begin{tabular}",
                                "\\rowcolor[HTML]{EFEFEF}\n\\begin{tabular}",
                                1)
        return "\n".join("    " + ln if fmt == "sbsr" else ln
                         for ln in body.splitlines())
    spec = "".join(align[:n])
    lines = [f"\\begin{{tabular}}{{{spec}}}", "\\hline",
             " & ".join(f"\\textbf{{{_latex_escape(str(c))}}}"
                        for c in columns) + " \\\\ \\hline"]
    for row in rows:
        lines.append(" & ".join(_latex_escape(str(c)) for c in row)
                     + " \\\\")
    lines.append("\\hline\n\\end{tabular}")
    return "\n".join(lines)


def render_figure(fmt: str, *, file: str, caption: str, label: str,
                  width: float = 0.8, placement: str = "t") -> str:
    return FIGURE_TEMPLATES[fmt].format(
        file=_latex_escape(file), caption=_latex_escape(caption),
        label=label, width=f"{width:g}", placement=placement)


def render_table(fmt: str, *, caption: str, label: str, body: str,
                 placement: str = "h") -> str:
    return TABLE_TEMPLATES[fmt].format(
        caption=_latex_escape(caption), label=label, body=body,
        placement=placement)


def templates_json(formats=DEFAULT_FORMATS) -> str:
    """JSON com os templates por formato, embutido na página para o popup."""
    return json.dumps(
        {fmt: {"figure": FIGURE_TEMPLATES[fmt],
               "table": TABLE_TEMPLATES[fmt],
               "name": FORMAT_LABELS.get(fmt, fmt)}
         for fmt in formats if fmt in FIGURE_TEMPLATES},
        ensure_ascii=False)
