#!/usr/bin/env python3
"""Ferramentas da base canônica findings/.

Comandos:
  list                 lista findings agrupados por área (a partir do front-matter)
  show <slug>          imprime um finding
  new <slug> --title T --area A [--playbook]
                       cria um finding a partir do TEMPLATE e dá instruções de INDEX
  search <termo>       busca textual em findings/ (knowledge + playbooks)
  check                valida: todo finding está indexado? toda entrada existe?
  fix                  acrescenta ao INDEX.md as entradas faltantes (resumo "(preencher)")

Áreas canônicas: infra, arquitetura, dados, protocolo, resultados, metodos,
relatorios, artigos, processo, playbook.
"""
from __future__ import annotations

import argparse
import datetime
import re
import sys
from pathlib import Path

FINDINGS_DIR = Path(__file__).resolve().parents[1]
INDEX = FINDINGS_DIR / "INDEX.md"
TEMPLATE = FINDINGS_DIR / "TEMPLATE.md"
SUBDIRS = ("knowledge", "playbooks")

AREA_ORDER = [
    "infra", "arquitetura", "dados", "protocolo", "resultados", "metodos",
    "relatorios", "artigos", "processo", "playbook",
]

FM_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.S)


def parse_front_matter(text: str) -> dict:
    m = FM_RE.match(text)
    fm: dict = {}
    if m:
        for line in m.group(1).splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                fm[k.strip()] = v.strip()
    return fm


def load_findings() -> list[dict]:
    out = []
    for sub in SUBDIRS:
        for f in sorted((FINDINGS_DIR / sub).glob("*.md")):
            text = f.read_text(encoding="utf-8")
            fm = parse_front_matter(text)
            out.append({
                "slug": f.stem,
                "path": f.relative_to(FINDINGS_DIR).as_posix(),
                "area": fm.get("area", "processo"),
                "title": fm.get("title", f.stem),
                "updated": fm.get("updated", ""),
            })
    return out


def index_slugs() -> set[str]:
    if not INDEX.exists():
        return set()
    links = re.findall(r"\]\(((?:knowledge|playbooks)/[^)]+)\)", INDEX.read_text(encoding="utf-8"))
    return {Path(l).stem for l in links}


def index_paths() -> set[str]:
    if not INDEX.exists():
        return set()
    links = re.findall(r"\]\(((?:knowledge|playbooks)/[^)]+)\)", INDEX.read_text(encoding="utf-8"))
    return set(links)


def cmd_list(_args) -> None:
    by_area: dict[str, list[dict]] = {}
    for f in load_findings():
        by_area.setdefault(f["area"], []).append(f)
    for area in AREA_ORDER + sorted(set(by_area) - set(AREA_ORDER)):
        if area not in by_area:
            continue
        print(f"\n## {area}")
        for f in by_area[area]:
            print(f"  {f['slug']:44s} {f['updated']}  {f['title']}")


def cmd_show(args) -> None:
    for sub in SUBDIRS:
        p = FINDINGS_DIR / sub / f"{args.slug}.md"
        if p.exists():
            print(p.read_text(encoding="utf-8"))
            return
    sys.exit(f"finding não encontrado: {args.slug}")


def cmd_new(args) -> None:
    sub = "playbooks" if args.playbook else "knowledge"
    area = "playbook" if args.playbook else args.area
    dest = FINDINGS_DIR / sub / f"{args.slug}.md"
    if dest.exists():
        sys.exit(f"já existe: {dest}")
    text = TEMPLATE.read_text(encoding="utf-8")
    text = text.replace("(título curto do conhecimento)", args.title)
    text = text.replace("area: processo", f"area: {area}")
    text = text.replace("updated: 1970-01-01", f"updated: {datetime.date.today().isoformat()}")
    text = text.replace("# (título)", f"# {args.title}")
    dest.write_text(text, encoding="utf-8")
    rel = dest.relative_to(FINDINGS_DIR).as_posix()
    print(f"criado: {dest}")
    print(f"1) preencha o conteúdo e o resumo de 1 linha;")
    print(f"2) adicione ao INDEX.md:  - [{args.slug}]({rel}) — <resumo de 1 linha>")
    print(f"3) rode: python findings/scripts/findings.py check")


def cmd_search(args) -> None:
    pat = re.compile(re.escape(args.termo), re.I)
    for sub in SUBDIRS:
        for f in sorted((FINDINGS_DIR / sub).glob("*.md")):
            lines = f.read_text(encoding="utf-8").splitlines()
            hits = [i for i, l in enumerate(lines, 1) if pat.search(l)]
            if hits:
                print(f"\n{f.relative_to(FINDINGS_DIR)} ({len(hits)} matches)")
                for i in hits[:10]:
                    print(f"  {i:4d}: {lines[i - 1].strip()[:110]}")


def cmd_check(_args) -> None:
    findings = load_findings()
    errors: list[str] = []
    indexed_slugs = index_slugs()
    indexed_paths = index_paths()
    paths = {f["path"] for f in findings}
    for f in findings:
        if f["slug"] not in indexed_slugs or f["path"] not in indexed_paths:
            errors.append(f"NÃO INDEXADO: {f['path']}")
        fm = parse_front_matter((FINDINGS_DIR / f["path"]).read_text(encoding="utf-8"))
        for key in ("title", "area", "updated"):
            if key not in fm:
                errors.append(f"front-matter sem '{key}': {f['path']}")
            elif key == "updated" and fm[key] == "1970-01-01":
                errors.append(f"front-matter 'updated' não preenchido: {f['path']}")
    for p in indexed_paths - paths:
        errors.append(f"INDEX aponta para arquivo inexistente: {p}")
    if errors:
        print("check FALHOU:")
        for e in errors:
            print(f"  - {e}")
        print("corrija ou rode: python findings/scripts/findings.py fix")
        sys.exit(1)
    print(f"check OK — {len(findings)} findings indexados.")


def cmd_fix(_args) -> None:
    findings = load_findings()
    indexed_paths = index_paths()
    missing = [f for f in findings if f["path"] not in indexed_paths]
    if not missing:
        print("nada a corrigir.")
        return
    text = INDEX.read_text(encoding="utf-8")
    blocks: dict[str, list[str]] = {}
    for f in missing:
        blocks.setdefault(f["area"], []).append(
            f"- [{f['slug']}]({f['path']}) — (preencher resumo)")
    lines = text.rstrip("\n").splitlines()
    for area, entries in blocks.items():
        header = f"### {area}"
        if header in lines:
            pos = max(i for i, l in enumerate(lines) if l == header)
            j = pos + 1
            while j < len(lines) and not lines[j].startswith(("### ", "## ")):
                j += 1
            lines[j:j] = entries
        else:
            lines += ["", header, *entries]
    INDEX.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"INDEX.md atualizado com {len(missing)} entrada(s) — preencha os resumos.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list").set_defaults(fn=cmd_list)
    p = sub.add_parser("show"); p.add_argument("slug"); p.set_defaults(fn=cmd_show)
    p = sub.add_parser("new")
    p.add_argument("slug")
    p.add_argument("--title", required=True)
    p.add_argument("--area", default="processo", choices=[a for a in AREA_ORDER if a != "playbook"])
    p.add_argument("--playbook", action="store_true", help="cria em playbooks/ com area=playbook")
    p.set_defaults(fn=cmd_new)
    p = sub.add_parser("search"); p.add_argument("termo"); p.set_defaults(fn=cmd_search)
    sub.add_parser("check").set_defaults(fn=cmd_check)
    sub.add_parser("fix").set_defaults(fn=cmd_fix)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
