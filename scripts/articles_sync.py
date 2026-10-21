#!/usr/bin/env python3
"""Ponto de commit/push dos artigos Overleaf (articles/steam, articles/sbr).

Cada artigo tem um `sync.json` descrevendo o que sincronizar (figuras de
resultados etc.) e, quando o Overleaf exige credenciais, usa as variáveis
OVERLEAF_USER/OVERLEAF_TOKEN (ver paper/README.md para o formato).

Estrutura de articles/<artigo>/sync.json:
{
  "figures_dir": "figuras",          // subpasta do repo Overleaf
  "clear_before": false,             // limpar figuras antigas do alvo
  "sources": {                       // nome-no-overleaf: arquivo local
    "fig_mse_heatmap.png": "data/results/report_8020/images/06_heatmap_mse_horizonte.png",
    "fig_r2_heatmap.png": "data/results/report_8020/images/06_heatmap_r2_horizonte.png"
  }
}

Comandos:
    python scripts/articles_sync.py list                 # estado dos artigos
    python scripts/articles_sync.py sync sbr             # copia+commit+push
    python scripts/articles_sync.py sync sbr --no-push   # só commit local
    python scripts/articles_sync.py init steam <url>     # liga o git Overleaf
    python scripts/articles_sync.py push-all --auto      # best-effort (hook)

Fluxo recomendado (commit point): depois de regenerar um relatório
(`reports/lidar/8020/report.py`), rode `sync sbr` para levar as figuras
novas ao Overleaf; o `post-commit` do repo principal roda `push-all --auto`.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from urllib.parse import quote

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTICLES_DIR = PROJECT_ROOT / "articles"
KNOWN = ("steam", "sbr")


def run_git(args, cwd, check=False):
    return subprocess.run(["git", *args], cwd=cwd, check=check,
                          capture_output=True, text=True)


def load_config(article: str) -> dict:
    path = ARTICLES_DIR / article / "sync.json"
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"figures_dir": "figures", "sources": {}}


def authenticated_remote(article: str) -> tuple[str, str] | None:
    """(url_com_credenciais, url_para_display) do remote origin."""
    repo = ARTICLES_DIR / article
    if not (repo / ".git").is_dir():
        return None
    remote = run_git(["remote", "get-url", "origin"], repo).stdout.strip()
    if not remote:
        return None
    user = os.environ.get("OVERLEAF_USER", "").strip()
    token = os.environ.get("OVERLEAF_TOKEN", "").strip()
    if user and token and "@" not in remote:
        remote = re.sub(r"^https://",
                        f"https://{quote(user)}:{quote(token)}@", remote)
    display = re.sub(r"//[^@/]+@", "//<credenciais>@", remote)
    return remote, display


def default_branch(repo: Path) -> str:
    head = run_git(["symbolic-ref", "--short", "HEAD"], repo).stdout.strip()
    return head or "master"


def list_articles() -> int:
    for article in KNOWN:
        repo = ARTICLES_DIR / article
        if not repo.is_dir():
            print(f"[{article}] ausente")
            continue
        if not (repo / ".git").is_dir():
            print(f"[{article}] sem .git — rode: articles_sync.py init "
                  f"{article} https://git.overleaf.com/<id>")
            continue
        info = authenticated_remote(article)
        remote = info[1] if info else "(sem remote)"
        dirty = run_git(["status", "--porcelain"], repo).stdout.strip()
        cfg = load_config(article)
        missing = [t for t in cfg.get("sources", {})
                   if not (PROJECT_ROOT / cfg["sources"][t]).is_file()]
        print(f"[{article}] remote={remote} | sources="
              f"{len(cfg.get('sources', {}))} (ausentes: {len(missing)})"
              f" | {'SUJO: ' + str(len(dirty.splitlines())) + ' mudanças' if dirty else 'limpo'}")
        for name, target in missing:
            print(f"    fonte ausente para {name}: {target}")
    return 0


def init_article(article: str, url: str) -> int:
    repo = ARTICLES_DIR / article
    if not repo.is_dir():
        sys.exit(f"{repo} não existe")
    if (repo / ".git").is_dir():
        print(f"[{article}] já é um repositório git")
        run_git(["remote", "set-url", "origin", url], repo)
        print(f"remote origin atualizado para {url}")
        return 0
    run_git(["init", "-b", "master"], repo, check=True)
    run_git(["remote", "add", "origin", url], repo)
    print(f"[{article}] git init concluído; remote origin = {url}")
    print("próximo passo (baixa o estado do Overleaf mantendo seus arquivos "
          "locais):")
    print(f"  cd articles/{article} && git fetch origin && "
          f"git reset origin/master  && git add -A && git commit -m 'local'")
    return 0


def sync_article(article: str, push: bool = True, message: str | None = None,
                 auto: bool = False) -> int:
    repo = ARTICLES_DIR / article
    if not (repo / ".git").is_dir():
        note = f"[{article}] sem .git — ignorado (init primeiro)"
        print(note)
        return 0 if auto else 1
    cfg = load_config(article)
    figures_dir = repo / cfg.get("figures_dir", "figures")
    figures_dir.mkdir(parents=True, exist_ok=True)
    if cfg.get("clear_before"):
        for old in figures_dir.glob("*.png"):
            old.unlink()
    copied = missing = 0
    for name, source in cfg.get("sources", {}).items():
        src = PROJECT_ROOT / source
        if not src.is_file():
            print(f"[{article}] fonte ausente: {source}")
            missing += 1
            continue
        target = figures_dir / name
        target.parent.mkdir(parents=True, exist_ok=True)
        __import__("shutil").copy2(src, target)
        copied += 1
    # stage apenas das figuras — .tex e demais arquivos do autor nunca
    # entram no commit automatico
    rel_figures = str(figures_dir.relative_to(repo))
    run_git(["add", "--", rel_figures], repo)
    dirty = run_git(["status", "--porcelain", "--", rel_figures],
                    repo).stdout.strip()
    if not dirty:
        print(f"[{article}] nada a sincronizar ({copied} figura(s) já "
              "atualizadas)")
        return 0
    commit = run_git(
        ["-c", "user.name=articles-sync", "-c",
         "user.email=articles-sync@local", "commit", "-m",
         message or f"sync figures ({copied} do relatório)"],
        repo)
    if commit.returncode != 0:
        print(f"[{article}] commit falhou: {commit.stderr.strip()}")
        return 1
    print(f"[{article}] commit criado ({copied} figura(s), "
          f"{missing} ausente(s))")
    if not push:
        return 0
    info = authenticated_remote(article)
    if info is None:
        print(f"[{article}] push ignorado (sem remote)")
        return 0 if auto else 1
    branch = default_branch(repo)
    pushed = run_git(["push", "origin", branch], repo)
    if pushed.returncode == 0:
        print(f"[{article}] Overleaf atualizado ({info[1]}, ramo {branch})")
        return 0
    note = f"[{article}] push falhou: {pushed.stderr.strip()}"
    print(note)
    if "Authentication" in pushed.stderr or "403" in pushed.stderr \
            or "could not read" in pushed.stderr.lower():
        print("    configure OVERLEAF_USER/OVERLEAF_TOKEN (ver paper/README.md)")
    return 0 if auto else 1


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list")
    init_p = sub.add_parser("init")
    init_p.add_argument("article", choices=KNOWN)
    init_p.add_argument("url")
    sync_p = sub.add_parser("sync")
    sync_p.add_argument("article", choices=KNOWN)
    sync_p.add_argument("--no-push", action="store_true")
    sync_p.add_argument("-m", "--message", default=None)
    pushall_p = sub.add_parser("push-all")
    pushall_p.add_argument("--auto", action="store_true",
                           help="modo hook: nunca falha, erros são avisos")
    args = parser.parse_args()

    if args.command == "list":
        return list_articles()
    if args.command == "init":
        return init_article(args.article, args.url)
    if args.command == "sync":
        return sync_article(args.article, push=not args.no_push,
                            message=args.message)
    if args.command == "push-all":
        rc = 0
        for article in KNOWN:
            try:
                rc |= sync_article(article, push=True, auto=args.auto)
            except Exception as error:  # noqa: BLE001 - hook nunca falha
                print(f"[{article}] erro: {error}")
                if not args.auto:
                    rc = 1
        return rc
    return 0


if __name__ == "__main__":
    sys.exit(main())
