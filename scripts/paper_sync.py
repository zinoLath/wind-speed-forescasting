#!/usr/bin/env python3
"""Synchronize the paper figures between the report outputs and Overleaf.

The figures of the article live in ``paper/figures/`` (tracked) and are
produced from the report outputs (gitignored, regenerated on every run):

- ``data/results/report_homologado/images``  (resultados dos modelos)
- ``data/results/baselines/arima_rf``        (comparacao por horizonte)
- ``data/eda/images``                        (EDA do lidar)

``paper/figures_manifest.json`` maps each source file to its canonical name
inside ``paper/figures/``. The ``push`` command sends the tracked figures to
the Overleaf project through the git bridge (premium feature):

    Menu -> Git -> https://git.overleaf.com/<project-id>

Credentials (Overleaf email + Git integration token) come from the
``OVERLEAF_USER``/``OVERLEAF_TOKEN`` environment variables or are embedded in
the remote URL stored in ``paper/overleaf_remote.txt`` (gitignored).

The post-commit hook (``scripts/hooks``) runs ``push --auto`` so every commit
refreshes the Overleaf project. Standalone usage:

    python scripts/paper_sync.py init-manifest   # rescan sources
    python scripts/paper_sync.py list            # manifest + availability
    python scripts/paper_sync.py copy            # report outputs -> paper/figures
    python scripts/paper_sync.py push            # paper/figures -> Overleaf
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from urllib.parse import quote

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PAPER_DIR = PROJECT_ROOT / "paper"
FIGURES_DIR = PAPER_DIR / "figures"
MANIFEST_PATH = PAPER_DIR / "figures_manifest.json"
REMOTE_FILE = PAPER_DIR / "overleaf_remote.txt"
CLONE_DIR = PAPER_DIR / ".overleaf-clone"

DEFAULT_SOURCES = {
    "resultados": "data/results/report_homologado/images",
    "baselines": "data/results/baselines/arima_rf",
    "eda": "data/eda/images",
}


def run_git(args, cwd, check=True, capture=True):
    return subprocess.run(
        ["git", *args], cwd=cwd, check=check,
        capture_output=capture, text=True,
    )


def load_manifest():
    if not MANIFEST_PATH.is_file():
        sys.exit(f"manifest ausente: {MANIFEST_PATH} (rode 'init-manifest')")
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def init_manifest():
    """Scan the report output dirs and (re)write the figures manifest."""
    figures = []
    for tag, rel in DEFAULT_SOURCES.items():
        source_dir = PROJECT_ROOT / rel
        if not source_dir.is_dir():
            print(f"aviso: fonte ausente, ignorada no manifest: {rel}")
            continue
        for path in sorted(source_dir.glob("*.png")):
            figures.append({"source": f"{tag}/{path.name}", "target": path.name})
    manifest = {"sources": dict(DEFAULT_SOURCES), "figures": figures}
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"manifest gravado com {len(figures)} figura(s): {MANIFEST_PATH}")


def list_figures():
    manifest = load_manifest()
    for entry in manifest["figures"]:
        source = manifest["sources"].get(entry["source"].split("/", 1)[0], "")
        path = PROJECT_ROOT / source / entry["source"].split("/", 1)[1]
        target = FIGURES_DIR / entry["target"]
        state = "ok  " if path.is_file() else "MISS"
        synced = "sincronizada" if target.is_file() else "-"
        print(f"[{state}] {entry['source']:<45} -> {entry['target']:<45} {synced}")


def copy_figures(only=None, report_dir=None):
    manifest = load_manifest()
    sources = dict(manifest["sources"])
    if report_dir:
        sources["resultados"] = str(report_dir)
    copied = missing = 0
    for entry in manifest["figures"]:
        if only and not any(pattern in entry["target"] for pattern in only):
            continue
        tag, name = entry["source"].split("/", 1)
        path = PROJECT_ROOT / sources[tag] / name
        if not path.is_file():
            print(f"aviso: fonte ausente: {path}")
            missing += 1
            continue
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, FIGURES_DIR / entry["target"])
        copied += 1
    print(f"{copied} figura(s) copiada(s) para {FIGURES_DIR}"
          + (f" | {missing} ausente(s)" if missing else ""))


def resolve_remote():
    """Return (authenticated_url, display_url) or None when not configured."""
    remote = os.environ.get("OVERLEAF_REMOTE", "").strip()
    if not remote and REMOTE_FILE.is_file():
        remote = REMOTE_FILE.read_text(encoding="utf-8").strip()
    if not remote:
        return None
    user = os.environ.get("OVERLEAF_USER", "").strip()
    token = os.environ.get("OVERLEAF_TOKEN", "").strip()
    if user and token and "@" not in remote:
        remote = re.sub(r"^https://", f"https://{quote(user)}:{quote(token)}@", remote)
    display = re.sub(r"//[^@/]+@", "//<credenciais>@", remote)
    return remote, display


def _default_branch():
    head = run_git(["symbolic-ref", "--short", "refs/remotes/origin/HEAD"],
                   CLONE_DIR, check=False)
    if head.returncode == 0:
        return head.stdout.strip().split("/", 1)[-1]
    for branch in ("master", "main"):
        probe = run_git(["rev-parse", "--verify", f"origin/{branch}"],
                        CLONE_DIR, check=False)
        if probe.returncode == 0:
            return branch
    # empty clone (project without commits): use the local HEAD branch
    head = run_git(["symbolic-ref", "--short", "HEAD"], CLONE_DIR, check=False)
    if head.returncode == 0:
        return head.stdout.strip()
    sys.exit("ramo padrao do clone Overleaf nao identificado")


def push(remote, auto=False, dry_run=False, message=None):
    authed, display = remote
    if dry_run:
        print(f"[dry-run] sincronizaria {FIGURES_DIR} -> {display}")
        return 0
    if CLONE_DIR.exists():
        run_git(["fetch", "origin"], CLONE_DIR, check=not auto)
        branch = _default_branch()
        run_git(["reset", "--hard", f"origin/{branch}"], CLONE_DIR, check=not auto)
    else:
        CLONE_DIR.parent.mkdir(parents=True, exist_ok=True)
        run_git(["clone", authed, str(CLONE_DIR)], CLONE_DIR.parent, check=not auto)
        print(f"clone Overleaf criado em {CLONE_DIR}")
    branch = _default_branch()

    target_dir = CLONE_DIR / "figures"
    target_dir.mkdir(exist_ok=True)
    stale = {p.name for p in target_dir.glob("*.png")} \
        - {p.name for p in FIGURES_DIR.glob("*.png")}
    for name in stale:
        (target_dir / name).unlink()
    for png in sorted(FIGURES_DIR.glob("*.png")):
        shutil.copy2(png, target_dir / png.name)

    run_git(["add", "figures"], CLONE_DIR, check=not auto)
    diff = run_git(["diff", "--cached", "--quiet"], CLONE_DIR, check=False)
    if diff.returncode != 0:
        run_git(
            ["-c", "user.name=paper-sync", "-c", "user.email=paper-sync@local",
             "commit", "-m", message or f"sync figures ({FIGURES_DIR.name})"],
            CLONE_DIR, check=not auto,
        )
        run_git(["push", "origin", branch], CLONE_DIR, check=not auto)
        print(f"Overleaf atualizado ({display}, ramo {branch})")
    else:
        print("Overleaf ja atualizado (nenhuma mudanca nas figuras)")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("init-manifest", help="rescan sources and rewrite the manifest")
    sub.add_parser("list", help="show manifest entries and availability")
    copy_p = sub.add_parser("copy", help="copy report outputs into paper/figures")
    copy_p.add_argument("--report-dir", type=Path, default=None,
                        help="override the 'resultados' source dir")
    copy_p.add_argument("only", nargs="*", help="substring filters on target names")
    push_p = sub.add_parser("push", help="push paper/figures to the Overleaf project")
    push_p.add_argument("--auto", action="store_true",
                        help="hook mode: best-effort, nunca falha o commit")
    push_p.add_argument("--dry-run", action="store_true")
    push_p.add_argument("-m", "--message", default=None)
    args = parser.parse_args()

    if args.command == "init-manifest":
        init_manifest()
    elif args.command == "list":
        list_figures()
    elif args.command == "copy":
        copy_figures(only=args.only or None, report_dir=args.report_dir)
    elif args.command == "push":
        remote = resolve_remote()
        if remote is None:
            note = ("push ignorado: configure paper/overleaf_remote.txt ou "
                    "OVERLEAF_REMOTE (veja paper/README.md)")
            if args.auto:
                print(note)
                return 0
            sys.exit(note)
        try:
            return push(remote, auto=args.auto, dry_run=args.dry_run,
                        message=args.message)
        except Exception as error:  # noqa: BLE001 - hook mode must never fail a commit
            print(f"push Overleaf falhou: {error}")
            return 0 if args.auto else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
