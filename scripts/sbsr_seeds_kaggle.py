"""10 seeds (1-10) dos 4 wrappers canônicos SBSR no Kaggle (2 kernels).

Kernel sbsr10-a: LSTM + TCN-Bi · kernel sbsr10-b: LSTM-Bi + TCN — seeds
1 a 10, HPs da canônica 16/10 (com dilatações C′ nas TCNs), perdas da
rodada (MSE pura), grade 80/20 com filtro de observadas. Notifica
início/progresso/fim no ntfy do projeto.

    python scripts/sbsr_seeds_kaggle.py push [--dry-run]
    python scripts/sbsr_seeds_kaggle.py status
    python scripts/sbsr_seeds_kaggle.py collect

Auth: ~/.kaggle/kaggle.json (ou KAGGLE_USERNAME + access token).
"""

import argparse
import json
import shutil
import sys
import time
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DIST = PROJECT_ROOT / "dist"
DATASET_SLUG = "wind-speed-forecasting"
DATASET_TITLE = "wind speed forecasting (seq2seq lidar)"
OUTPUT_GLOB = "pipeline/tmp/sbsr10"
NTFY = "https://n.sh/zino-imputeoptuna"
SEEDS = list(range(1, 11))

KERNELS = {
    "sbsr10-lstm": ("lstm", True),        # leve
    "sbsr10-tcn": ("tcn", True),          # leve
    "sbsr10-lstm-bi": ("lstm_bi", False),  # pesado
    "sbsr10-tcn-bi": ("tcn_bi", False),    # pesado
}
PACKAGE_EXTRA_GLOBS = ("pipeline/tmp/trained_8020_itildeq/*/model.json",)


def _cli():
    from scripts.colab_round2 import _kaggle_run, _kaggle_user
    return _kaggle_run, _kaggle_user


def _slug(slug):
    _run, user = _cli()
    return f"{user()}/{slug}"


def cmd_package():
    stamp = time.strftime("%Y%m%d_%H%M")
    out = DIST / f"colab_round2_package_{stamp}.zip"
    out.parent.mkdir(parents=True, exist_ok=True)
    patterns = ["src", "scripts", "tests"]
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        for pattern in patterns:
            for path in (PROJECT_ROOT / pattern).rglob("*"):
                if path.is_file() and "__pycache__" not in path.parts \
                        and path.suffix != ".pyc":
                    zf.write(path, path.relative_to(PROJECT_ROOT))
        for pattern in ("pipeline/*.py", "pipeline/*.json",
                        "requirements.txt", "data/dataset.csv",
                        *PACKAGE_EXTRA_GLOBS):
            for path in PROJECT_ROOT.glob(pattern):
                if path.is_file():
                    zf.write(path, path.relative_to(PROJECT_ROOT))
    print(f"pacote: {out} ({out.stat().st_size / 1e6:.1f} MB)")
    return out


def _dataset(zip_path, dry=False):
    _run, user = _cli()
    full_id = f"{user()}/{DATASET_SLUG}"
    folder = DIST / "kaggle" / "dataset"
    if folder.exists():
        shutil.rmtree(folder)
    folder.mkdir(parents=True)
    shutil.copy2(zip_path, folder / zip_path.name)
    with open(folder / "dataset-metadata.json", "w", encoding="utf-8") as fh:
        json.dump({"title": DATASET_TITLE, "id": full_id,
                   "licenses": [{"name": "CC0-1.0"}]}, fh, indent=2)
    exists = _run("datasets", "status", full_id, check=False).returncode == 0
    verb = "versionando" if exists else "criando"
    print(f"dataset {full_id}: {verb} com {zip_path.name}")
    if dry:
        return full_id
    if exists:
        _run("datasets", "version", "-p", str(folder), "-m",
             f"sbsr 10 seeds {time.strftime('%F %T')}", "--dir-mode", "zip")
    else:
        _run("datasets", "create", "-p", str(folder), "--dir-mode", "zip")
    return full_id


def _push_kernels(full_id, dry=False, only=()):
    _run, user = _cli()
    from scripts.kaggle_runner import KERNEL_TEMPLATE
    seeds = f"1-{max(SEEDS)}"
    for slug, (wrappers, _light) in KERNELS.items():
        if only and slug not in only:
            continue
        command = (f"curl -s -d 'sbsr10 {slug} INICIADO ({wrappers}, "
                   f"seeds 1-10)' {NTFY} > /dev/null; "
                   f"python scripts/kaggle_seeds_dispatch.py "
                   f"--wrappers {wrappers} --seeds {seeds} "
                   f"--output-dir {OUTPUT_GLOB} --ntfy {NTFY}; "
                   f"curl -s -d 'sbsr10 {slug} OK — 10 seeds concluidas "
                   f"({wrappers})' {NTFY} > /dev/null || "
                   f"curl -s -d 'sbsr10 {slug} FALHOU ({wrappers})' "
                   f"{NTFY} > /dev/null")
        script = KERNEL_TEMPLATE.format(
            command=command,
            output_glob=OUTPUT_GLOB,
            progress_subdir="sbsr10",
        )
        kernel_dir = DIST / "kaggle" / f"kernel-{slug}"
        if kernel_dir.exists():
            shutil.rmtree(kernel_dir)
        kernel_dir.mkdir(parents=True)
        script_path = kernel_dir / f"{slug}.py"
        script_path.write_text(script, encoding="utf-8")
        metadata = {
            "id": f"{user()}/{slug}",
            "title": slug.replace("-", " "),
            "code_file": script_path.name,
            "language": "python",
            "kernel_type": "script",
            "is_private": True,
            "enable_gpu": True,
            "enable_internet": True,
            "dataset_sources": [full_id],
            "competition_sources": [],
            "kernel_sources": [],
            "model_sources": [],
        }
        with open(kernel_dir / "kernel-metadata.json", "w",
                  encoding="utf-8") as fh:
            json.dump(metadata, fh, indent=2)
        if dry:
            print(f"--- dry-run {slug} ({wrappers}) ---")
            print(command)
            continue
        _run("kernels", "push", "-p", str(kernel_dir))
        print(f"kernel {user()}/{slug} enviado ({wrappers})")


def cmd_push(args):
    zip_path = cmd_package()
    full_id = _dataset(zip_path, dry=args.dry_run)
    only = ([k for k in KERNELS
             if any(k.endswith(x) or k == f"sbsr10-{x}" for x in args.kernels)]
            if "all" in args.kernels else
            [f"sbsr10-{k}" for k in args.kernels])
    _push_kernels(full_id, dry=args.dry_run, only=only)
    if not args.dry_run:
        print(f"Acompanhe: python scripts/sbsr_seeds_kaggle.py status "
              f"(ntfy em {NTFY})")
        print("Ao terminar: python scripts/sbsr_seeds_kaggle.py collect")


def cmd_status(_args):
    _run, _user = _cli()
    for slug in KERNELS:
        result = _run("kernels", "status", _slug(slug), check=False)
        state = (result.stdout or result.stderr or "").strip()
        print(f"{slug}: {state or 'UNKNOWN'}")


def cmd_collect(_args):
    _run, _user = _cli()
    dest_root = DIST / "kaggle_output" / "sbsr10"
    for slug in KERNELS:
        dest = dest_root / slug
        dest.mkdir(parents=True, exist_ok=True)
        result = _run("kernels", "output", _slug(slug), "-p", str(dest),
                      check=False)
        if result.returncode != 0:
            print(f"aviso: falha ao baixar {slug}")
            continue
        for zip_path in sorted(dest.glob("*.zip")):
            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(PROJECT_ROOT)
            print(f"restaurado {zip_path.name}")

    import numpy as np
    print("\n=== 10 seeds (1-10) — grade 80/20, filtro de observadas ===")
    for wrapper in ("lstm", "lstm_bi", "tcn", "tcn_bi"):
        paths = sorted((PROJECT_ROOT / OUTPUT_GLOB).glob(
            f"**/{wrapper}/summary.json"))
        if not paths:
            print(f"{wrapper}: summary ausente")
            continue
        runs = {}
        for path in paths:
            runs.update(json.loads(path.read_text(encoding="utf-8"))["runs"])
        r2 = [runs[k]["evaluation"]["all"]["r2"] for k in runs]
        roll = [runs[k]["evaluation"]["rolling"]["r2"] for k in runs]
        print(f"{wrapper:8s} R² {np.mean(r2):.3f} ± {np.std(r2, ddof=1):.3f} "
              f"[{min(r2):.3f}–{max(r2):.3f}] | rolling "
              f"{np.mean(roll):+.3f} ± {np.std(roll, ddof=1):.3f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("push")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--kernels", nargs="+",
                   default=["lstm", "tcn"],
                   choices=["lstm", "tcn", "lstm-bi", "tcn-bi", "all"],
                   help="quais kernels enviar (default: os leves lstm+tcn)")
    p.set_defaults(func=cmd_push)
    sub.add_parser("status").set_defaults(func=cmd_status)
    sub.add_parser("collect").set_defaults(func=cmd_collect)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
