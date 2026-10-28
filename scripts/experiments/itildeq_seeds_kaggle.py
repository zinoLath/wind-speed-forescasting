"""Orquestra o estudo de seeds do ITILDE-Q no Kaggle (2 kernels = 2 GPUs).

Cada arquitetura roda em um kernel GPU próprio — ``lstm`` em uma GPU e
``lstm_bi`` em outra — com o mesmo dataset e o mesmo protocolo 80/20. O
comando de cada kernel é o ``lidar_8020_itildeq_seeds_train.py`` com a lista
de seeds.

    # 1. empacota código + dataset
    python scripts/experiments/itildeq_seeds_kaggle.py package

    # 2. versiona o Dataset e empurra os 2 kernels
    python scripts/experiments/itildeq_seeds_kaggle.py push --seeds 42 43 44 45 46 47 48 49 50 51

    # 3. acompanha
    python scripts/experiments/itildeq_seeds_kaggle.py status

    # 4. baixa os outputs, restaura as predições, roda a análise e o site
    python scripts/experiments/itildeq_seeds_kaggle.py collect --build-report

`run-all` encadeia 1–4 (empurra, espera ambos terminarem e coleta).

Auth: ``~/.kaggle/access_token`` (bearer) ou ``~/.kaggle/kaggle.json``.
"""

import argparse
import json
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DIST = PROJECT_ROOT / "dist"
DATASET_SLUG = "wind-speed-forecasting"
DATASET_TITLE = "wind speed forecasting (seq2seq lidar)"
OUTPUT_SUBDIR = "trained_8020_itildeq_seeds"
OUTPUT_GLOB = f"pipeline/tmp/{OUTPUT_SUBDIR}"

KERNELS = {
    "lstm": "itildeq-seeds-lstm",
    "lstm_bi": "itildeq-seeds-lstm-bi",
}
DEFAULT_SEEDS = list(range(42, 52))


# ---------------------------------------------------------------------------
# Helpers de auth/CLI (reaproveita o fluxo bearer do colab_round2)
# ---------------------------------------------------------------------------

def _cli():
    from scripts.dist.colab_round2 import _kaggle_run, _kaggle_user
    return _kaggle_run, _kaggle_user


def _slug(slug):
    _run, user = _cli()
    return f"{user()}/{slug}"


# ---------------------------------------------------------------------------
# package / dataset / kernels
# ---------------------------------------------------------------------------

def cmd_package(args):
    stamp = time.strftime("%Y%m%d_%H%M")
    out = DIST / f"colab_round2_package_{stamp}.zip"
    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        for pattern in ("src", "scripts", "tests"):
            for path in (PROJECT_ROOT / pattern).rglob("*"):
                if path.is_file() and "__pycache__" not in path.parts \
                        and path.suffix != ".pyc":
                    zf.write(path, path.relative_to(PROJECT_ROOT))
        for pattern in ("pipeline/*.py", "pipeline/*.json",
                        "requirements.txt", "data/dataset.csv"):
            for path in PROJECT_ROOT.glob(pattern):
                if path.is_file():
                    zf.write(path, path.relative_to(PROJECT_ROOT))
    print(f"pacote: {out} ({out.stat().st_size / 1e6:.1f} MB)")
    return out


def _dataset(args, zip_path):
    _run, user = _cli()
    full_id = f"{user()}/{args.dataset}"
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
    if args.dry_run:
        return full_id
    if exists:
        _run("datasets", "version", "-p", str(folder), "-m",
             f"itildeq seeds {time.strftime('%F %T')}", "--dir-mode", "zip")
    else:
        _run("datasets", "create", "-p", str(folder), "--dir-mode", "zip")
    return full_id


def _push_kernels(args, full_id, seeds):
    _run, user = _cli()
    from scripts.dist.kaggle_runner import KERNEL_TEMPLATE

    seed_args = " ".join(str(s) for s in seeds)
    for wrapper, slug in KERNELS.items():
        command = (f"python scripts/experiments/lidar_8020_itildeq_seeds_train.py "
                   f"--wrappers {wrapper} --seeds {seed_args} "
                   f"--output-dir {OUTPUT_GLOB}")
        script = KERNEL_TEMPLATE.format(
            command=command,
            output_glob=f"{OUTPUT_GLOB}/{wrapper}",
            progress_subdir=OUTPUT_SUBDIR,
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
        if args.dry_run:
            print(f"--- dry-run {slug} ---")
            print(script)
            print(json.dumps(metadata, indent=2))
            continue
        _run("kernels", "push", "-p", str(kernel_dir))
        print(f"kernel {user()}/{slug} enviado ({wrapper}, "
              f"{len(seeds)} seeds)")


def cmd_push(args):
    zip_path = cmd_package(args)
    full_id = _dataset(args, zip_path)
    _push_kernels(args, full_id, args.seeds)
    if not args.dry_run:
        print("Acompanhe com: python scripts/experiments/itildeq_seeds_kaggle.py status")
        print("Ao terminar: python scripts/experiments/itildeq_seeds_kaggle.py collect "
              "--build-report")


# ---------------------------------------------------------------------------
# status / collect
# ---------------------------------------------------------------------------

def cmd_status(args):
    _run, _user = _cli()
    for slug in KERNELS.values():
        result = _run("kernels", "status", _slug(slug), check=False)
        state = (result.stdout or result.stderr or "").strip()
        print(f"{slug}: {state or 'UNKNOWN'}")


def _wait_all(poll):
    _run, _user = _cli()
    pending = dict(KERNELS)
    while pending:
        for wrapper, slug in list(pending.items()):
            result = _run("kernels", "status", _slug(slug), check=False)
            state = (result.stdout or result.stderr or "").strip()
            low = state.lower()
            print(f"  {slug}: {state or 'UNKNOWN'}")
            if "complete" in low:
                pending.pop(wrapper, None)
            elif "error" in low or "cancel" in low:
                raise SystemExit(f"kernel {slug} terminou com erro: {state}")
        if pending:
            time.sleep(poll)


def _restore_zip(zip_path):
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        zf.extractall(PROJECT_ROOT)
    print(f"restaurado {zip_path.name}: {len(names)} arquivos")


def cmd_collect(args):
    _run, _user = _cli()
    dest_root = DIST / "kaggle_output" / "itildeq_seeds"
    restored = 0
    for slug in KERNELS.values():
        dest = dest_root / slug
        dest.mkdir(parents=True, exist_ok=True)
        result = _run("kernels", "output", _slug(slug), "-p", str(dest),
                      check=False)
        if result.returncode != 0:
            print(f"aviso: falha ao baixar {slug}: "
                  f"{(result.stderr or result.stdout or '').strip()[:200]}")
            continue
        zips = sorted(dest.glob("*.zip"))
        if not zips:
            print(f"aviso: nenhum zip no output de {slug}")
            continue
        for zip_path in zips:
            _restore_zip(zip_path)
            restored += 1
    if restored == 0:
        raise SystemExit("nenhum output restaurado")

    models_dir = PROJECT_ROOT / "pipeline" / "tmp" / OUTPUT_SUBDIR
    print(f"predições em: {models_dir}")
    if not args.no_analysis:
        subprocess.check_call([
            sys.executable, "studies/itildeq_seeds_8020.py",
            "--models-dir", str(models_dir.relative_to(PROJECT_ROOT)),
            "--out-dir", "data/results/itildeq_seeds_8020",
        ], cwd=PROJECT_ROOT)
    if args.build_report:
        subprocess.check_call([
            sys.executable, "reports/src/builder.py", "build",
        ], cwd=PROJECT_ROOT)


def cmd_run_all(args):
    cmd_push(args)
    _wait_all(args.poll)
    cmd_collect(args)


# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
        p.add_argument("--dataset", default=DATASET_SLUG)
        p.add_argument("--dry-run", action="store_true")

    p = sub.add_parser("package", help="empacota código + dataset")
    p.set_defaults(func=cmd_package)

    p = sub.add_parser("push", help="versiona dataset e empurra os 2 kernels")
    common(p)
    p.set_defaults(func=cmd_push)

    p = sub.add_parser("status", help="status dos 2 kernels")
    p.set_defaults(func=cmd_status)

    p = sub.add_parser("collect", help="baixa outputs e roda a análise")
    p.add_argument("--no-analysis", action="store_true")
    p.add_argument("--build-report", action="store_true")
    p.set_defaults(func=cmd_collect)

    p = sub.add_parser("run-all", help="push + espera + collect")
    common(p)
    p.add_argument("--poll", type=int, default=180)
    p.add_argument("--no-analysis", action="store_true")
    p.add_argument("--build-report", action="store_true")
    p.set_defaults(func=cmd_run_all)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
