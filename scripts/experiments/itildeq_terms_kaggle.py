"""Busca curta dos coeficientes da ITILDE-Q no Kaggle (2 kernels).

Combinações de (α, β, γ) da perda ITILDE-Q treinadas em LSTMs com os HPs
canônicos da rodada 8020-itildeq. Modo padrão: grade OFAT (9 combos) na
LSTM. Modo ``--micro``: coeficientes muito pequenos (pouca influência
sobre o MSE) nas DUAS LSTMs (kernel a = lstm, kernel b = lstm_bi). Uso:

    python scripts/experiments/itildeq_terms_kaggle.py push [--micro] [--dry-run]
    python scripts/experiments/itildeq_terms_kaggle.py status [--micro]
    python scripts/experiments/itildeq_terms_kaggle.py collect [--micro]

Auth: ``~/.kaggle/access_token`` (bearer) ou ``kaggle.json``.
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
OUTPUT_SUBDIR = "itildeq_terms_search"
OUTPUT_GLOB = f"pipeline/tmp/{OUTPUT_SUBDIR}"
RESULTS_JSON = (PROJECT_ROOT / "data" / "results" / "itildeq_terms_search"
                / "terms_search.json")

KERNELS = {
    "terms-a": "itildeq-terms-a",
    "terms-b": "itildeq-terms-b",
}

# grade OFAT (LSTM): default 0.3:0.5:0.1 + MSE pura + um fator por vez
COMBOS_A = [(0.3, 0.5, 0.1),   # controle (default)
            (0.0, 0.5, 0.0),   # MSE pura (referência)
            (0.1, 0.5, 0.1),
            (0.5, 0.5, 0.1),
            (1.0, 0.5, 0.1)]
COMBOS_B = [(0.3, 0.2, 0.1),   # mais fase
            (0.3, 0.8, 0.1),   # mais amplitude-shift
            (0.3, 0.5, 0.0),   # sem amplificação
            (0.3, 0.5, 0.3)]   # amplificação forte

# modo micro: coeficientes muito pequenos (pouca influência sobre o MSE),
# nas DUAS LSTMs; kernel micro-a = lstm, micro-b = lstm_bi
KERNELS_MICRO = {
    "terms-micro-a": ("lstm", "itildeq-terms-micro-a"),
    "terms-micro-b": ("lstm_bi", "itildeq-terms-micro-b"),
}
COMBOS_MICRO = [(0.0, 0.5, 0.0),   # MSE pura (controle de ruído)
                (0.01, 0.5, 0.0),
                (0.01, 0.5, 0.01),
                (0.05, 0.5, 0.0),
                (0.05, 0.5, 0.01),
                (0.1, 0.5, 0.01)]


def kernels_for(args):
    """{slug_full: wrapper} conforme o modo (normal ou micro)."""
    if getattr(args, "micro", False):
        return {full: wrapper for wrapper, full in KERNELS_MICRO.values()}
    return {slug: "lstm" for slug in KERNELS.values()}


def _fmt(combos):
    return ";".join(f"{a}:{b}:{g}" for a, b, g in combos)


def _cli():
    from scripts.dist.colab_round2 import _kaggle_run, _kaggle_user
    return _kaggle_run, _kaggle_user


def _slug(slug):
    _run, user = _cli()
    return f"{user()}/{slug}"


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
             f"itildeq terms {time.strftime('%F %T')}", "--dir-mode", "zip")
    else:
        _run("datasets", "create", "-p", str(folder), "--dir-mode", "zip")
    return full_id


def _push_kernels(args, full_id):
    _run, user = _cli()
    from scripts.dist.kaggle_runner import KERNEL_TEMPLATE

    kernels = kernels_for(args)
    if getattr(args, "micro", False):
        jobs = [(slug, _fmt(COMBOS_MICRO), wrapper)
                for slug, wrapper in kernels.items()]
    else:
        jobs = [(slug, _fmt(combos), "lstm")
                for slug, combos in zip(kernels, [COMBOS_A, COMBOS_B])]
    for slug, combos, wrapper in jobs:
        wflag = f"--wrapper {wrapper} " if wrapper != "lstm" else ""
        command = (f"python scripts/experiments/itildeq_terms_search_train.py "
                   f"{wflag}--combos '{combos}' --output-dir {OUTPUT_GLOB}")
        script = KERNEL_TEMPLATE.format(
            command=command,
            output_glob=OUTPUT_GLOB,
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
            print(f"--- dry-run {slug} ({wrapper}) ---")
            print(command)
            continue
        _run("kernels", "push", "-p", str(kernel_dir))
        print(f"kernel {user()}/{slug} enviado ({wrapper})")


def cmd_push(args):
    zip_path = cmd_package(args)
    full_id = _dataset(args, zip_path)
    _push_kernels(args, full_id)
    if not args.dry_run:
        print("Acompanhe com: python scripts/experiments/itildeq_terms_kaggle.py status "
              "[--micro]")
        print("Ao terminar: python scripts/experiments/itildeq_terms_kaggle.py collect "
              "[--micro]")


def cmd_status(args):
    _run, _user = _cli()
    for slug in kernels_for(args):
        result = _run("kernels", "status", _slug(slug), check=False)
        state = (result.stdout or result.stderr or "").strip()
        print(f"{slug}: {state or 'UNKNOWN'}")


def cmd_collect(args):
    _run, _user = _cli()
    dest_root = DIST / "kaggle_output" / "itildeq_terms"
    results = []
    for slug in kernels_for(args):
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

    out_dir = PROJECT_ROOT / "pipeline" / "tmp" / OUTPUT_SUBDIR
    for json_path in sorted(out_dir.glob("combo_*.json")):
        rec = json.loads(json_path.read_text(encoding="utf-8"))
        if "wrapper" not in rec:
            rec["wrapper"] = ("lstm_bi"
                              if json_path.stem.startswith("combo_lstm_bi_")
                              else "lstm")
        results.append(rec)
    if not results:
        raise SystemExit("nenhum resultado restaurado")

    results.sort(key=lambda r: (r.get("wrapper", "lstm"), r["all"]["mse"]))
    RESULTS_JSON.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_JSON.write_text(json.dumps(results, indent=1), encoding="utf-8")

    print(f"\n=== Ranking por MSE all ({len(results)} combos) — "
          f"salvo em {RESULTS_JSON.relative_to(PROJECT_ROOT)} ===")
    print(f"{'wrapper':<9} {'α':>5} {'β':>5} {'γ':>5} | {'R² all':>7} "
          f"{'MSE':>7} | {'R² h36':>7} {'MSE h36':>8}")
    for r in results:
        print(f"{r.get('wrapper', 'lstm'):<9} {r['alpha']:5.2f} "
              f"{r['beta']:5.2f} {r['gamma']:5.2f} | "
              f"{r['all']['r2']:7.4f} {r['all']['mse']:7.4f} | "
              f"{r['h36']['r2']:7.4f} {r['h36']['mse']:8.4f}")

    if args.build_report:
        subprocess.check_call([sys.executable, "reports/src/builder.py",
                               "build"], cwd=PROJECT_ROOT)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("push", "collect", "status"):
        p = sub.add_parser(name)
        if name in ("push", "collect"):
            p.add_argument("--dataset", default=DATASET_SLUG)
            p.add_argument("--dry-run", action="store_true")
        if name == "collect":
            p.add_argument("--build-report", action="store_true")
        p.add_argument("--micro", action="store_true",
                       help="grade micro (α≤0.1) nas duas LSTMs")
    args = ap.parse_args()
    {"push": cmd_push, "status": cmd_status,
     "collect": cmd_collect}[args.cmd](args)


if __name__ == "__main__":
    main()
