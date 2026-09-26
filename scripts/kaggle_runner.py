"""Orquestra execuções do projeto no Kaggle (dataset + kernel + resultados).

Ciclo completo sem sair do terminal, usando a CLI oficial do Kaggle
(`pip install kaggle` + token em ~/.kaggle/kaggle.json):

    # 1. Empacota o projeto (codigo + dataset.csv + progresso optuna_nogate)
    python scripts/kaggle_runner.py package

    # 2. Cria (ou versiona) o Dataset no Kaggle com o pacote
    python scripts/kaggle_runner.py dataset --message "snapshot do estudo"

    # 3. Empurra um kernel que executa um comando do pipeline na GPU do Kaggle
    python scripts/kaggle_runner.py push \
        --command "python pipeline/step_optuna.py --config pipeline/pipeline.optuna_nogate.json --wrappers tcn" \
        --slug optuna-nogate-tcn

    # 4. Acompanha (ou usa --wait para bloquear ate terminar)
    python scripts/kaggle_runner.py status --slug optuna-nogate-tcn

    # 5. Baixa o resultado e restaura o progresso localmente
    python scripts/kaggle_runner.py output --slug optuna-nogate-tcn

`run-all` encadeia 1-5 com polling. O kernel gerado descompacta o pacote em
/kaggle/working, executa o comando informado e zipa os outputs de volta.

Notas:
  - Kernels via API usam UMA GPU (T4 ou P100); para T4 x2 em paralelo, use o
    fluxo de notebook (notebooks/kaggle_optuna_nogate.ipynb) pela interface.
  - Requer conta com telefone verificado (GPU + Internet no kernel).
  - O comando roda a partir da raiz do repo descompactado; caminhos relativos
    do config funcionam normalmente.
"""

import argparse
import json
import os
import shutil
import subprocess
import time
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DIST = PROJECT_ROOT / "dist"
KAGGLE_DIR = DIST / "kaggle"
DEFAULT_DATASET_SLUG = "wind-speed-forecasting"
DEFAULT_DATASET_TITLE = "wind speed forecasting (seq2seq lidar)"
DEFAULT_OUTPUT_GLOB = "pipeline/tmp/optuna_nogate"

KERNEL_TEMPLATE = '''"""Kernel gerado por scripts/kaggle_runner.py — NAO editar a mao."""
import glob
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

INPUT = Path("/kaggle/input")
WORK = Path("/kaggle/working")
REPO = WORK / "wind-speed-forecasting"
COMMAND = {command!r}
OUTPUT_GLOB = {output_glob!r}
PROGRESS_SUBDIR = {progress_subdir!r}

print("Conteudo de /kaggle/input:")
for p in sorted(INPUT.rglob("*")):
    if p.is_file():
        print("  ", p.relative_to(INPUT))

REPO.mkdir(parents=True, exist_ok=True)
# TF tende a reter blocos liberados no host (RAM creep -> OOM-kill em
# sessoes longas); o mesmo remedio do watchdog local.
os.environ.setdefault("MALLOC_TRIM_THRESHOLD_", "134217728")
zips = sorted(INPUT.rglob("*.zip"))
# Prioriza o pacote atual (colab_round2_package_<data>.zip): o dataset acumula
# zips legados (kaggle_optuna_*) de versoes anteriores que viriam antes na
# ordenacao e nao contem o repo/scripts atuais.
colab_zips = [z for z in zips if "colab_round2_package_" in z.name]
legacy_zips = [z for z in zips if "kaggle_optuna_" in z.name]
repo_zip = (sorted(colab_zips)[-1] if colab_zips
            else (sorted(legacy_zips)[-1] if legacy_zips else None))
if repo_zip is not None:
    print("repo zip:", repo_zip)
    with zipfile.ZipFile(repo_zip) as z:
        z.extractall(REPO)
else:
    roots = set()
    for csv in INPUT.rglob("dataset.csv"):
        d = csv.parent
        while d != d.parent:
            if (d / "src").is_dir() or (d / "pipeline").is_dir():
                roots.add(d)
                break
            d = d.parent
    if not roots:
        raise FileNotFoundError("pacote do projeto nao encontrado em /kaggle/input")
    # Prefere o repo atual (com o worker desta rodada) sobre snapshots legados
    # (kaggle_optuna_*, optuna_nogate_v2, ...) que o dataset acumulou.
    src_root = next(
        (r for r in sorted(roots)
         if (r / "scripts" / "kaggle_round2_worker.py").is_file()),
        sorted(roots)[0],
    )
    print("repo extraido pelo Kaggle em:", src_root, "-> copiando")
    shutil.copytree(src_root, REPO, dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))

# Ultimo recurso: repo incompleto -> clona o publico do GitHub (o kernel tem
# internet). dataset.csv (gitignored) vem do input.
if not (REPO / "scripts" / "kaggle_round2_worker.py").is_file():
    print("repo sem o worker da rodada; clonando do GitHub")
    import tempfile
    tmp = REPO.parent / "repo_clone"
    subprocess.check_call(["git", "clone", "--depth", "1",
                           "https://github.com/zinoLath/wind-speed-forescasting.git",
                           str(tmp)])
    shutil.copytree(tmp, REPO, dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".git"))
if not (REPO / "data" / "dataset.csv").is_file():
    src_csv = next(INPUT.rglob("dataset.csv"), None)
    if src_csv is not None:
        (REPO / "data").mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_csv, REPO / "data" / "dataset.csv")
        print("dataset.csv copiado de:", src_csv)

progress_zip = next(
    (z for z in zips if PROGRESS_SUBDIR in z.name and "kaggle_optuna_" not in z.name),
    None,
)
if progress_zip is not None:
    print("restaurando progresso:", progress_zip)
    with zipfile.ZipFile(progress_zip) as z:
        z.extractall(REPO)
else:
    # Restaura apenas sqlites da RODADA ATUAL: o dataset acumula progresso de
    # estudos antigos (optuna_nogate, optuna_round2...), e copia-los para a
    # pasta desta rodada contaminaria o estudo novo.
    _cfg_path = REPO / COMMAND.split("--config")[1].split()[0]
    try:
        import json as _json
        with open(_cfg_path, encoding="utf-8") as _h:
            _study = _json.load(_h).get("optuna", {{}}).get("study_name")
    except Exception:
        _study = None
    target = REPO / "pipeline" / "tmp" / PROGRESS_SUBDIR
    for d in sorted({{p.parent for p in INPUT.rglob("optuna.db")}}):
        if PROGRESS_SUBDIR not in str(d) or not _study:
            continue
        try:
            import sqlite3 as _sq
            _con = _sq.connect(d / "optuna.db")
            _names = [r[0] for r in _con.execute("SELECT study_name FROM studies")]
            _con.close()
        except Exception:
            _names = []
        if _study not in _names:
            continue
        print("restaurando progresso (pasta):", d)
        shutil.copytree(d, target / d.name, dirs_exist_ok=True)

assert (REPO / "data" / "dataset.csv").is_file(), "dataset.csv ausente"
print("repo em:", REPO)

os.chdir(REPO)
print("instalando dependencias do projeto no kernel...")
subprocess.check_call([
    sys.executable, "-m", "pip", "install", "-q",
    "tensorflow[and-cuda]==2.21.0", "keras==3.12.1", "keras-tcn==3.5.6",
    "optuna==4.9.0", "PyWavelets==1.8.0", "pandas>=2.0",
    "scikit-learn>=1.3", "numpy>=1.26",
])
print("\\n$ " + COMMAND)
code = subprocess.call(COMMAND, shell=True)
print("\\ncomando terminou com codigo", code)

# Zipa o progresso SEMPRE (inclusive em erro): os arquivos presentes em
# /kaggle/working sao expostos como output mesmo com o kernel morto, e o
# sqlite permite retomar o estudo na proxima rodada.
for pattern in glob.glob(OUTPUT_GLOB):
    out = WORK / (Path(pattern).name + ".zip")
    if os.path.isdir(pattern):
        # base_dir + root_dir preservam o caminho relativo completo no
        # zip (pipeline/tmp/...), para que o restore via extractall(REPO)
        # coloque o progresso no lugar certo.
        shutil.make_archive(str(out.with_suffix("")), "zip", ".", pattern)
    else:
        shutil.copy2(pattern, out)
    print("output:", out)
sys.exit(code)
'''


def kaggle_username():
    if os.environ.get("KAGGLE_USERNAME"):
        return os.environ["KAGGLE_USERNAME"]
    candidates = [Path.home() / ".kaggle" / "kaggle.json"]
    if os.environ.get("KAGGLE_CONFIG_DIR"):
        candidates.insert(0, Path(os.environ["KAGGLE_CONFIG_DIR"]) / "kaggle.json")
    token = next((c for c in candidates if c.is_file()), None)
    if token is None:
        raise SystemExit(
            "Token do Kaggle nao encontrado em ~/.kaggle/kaggle.json.\n"
            "Crie em kaggle.com -> Settings -> API -> Create New Token e salve"
            " o kaggle.json nesse caminho (pip install kaggle)."
        )
    with open(token, encoding="utf-8") as handle:
        return json.load(handle)["username"]


def _cli_env():
    """Env de auth para a CLI do Kaggle (usada em todos os subprocessos).

    Access tokens novos (KGAT, ``~/.kaggle/access_token``) nao passam pelo
    gate legado da CLI (kaggle.json/KAGGLE_USERNAME+KAGGLE_KEY), mas o
    transporte HTTP (kagglesdk) autentica via ``KAGGLE_API_TOKEN`` com
    prioridade. O KAGGLE_KEY placeholder so satisfaz o gate; as requisicoes
    usam o bearer token. Requer KAGGLE_USERNAME no env quando nao ha
    kaggle.json.
    """
    env = dict(os.environ)
    access_token = Path.home() / ".kaggle" / "access_token"
    if access_token.is_file() and not env.get("KAGGLE_API_TOKEN"):
        env["KAGGLE_API_TOKEN"] = access_token.read_text().strip()
    if env.get("KAGGLE_API_TOKEN"):
        if not env.get("KAGGLE_USERNAME"):
            raise SystemExit(
                "KAGGLE_API_TOKEN configurado mas KAGGLE_USERNAME ausente"
                " (necessario sem kaggle.json): export KAGGLE_USERNAME=<seu user>"
            )
        env.setdefault("KAGGLE_KEY", "unused-with-api-token")
    return env


def run(cmd, **kwargs):
    print("+", " ".join(str(c) for c in cmd))
    kwargs.setdefault("env", _cli_env())
    return subprocess.run([str(c) for c in cmd], check=False, **kwargs)


def cmd_package(args):
    result = run(["bash", PROJECT_ROOT / "scripts" / "kaggle_package.sh"])
    if result.returncode != 0:
        raise SystemExit(result.returncode)
    zips = sorted(DIST.glob("kaggle_optuna_*.zip"))
    if zips:
        print("zip pronto:", zips[-1])


def _prepare_dataset_folder(zip_path, slug, title):
    folder = KAGGLE_DIR / "dataset"
    if folder.exists():
        shutil.rmtree(folder)
    folder.mkdir(parents=True)
    shutil.copy2(zip_path, folder / zip_path.name)
    metadata = {
        "title": title,
        "id": f"{kaggle_username()}/{slug}",
        "licenses": [{"name": "CC0-1.0"}],
    }
    with open(folder / "dataset-metadata.json", "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    return folder, zip_path.name


def cmd_dataset(args):
    zip_path = Path(args.zip) if args.zip else max(
        DIST.glob("kaggle_optuna_*.zip"), key=lambda p: p.stat().st_mtime
    )
    folder, zip_name = _prepare_dataset_folder(zip_path, args.slug, args.title)
    full_id = f"{kaggle_username()}/{args.slug}"
    exists = run(["kaggle", "datasets", "status", full_id]).returncode == 0
    if exists:
        print(f"dataset {full_id} ja existe -> versionando")
        result = run(["kaggle", "datasets", "version", "-p", folder,
                      "-m", args.message, "--dir-mode", "zip"])
    else:
        print(f"dataset {full_id} novo -> criando")
        result = run(["kaggle", "datasets", "create", "-p", folder,
                      "--dir-mode", "zip"])
    if result.returncode != 0:
        raise SystemExit(result.returncode)
    print(f"dataset {full_id} atualizado com {zip_name}")


def cmd_push(args):
    username = kaggle_username()
    kernel_dir = KAGGLE_DIR / "kernel"
    if kernel_dir.exists():
        shutil.rmtree(kernel_dir)
    kernel_dir.mkdir(parents=True)

    script = KERNEL_TEMPLATE.format(
        command=args.command,
        output_glob=args.output_glob,
        progress_subdir=getattr(args, "progress_subdir", "optuna_nogate"),
    )
    script_path = kernel_dir / f"{args.slug}.py"
    script_path.write_text(script, encoding="utf-8")

    metadata = {
        "id": f"{username}/{args.slug}",
        "title": args.title or args.slug.replace("-", " "),
        "code_file": script_path.name,
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,
        "dataset_sources": [f"{kaggle_username()}/{args.dataset}"],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }
    with open(kernel_dir / "kernel-metadata.json", "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    if args.dry_run:
        print(f"--- {script_path} ---")
        print(script)
        print("--- kernel-metadata.json ---")
        print(json.dumps(metadata, indent=2))
        return
    result = run(["kaggle", "kernels", "push", "-p", kernel_dir])
    if result.returncode != 0:
        raise SystemExit(result.returncode)
    print(f"kernel {username}/{args.slug} enviado. Acompanhe com:")
    print(f"  python scripts/kaggle_runner.py status --slug {args.slug}")


def cmd_status(args):
    full_id = f"{kaggle_username()}/{args.slug}"
    result = run(["kaggle", "kernels", "status", full_id])
    raise SystemExit(result.returncode)


def cmd_output(args):
    username = kaggle_username()
    full_id = f"{username}/{args.slug}"
    dest = Path(args.dest) / args.slug
    dest.mkdir(parents=True, exist_ok=True)
    result = run(["kaggle", "kernels", "output", full_id, "-p", dest])
    if result.returncode != 0:
        raise SystemExit(result.returncode)
    print("arquivos baixados em", dest)
    pattern = getattr(args, "pattern", None) or "optuna_nogate*"
    for progress_zip in sorted(dest.glob(pattern + ".zip")):
        if args.restore:
            print("restaurando progresso local:", progress_zip, "->", PROJECT_ROOT)
            with zipfile.ZipFile(progress_zip) as z:
                z.extractall(PROJECT_ROOT)
        else:
            print(f"progresso disponivel em {progress_zip} (use --restore para"
                  f" aplica-lo em pipeline/tmp/{pattern.rstrip('*')}*)")


def cmd_run_all(args):
    cmd_package(args)
    zip_path = max(DIST.glob("kaggle_optuna_*.zip"), key=lambda p: p.stat().st_mtime)
    cmd_dataset(argparse.Namespace(zip=str(zip_path), slug=args.dataset,
                                   title=DEFAULT_DATASET_TITLE,
                                   message=f"snapshot {time.strftime('%F %T')}"))
    cmd_push(argparse.Namespace(command=args.command, slug=args.slug,
                                dataset=args.dataset, title=None,
                                output_glob=args.output_glob,
                                dry_run=False))
    print(f"\nAguardando kernel {args.slug} (poll a cada {args.poll}s)...")
    while True:
        time.sleep(args.poll)
        result = run(["kaggle", "kernels", "status",
                      f"{kaggle_username()}/{args.slug}"],
                     capture_output=True, text=True)
        state = (result.stdout or "").strip()
        print("  status:", state)
        low = state.lower()
        if "complete" in low:
            break
        if "error" in low or "cancel" in low:
            raise SystemExit(f"kernel terminou com erro: {state}")
    cmd_output(argparse.Namespace(slug=args.slug, dest=args.dest,
                                  restore=not args.no_restore))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("package", help="empacota o projeto em dist/kaggle_optuna_*.zip")
    p.set_defaults(func=cmd_package)

    p = sub.add_parser("dataset", help="cria/versiona o Dataset com o pacote")
    p.add_argument("--zip", default=None, help="pacote (default: mais recente em dist/)")
    p.add_argument("--slug", default=DEFAULT_DATASET_SLUG)
    p.add_argument("--title", default=DEFAULT_DATASET_TITLE)
    p.add_argument("--message", default="snapshot")
    p.set_defaults(func=cmd_dataset)

    p = sub.add_parser("push", help="empurra um kernel que roda um comando")
    p.add_argument("--command", required=True,
                   help="comando do pipeline a executar na raiz do repo")
    p.add_argument("--slug", required=True, help="slug do kernel (dashes)")
    p.add_argument("--dataset", default=DEFAULT_DATASET_SLUG)
    p.add_argument("--title", default=None)
    p.add_argument("--output-glob", default=DEFAULT_OUTPUT_GLOB,
                   help="arquivo/pasta zipeado de volta no output")
    p.add_argument("--progress-subdir", default="optuna_nogate",
                   help="subdir de pipeline/tmp restaurado dos inputs")
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(func=cmd_push)

    p = sub.add_parser("status", help="status do kernel")
    p.add_argument("--slug", required=True)
    p.set_defaults(func=cmd_status)

    p = sub.add_parser("output", help="baixa o output do kernel")
    p.add_argument("--slug", required=True)
    p.add_argument("--dest", default=str(DIST / "kaggle_output"))
    p.add_argument("--restore", action="store_true",
                   help="restaura o progresso em pipeline/tmp")
    p.add_argument("--pattern", default="optuna_nogate*",
                   help="glob dos zips de progresso no output (ex.: optuna_round2*)")
    p.set_defaults(func=cmd_output)

    p = sub.add_parser("run-all", help="pacote + dataset + push + espera + output")
    p.add_argument("--command", required=True)
    p.add_argument("--slug", required=True)
    p.add_argument("--dataset", default=DEFAULT_DATASET_SLUG)
    p.add_argument("--output-glob", default=DEFAULT_OUTPUT_GLOB)
    p.add_argument("--dest", default=str(DIST / "kaggle_output"))
    p.add_argument("--poll", type=int, default=120)
    p.add_argument("--no-restore", action="store_true")
    p.set_defaults(func=cmd_run_all)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
