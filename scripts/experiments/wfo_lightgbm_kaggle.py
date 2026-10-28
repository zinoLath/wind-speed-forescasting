"""Distribuicao do walk-forward LightGBM no Kaggle (tcn, tcn_bi).

Ciclo (reusa a infra da rodada Optuna: dataset + kernels GPU + snapshot):

    # 1. Empacota codigo + serie imputada + best_trial.json dos wrappers
    python scripts/experiments/wfo_lightgbm_kaggle.py package

    # 2. Cria/versiona o Dataset no Kaggle com o pacote
    python scripts/experiments/wfo_lightgbm_kaggle.py dataset

    # 3. Empurra 1 kernel GPU por wrapper (snapshot do resultado a cada 10 min)
    python scripts/experiments/wfo_lightgbm_kaggle.py push            # tcn tcn_bi
    python scripts/experiments/wfo_lightgbm_kaggle.py push --wrappers tcn

    # 4. Acompanha (progresso local + status dos kernels)
    python scripts/experiments/wfo_lightgbm_kaggle.py status

    # 5. Traz o resultado, mescla em data/results/wfo_lightgbm e agrega
    python scripts/experiments/wfo_lightgbm_kaggle.py collect

`all` encadeia 1-3. O kernel gerado NUNCA deve ser editado a mao: regenere.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import wfo_lightgbm_common as proto  # noqa: E402
from wfo_notify import notify  # noqa: E402

DIST = PROJECT_ROOT / "dist"
DATASET_SLUG = "wind-speed-wfo-lightgbm"
# OBS: a API do Kaggle derivou o slug do TITULO (ignorando o id do metadata),
# entao o prefixo abaixo precisa bater com o slug real dos kernels.
KERNEL_SLUG_PREFIX = "walk-forward-lightgbm"
SNAPSHOT_SECS = 600

KERNEL_TEMPLATE = '''"""Kernel gerado por scripts/experiments/wfo_lightgbm_kaggle.py — NAO editar a mao."""
import os
import shutil
import subprocess
import sys
import threading
import time
import urllib.request
import zipfile
from pathlib import Path

INPUT = Path("/kaggle/input")
WORK = Path("/kaggle/working")
REPO = WORK / "wind-speed-forecasting"
WRAPPER = {wrapper!r}
SNAPSHOT_SECS = {snapshot_secs!r}
NTFY_URL = "https://ntfy.sh/zino-imputeoptuna"


def _ntfy(title, message, priority="default", tags="satellite"):
    try:
        request = urllib.request.Request(
            NTFY_URL,
            data=message.encode("utf-8"),
            headers={{"Title": title, "Priority": priority, "Tags": tags}},
        )
        urllib.request.urlopen(request, timeout=10).read()
    except Exception as exc:
        print("ntfy falhou:", exc)


os.environ.setdefault("MALLOC_TRIM_THRESHOLD_", "134217728")
REPO.mkdir(parents=True, exist_ok=True)
_ntfy(f"WFO Kaggle {{WRAPPER}} inicio", f"☁️ Kernel Kaggle {{WRAPPER}} iniciado.")

zips = sorted(INPUT.rglob("wfo_lightgbm_package_*.zip"))
if zips:
    print("pacote:", zips[-1])
    with zipfile.ZipFile(zips[-1]) as z:
        z.extractall(REPO)
else:
    # O Kaggle descompacta zips enviados ao dataset: o repo aparece
    # extraido em /kaggle/input — localize a raiz e copie.
    roots = set()
    for marker in INPUT.rglob("wfo_lightgbm_run.py"):
        root = marker.parent.parent
        if (root / "src" / "common.py").is_file() and (root / "data").is_dir():
            roots.add(root)
    if not roots:
        _ntfy("WFO Kaggle erro", "❌ {{WRAPPER}}: nem pacote nem repo extraido em "
              "/kaggle/input (dataset vazio/nao processado).",
              priority="urgent", tags="rotating_light")
        raise FileNotFoundError("pacote/repo ausente em /kaggle/input")
    src_root = sorted(roots)[0]
    print("repo extraido pelo Kaggle em:", src_root, "-> copiando")
    shutil.copytree(src_root, REPO, dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
os.chdir(REPO)

print("instalando dependencias...")
subprocess.check_call([
    sys.executable, "-m", "pip", "install", "-q",
    "tensorflow[and-cuda]==2.21.0", "keras==3.12.1", "keras-tcn==3.5.6",
    "PyWavelets==1.8.0", "pandas>=2.0", "scikit-learn>=1.3", "numpy>=1.26",
])


def _zip_results(target):
    src = Path("data/results/wfo_lightgbm")
    if not src.is_dir():
        return False
    tmp = Path(str(target) + ".tmp")
    shutil.make_archive(str(tmp.with_suffix("")), "zip", ".", str(src))
    os.replace(tmp.with_suffix(".zip"), target)
    return True


def _snapshot_loop():
    while True:
        time.sleep(SNAPSHOT_SECS)
        try:
            if _zip_results(WORK / "wfo_lightgbm.zip"):
                print("snapshot:", time.strftime("%F %T"))
        except Exception as exc:  # snapshot nunca derruba o kernel
            print("snapshot falhou:", exc)


threading.Thread(target=_snapshot_loop, daemon=True).start()

command = f"python scripts/experiments/wfo_lightgbm_run.py --wrappers {{WRAPPER}}"
print("$", command, flush=True)
try:
    code = subprocess.call(command, shell=True)
except Exception as exc:
    code = 1
    _ntfy("WFO Kaggle crash", f"❌ {{WRAPPER}}: excecao no kernel: {{exc!r}}",
          priority="urgent", tags="rotating_light")
print("comando terminou com codigo", code)

if code == 0:
    _ntfy("WFO Kaggle OK", f"✅ Kaggle {{WRAPPER}}: concluido (exit 0) — "
          "traga com wfo_lightgbm_kaggle.py collect.")
else:
    _ntfy("WFO Kaggle falha", f"❌ Kaggle {{WRAPPER}}: terminou com exit {{code}} — "
          "leia o log do kernel no site.",
          priority="high", tags="warning")

try:
    _zip_results(WORK / "wfo_lightgbm.zip")
    print("zip final gravado em /kaggle/working/wfo_lightgbm.zip")
except Exception as exc:
    print("zip final falhou:", exc)
sys.exit(code)
'''


# --- CLI Kaggle (bearer token) — mesmo padrao de scripts/dist/colab_round2.py -----

def _kaggle_env():
    env = dict(os.environ)
    token_file = Path.home() / ".kaggle" / "access_token"
    if token_file.is_file() and not env.get("KAGGLE_API_TOKEN"):
        env["KAGGLE_API_TOKEN"] = token_file.read_text().strip()
    if env.get("KAGGLE_API_TOKEN"):
        env.setdefault("KAGGLE_USERNAME", "gate")
        env.setdefault("KAGGLE_KEY", "unused-with-api-token")
    return env


def _kaggle_run(*args, check=True, timeout=90):
    exe = shutil.which("kaggle")
    if not exe:
        raise SystemExit("CLI kaggle ausente no ambiente: pip install kaggle")
    print("+", "kaggle", " ".join(str(a) for a in args), file=sys.stderr)
    try:
        result = subprocess.run([exe, *(str(a) for a in args)], capture_output=True,
                                text=True, env=_kaggle_env(), timeout=timeout)
    except subprocess.TimeoutExpired:
        result = subprocess.CompletedProcess(args, returncode=124,
                                             stdout="", stderr=f"timeout apos {timeout}s")
    if check and result.returncode != 0:
        message = (result.stderr or result.stdout or "").strip()[:400]
        raise SystemExit(f"kaggle {' '.join(str(a) for a in args)} falhou: {message}")
    return result


_KAGGLE_USER_CACHE = None


def _kaggle_user():
    global _KAGGLE_USER_CACHE
    if _KAGGLE_USER_CACHE:
        return _KAGGLE_USER_CACHE
    result = _kaggle_run("kernels", "list", "--mine", check=False)
    for line in (result.stdout or "").splitlines():
        if "/" in line and not line.startswith("-"):
            _KAGGLE_USER_CACHE = line.split("/", 1)[0].strip()
            return _KAGGLE_USER_CACHE
    raise SystemExit(
        "Usuario Kaggle nao detectado (kaggle kernels list --mine vazio). "
        "Verifique o token em ~/.kaggle/access_token ou exporte KAGGLE_USERNAME."
    )


def _quota_hint(result):
    """Detecta bloqueio de cota de GPU na resposta da CLI."""
    text = f"{result.stdout or ''} {result.stderr or ''}".lower()
    return ("quota" in text) or ("hours" in text and "exceed" in text)


# --- Subcomandos -------------------------------------------------------------

def cmd_package(args):
    """Zip: codigo + serie/flags/meta + best_trial.json dos wrappers pedidos."""
    from src import common

    wrappers = args.wrappers or list(proto.KAGGLE_WRAPPERS)
    for required in (proto.SERIES_CSV, proto.FLAGS_CSV, proto.META_JSON):
        if not required.is_file():
            raise SystemExit(f"{required} ausente — rode scripts/experiments/wfo_lightgbm_prepare.py")

    stamp = time.strftime("%Y%m%d")
    DIST.mkdir(parents=True, exist_ok=True)
    out = DIST / f"wfo_lightgbm_package_{stamp}.zip"
    if out.exists() and not args.repackage:
        print(f"Pacote ja existe: {out} (use --repackage para refazer)")
        return

    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for pattern in ("src/**/*.py", "scripts/*.py", "scripts/*.sh"):
            for path in sorted(PROJECT_ROOT.glob(pattern)):
                if "__pycache__" in str(path):
                    continue
                z.write(path, path.relative_to(PROJECT_ROOT))
        for data_file in (proto.SERIES_CSV, proto.FLAGS_CSV, proto.META_JSON,
                          proto.DATA_DIR / "windows_plan.json"):
            if data_file.is_file():
                z.write(data_file, data_file.relative_to(PROJECT_ROOT))
        for wrapper_key in wrappers:
            best = common.find_optuna_result(wrapper_key)
            if best is None:
                raise SystemExit(
                    f"best_trial.json ausente para {wrapper_key}; o kernel rodaria "
                    "com defaults. Use --params-source default para aceitar isso."
                )
            z.write(best, best.relative_to(PROJECT_ROOT))
    print(f"Pacote gerado: {out} ({out.stat().st_size / 1e6:.1f} MB)")


def _latest_package():
    zips = sorted(DIST.glob("wfo_lightgbm_package_*.zip"), key=lambda p: p.stat().st_mtime)
    if not zips:
        raise SystemExit("Nenhum dist/wfo_lightgbm_package_*.zip — rode 'package' primeiro.")
    return zips[-1]


def cmd_dataset(args):
    zip_path = Path(args.zip) if args.zip else _latest_package()
    folder = DIST / "kaggle_wfo_dataset"
    if folder.exists():
        shutil.rmtree(folder)
    folder.mkdir(parents=True)
    shutil.copy2(zip_path, folder / zip_path.name)
    user = _kaggle_user()
    metadata = {
        "title": "wind speed walk-forward lightgbm",
        "id": f"{user}/{args.dataset}",
        "licenses": [{"name": "CC0-1.0"}],
    }
    with open(folder / "dataset-metadata.json", "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    full_id = f"{user}/{args.dataset}"
    exists = _kaggle_run("datasets", "status", full_id, check=False).returncode == 0
    if exists:
        print(f"dataset {full_id} ja existe -> versionando")
        result = _kaggle_run("datasets", "version", "-p", folder,
                             "-m", f"snapshot {time.strftime('%F %T')}",
                             "--dir-mode", "zip", check=False)
    else:
        print(f"dataset {full_id} novo -> criando")
        result = _kaggle_run("datasets", "create", "-p", folder,
                             "--dir-mode", "zip", check=False)
    if result.returncode != 0:
        print((result.stderr or result.stdout or "")[:800])
        raise SystemExit(result.returncode)
    print(f"dataset {full_id} atualizado com {zip_path.name}")

    # kernels empurrados logo apos criar o dataset rodaram sem o pacote em
    # /kaggle/input (dataset ainda processando) — espere ficar "ready".
    if not args.dont_wait:
        for attempt in range(30):
            status = _kaggle_run("datasets", "status", full_id, check=False)
            state = (status.stdout or status.stderr or "").strip().lower()
            print(f"  dataset status: {state or 'desconhecido'}")
            if "ready" in state or "complete" in state:
                return
            if "error" in state:
                raise SystemExit(f"dataset {full_id} em erro: {state}")
            time.sleep(20)
        print("AVISO: dataset ainda nao 'ready' apos 10 min; prosseguindo assim mesmo.")


def cmd_push(args):
    user = _kaggle_user()
    zip_path = _latest_package()
    wrappers = args.wrappers or list(proto.KAGGLE_WRAPPERS)
    pushed = []
    for wrapper_key in wrappers:
        slug = f"{KERNEL_SLUG_PREFIX}-{wrapper_key.replace('_', '-')}"
        kernel_dir = DIST / "kaggle_wfo_kernel" / wrapper_key
        if kernel_dir.exists():
            shutil.rmtree(kernel_dir)
        kernel_dir.mkdir(parents=True)

        script = KERNEL_TEMPLATE.format(wrapper=wrapper_key, snapshot_secs=SNAPSHOT_SECS)
        script_path = kernel_dir / f"{slug.replace('-', '_')}.py"
        script_path.write_text(script, encoding="utf-8")
        metadata = {
            "id": f"{user}/{slug}",
            "title": f"walk-forward lightgbm {wrapper_key}",
            "code_file": script_path.name,
            "language": "python",
            "kernel_type": "script",
            "is_private": True,
            "enable_gpu": True,
            "enable_internet": True,
            "dataset_sources": [f"{user}/{args.dataset}"],
            "competition_sources": [],
            "kernel_sources": [],
            "model_sources": [],
        }
        with open(kernel_dir / "kernel-metadata.json", "w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2)

        if args.dry_run:
            print(f"--- {script_path} ---\n{script}\n--- kernel-metadata.json ---")
            print(json.dumps(metadata, indent=2))
            continue
        print(f"empurrando kernel {user}/{slug} (pacote {zip_path.name})...")
        result = _kaggle_run("kernels", "push", "-p", kernel_dir, check=False)
        if result.returncode != 0:
            if _quota_hint(result):
                print(f"SEM COTA GPU no Kaggle para {wrapper_key}: rode localmente com")
                print(f"  python scripts/experiments/wfo_lightgbm_run.py --wrappers {wrapper_key}")
                notify(
                    "WFO Kaggle cota",
                    f"🚫 Sem cota GPU no Kaggle para {wrapper_key} — rode localmente: "
                    f"python scripts/experiments/wfo_lightgbm_run.py --wrappers {wrapper_key}",
                    priority="high", tags="no_entry_sign",
                )
            print((result.stderr or result.stdout or "")[:800])
            continue
        pushed.append(slug)
        print(f"  {user}/{slug} enviado.")
    if not args.dry_run and pushed:
        print("\nAcompanhe com: python scripts/experiments/wfo_lightgbm_kaggle.py status")


def cmd_status(args):
    user = _kaggle_user()
    wrappers = args.wrappers or list(proto.WRAPPERS)
    if proto.PROGRESS_JSON.is_file():
        with open(proto.PROGRESS_JSON, encoding="utf-8") as handle:
            progress = json.load(handle)
        print("Local (progress.json):")
        for wrapper_key, info in progress.get("wrappers", {}).items():
            print(f"  {wrapper_key}: {info['done']}/{info['total']} "
                  f"[{info['status']}] ram={info.get('ram_gb')}GB "
                  f"atualizado={info.get('updated')}")
    else:
        print("Local: sem progress.json (nada rodou ainda).")
    print("Kaggle:")
    for wrapper_key in wrappers:
        slug = f"{KERNEL_SLUG_PREFIX}-{wrapper_key.replace('_', '-')}"
        result = _kaggle_run("kernels", "status", f"{user}/{slug}", check=False)
        state = (result.stdout or result.stderr or "").strip()
        print(f"  {wrapper_key} ({slug}): {state or 'sem kernel'}")


def cmd_collect(args):
    user = _kaggle_user()
    wrappers = args.wrappers or list(proto.KAGGLE_WRAPPERS)
    dest_base = DIST / "kaggle_output"
    for wrapper_key in wrappers:
        slug = f"{KERNEL_SLUG_PREFIX}-{wrapper_key.replace('_', '-')}"
        result = _kaggle_run("kernels", "status", f"{user}/{slug}", check=False)
        state = (result.stdout or "").lower()
        print(f"{wrapper_key}: {state.strip() or 'status indisponivel'}")
        if "complete" not in state and not args.force:
            print("  ainda nao terminou (use --force para baixar mesmo assim); pulando.")
            continue
        dest = dest_base / slug
        dest.mkdir(parents=True, exist_ok=True)
        out = _kaggle_run("kernels", "output", f"{user}/{slug}", "-p", dest, check=False)
        if out.returncode != 0:
            print((out.stderr or out.stdout or "")[:400])
            continue
        zips = sorted(dest.glob("wfo_lightgbm*.zip"))
        if not zips:
            print(f"  nenhum wfo_lightgbm*.zip no output de {slug}")
            continue
        print(f"  extraindo {zips[-1].name} -> {PROJECT_ROOT}")
        with zipfile.ZipFile(zips[-1]) as z:
            z.extractall(PROJECT_ROOT)
        proto.finalize_wrapper(wrapper_key)
    print("\nDepois do collect completo: python scripts/experiments/wfo_lightgbm_report.py")


def cmd_all(args):
    if not hasattr(args, "zip"):
        args.zip = None
    cmd_package(args)
    cmd_dataset(args)
    cmd_push(args)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)

    def add_common(p):
        p.add_argument("--wrappers", nargs="+", default=None,
                       choices=proto.WRAPPERS,
                       help="default: tcn tcn_bi (package/push/collect) ou todas (status)")
        p.add_argument("--dataset", default=DATASET_SLUG)

    p = sub.add_parser("package", help="gera dist/wfo_lightgbm_package_<data>.zip")
    p.add_argument("--repackage", action="store_true")
    add_common(p)
    p.set_defaults(func=cmd_package)

    p = sub.add_parser("dataset", help="cria/versiona o Dataset com o pacote")
    p.add_argument("--zip", default=None)
    p.add_argument("--dont-wait", action="store_true",
                   help="nao espera o dataset ficar 'ready' antes de retornar")
    add_common(p)
    p.set_defaults(func=cmd_dataset)

    p = sub.add_parser("push", help="empurra 1 kernel GPU por wrapper")
    p.add_argument("--dry-run", action="store_true")
    add_common(p)
    p.set_defaults(func=cmd_push)

    p = sub.add_parser("status", help="progresso local + status dos kernels")
    add_common(p)
    p.set_defaults(func=cmd_status)

    p = sub.add_parser("collect", help="baixa outputs, mescla e agrega")
    p.add_argument("--force", action="store_true")
    add_common(p)
    p.set_defaults(func=cmd_collect)

    p = sub.add_parser("all", help="package + dataset + push")
    p.add_argument("--repackage", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    add_common(p)
    p.set_defaults(func=cmd_all)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
