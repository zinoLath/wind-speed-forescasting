"""Dashboard de treino no terminal (estilo nvtop).

Desenha, ao vivo, as curvas de loss/val_loss (e lr/mae quando existirem) dos
treinos gravados por ``src.dashboard.epoch_logger`` — cada ``wrapper.fit()``
appenda uma linha JSON por epoca em ``pipeline/tmp/train_dashboard/``. Rode em
outro painel do tmux enquanto o treino acontece:

    python scripts/train_dashboard.py               # segue o run ativo mais recente
    python scripts/train_dashboard.py --run tcn     # filtra por substring do nome
    python scripts/train_dashboard.py --list        # lista runs e sai
    python scripts/train_dashboard.py --once        # snapshot estatico (sem loop)
    python scripts/train_dashboard.py --dir <dir>   # diretorio alternativo

O run "ativo" e o JSONL modificado ha menos de ``--stale`` segundos; quando o
treino corrente para (early stop/fim), o dashboard continua exibindo-o marcado
como PARADO ate um novo comecar. Sai com 'q' (tty) ou Ctrl+C.
"""

import argparse
import select
import shutil
import sys
import termios
import time
import tty
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import dashboard as db  # noqa: E402

# Metricas plotadas por painel: (chave treino, chave validacao, rotulo)
LOSS_KEY = ("loss", "val_loss", "loss")
MAE_KEY = ("mae", "val_mae", "mae")

COLOR_GREEN = "\033[32m"
COLOR_RED = "\033[31m"
COLOR_YELLOW = "\033[33m"
COLOR_CYAN = "\033[36m"
COLOR_BOLD = "\033[1m"
COLOR_RESET = "\033[0m"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", default=None,
                        help="Filtra o run por substring do nome do arquivo.")
    parser.add_argument("--dir", type=Path, default=None,
                        help="Diretorio dos runs (default: TRAIN_DASHBOARD_DIR "
                             "ou pipeline/tmp/train_dashboard).")
    parser.add_argument("--interval", type=float, default=2.0,
                        help="Segundos entre atualizacoes (default: 2).")
    parser.add_argument("--stale", type=float, default=120.0,
                        help="Run sem atualizacao ha mais que isso (s) aparece "
                             "como PARADO (default: 120).")
    parser.add_argument("--list", action="store_true",
                        help="Lista os runs recentes e sai.")
    parser.add_argument("--once", action="store_true",
                        help="Desenha um snapshot e sai (sem loop).")
    return parser.parse_args()


def fmt(value, digits=5):
    if value is None:
        return "-"
    return f"{value:.{digits}f}"


def human_secs(seconds):
    seconds = int(seconds)
    if seconds >= 3600:
        return f"{seconds // 3600}h{(seconds % 3600) // 60:02d}m"
    if seconds >= 60:
        return f"{seconds // 60}m{seconds % 60:02d}s"
    return f"{seconds}s"


def pick_series(rows, keys):
    """Valores (treino, validacao) por epoca; None quando a chave falta."""
    train_key, val_key, _ = keys
    train = [row.get(train_key) for row in rows]
    val = [row.get(val_key) for row in rows]
    if all(v is None for v in train):
        return [], []
    return train, val


def nan_safe(values):
    """Substitui buracos por NaN para o plotext nao quebrar."""
    return [float("nan") if v is None else v for v in values]


def render(run_path, rows, fresh, term_size, args, other_runs):
    ended = db.last_status(rows)
    rows = db.epoch_rows(rows)
    width = max(40, term_size.columns - 1)
    body_height = max(12, term_size.lines - 12)
    name = run_path.stem if run_path else "-"

    # --- Cabecalho -----------------------------------------------------------
    print(COLOR_BOLD, end="")
    print("=" * width)
    print(f" {name}  ", end="")
    if not rows:
        print(f"{COLOR_YELLOW}sem epocas registradas{COLOR_RESET}")
    else:
        last = rows[-1]
        total = last.get("total_epochs")
        progress = f"{last['epoch']}/{total}" if total else str(last["epoch"])
        best = min((r["val_loss"] for r in rows if r.get("val_loss") is not None),
                   default=None)
        best_epoch = None
        if best is not None:
            best_epoch = next(r["epoch"] for r in rows if r.get("val_loss") == best)
        state = (f"{COLOR_GREEN}TREINANDO{COLOR_RESET}" if fresh
                 else f"{COLOR_RED}PARADO{COLOR_RESET}")
        if ended and not fresh:
            label = "early stop" if ended.get("status") == "early_stop" else "finalizado"
            state += f" ({label})"
        eta = ""
        if fresh and total and len(rows) < total:
            secs = [r["epoch_sec"] for r in rows[-10:] if r.get("epoch_sec")]
            if secs:
                remaining = (total - len(rows)) * (sum(secs) / len(secs))
                eta = f" | ETA ~{human_secs(remaining)}"
        print(f"{state} | epoca {progress}{eta}")
        print("-" * width)
        lr = last.get("lr")
        print(
            f" loss {COLOR_CYAN}{fmt(last.get('loss'))}{COLOR_RESET}"
            f" | val_loss {COLOR_CYAN}{fmt(last.get('val_loss'))}{COLOR_RESET}"
            f" | best val_loss {COLOR_CYAN}{fmt(best)}{COLOR_RESET}"
            f" (ep {best_epoch})"
            f" | mae {fmt(last.get('mae'))} / val {fmt(last.get('val_mae'))}"
            f" | lr {fmt(lr, 6)}"
        )
        last_ts = last.get("ts", "-")
        age = time.time() - run_path.stat().st_mtime
        print(f" ultima epoca: {last_ts} (ha {human_secs(age)})"
              f" | epocas: {len(rows)}"
              + (f" | filtro: '{args.run}'" if args.run else ""))
    print("=" * width, COLOR_RESET, sep="")

    # --- Graficos ------------------------------------------------------------
    import plotext as plt

    try:
        plt.theme("pro")
    except Exception:
        pass
    plt.clf()

    loss_train, loss_val = pick_series(rows, LOSS_KEY) if rows else ([], [])
    mae_train, mae_val = pick_series(rows, MAE_KEY) if rows else ([], [])
    lrs = [row.get("lr") for row in rows] if rows else []
    has_mae = any(v is not None for v in mae_train)
    has_lr = any(v is not None for v in lrs)

    # Frames empilhados (subplots do plotext 5.x se sobrepoem): loss grande,
    # depois mae e lr em frames menores, quando houver espaco no terminal.
    loss_height = max(9, int(body_height * 0.55))
    small_height = max(7, int(body_height * 0.22))

    def frame(title, series, ylabel, height):
        plt.clf()
        plt.plotsize(width, height)
        for values, label in series:
            plt.plot(nan_safe(values), label=label)
        plt.title(title)
        plt.ylabel(ylabel)
        return plt.build()

    frames = []
    if loss_train:
        frames.append(frame(
            "loss (horizon-weighted MSE)",
            [(loss_train, "loss")] + (
                [(loss_val, "val_loss")]
                if any(v is not None for v in loss_val) else []
            ),
            "loss", loss_height,
        ))
    else:
        plt.clf()
        plt.plotsize(width, loss_height)
        plt.title("aguardando epocas...")
        frames.append(plt.build())

    if has_mae and body_height >= loss_height + small_height:
        series = [(mae_train, "mae")]
        if any(v is not None for v in mae_val):
            series.append((mae_val, "val_mae"))
        frames.append(frame("mae", series, "mae", small_height))
    if has_lr and body_height >= loss_height + 2 * small_height:
        frames.append(frame("learning rate", [(lrs, "lr")], "lr", small_height))

    print("\n".join(frames))

    # --- Rodape: outros runs recentes -----------------------------------------
    others = [path.stem for path, _ in other_runs
              if not (run_path and path == run_path)][:6]
    hint = "'q' sai | Ctrl+C sai" if sys.stdin.isatty() else "Ctrl+C sai"
    print(f"{COLOR_BOLD}outros runs:{COLOR_RESET} "
          f"{', '.join(others) if others else '-'}   [{hint}]")


def wait_screen(term_size, message):
    print(COLOR_BOLD + "=" * max(40, term_size.columns - 1) + COLOR_RESET)
    print(f" {message}")


def key_pressed(timeout=0.05):
    """True se uma tecla foi pressionada (stdin em modo raw, apenas tty)."""
    if not sys.stdin.isatty():
        return False
    ready, _, _ = select.select([sys.stdin], [], [], timeout)
    return bool(ready)


def main():
    args = parse_args()

    if args.list:
        for path, mtime in db.list_runs(args.dir):
            age = human_secs(time.time() - mtime)
            fresh = "ativo " if (time.time() - mtime) <= args.stale else "parado"
            print(f"{fresh}  ha {age:>6}  {path.name}")
        return

    term_size = shutil.get_terminal_size()
    old_settings = None
    if sys.stdin.isatty():
        old_settings = termios.tcgetattr(sys.stdin)
        tty.setcbreak(sys.stdin)

    try:
        while True:
            term_size = shutil.get_terminal_size()
            run_path, fresh = db.pick_run(args.run, args.dir, max_age=args.stale)
            print("\033[2J\033[H", end="")  # clear + home
            if run_path is None:
                run_dir = Path(args.dir) if args.dir else db.dashboard_dir()
                wait_screen(
                    term_size,
                    f"nenhum run em {run_dir} — inicie um treino "
                    f"(ex.: python pipeline/pipeline.py --stage train) e o "
                    f"dashboard conecta sozinho... [q sai]",
                )
            else:
                rows = db.read_run(run_path)
                render(run_path, rows, fresh, term_size, args,
                       db.list_runs(args.dir))
            if args.once:
                return
            deadline = time.time() + args.interval
            while time.time() < deadline:
                if key_pressed(0.1):
                    ch = sys.stdin.read(1)
                    if ch in ("q", "Q", "\x03"):
                        return
                else:
                    time.sleep(0.05)
    except KeyboardInterrupt:
        pass
    finally:
        if old_settings is not None:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
        print("\033[2J\033[H", end="")


if __name__ == "__main__":
    main()
