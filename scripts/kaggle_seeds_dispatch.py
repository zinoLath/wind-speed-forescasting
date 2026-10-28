"""Dispatch das seeds para as GPUs do kernel Kaggle (1 ou 2 GPUs).

Com 2 GPUs visíveis, divide as seeds em duas metades rodando em paralelo
(uma por GPU); com 1 GPU, roda tudo em fila. Grava em
``<output-dir>/gpu<N>/<wrapper>/`` (ou direto em <output-dir>/<wrapper>/
no modo sequencial) para o collect consolidar depois.
"""

import argparse
import os
import subprocess
import sys
import time
import urllib.request


def notify(ntfy, text):
    if not ntfy:
        return
    try:
        req = urllib.request.Request(
            ntfy, data=text.encode("utf-8"), method="POST")
        urllib.request.urlopen(req, timeout=10)
    except Exception as exc:  # noqa: BLE001 - notificação é best-effort
        print(f"[ntfy falhou: {exc}]", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--wrappers", required=True, nargs="+")
    ap.add_argument("--seeds", required=True,
                    help="faixa '1-10' ou lista '1 2 3'")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--ntfy", default="")
    args = ap.parse_args()

    if "-" in args.seeds:
        lo, hi = args.seeds.split("-")
        seeds = list(range(int(lo), int(hi) + 1))
    else:
        seeds = [int(s) for s in args.seeds.split()]

    import tensorflow as tf
    gpus = len(tf.config.list_physical_devices("GPU"))
    print(f"[dispatch] GPUs visíveis: {gpus} | wrappers: {args.wrappers} | "
          f"seeds: {seeds}", flush=True)
    notify(args.ntfy, f"dispatch: {gpus} GPU(s), "
           f"wrappers {args.wrappers}, {len(seeds)} seeds")

    halves = ([seeds[:len(seeds) // 2], seeds[len(seeds) // 2:]]
              if gpus >= 2 else [seeds])
    procs = []
    started = time.time()
    for gpu, half in enumerate(halves):
        out = f"{args.output_dir}/gpu{gpu}" if gpus >= 2 else args.output_dir
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
        cmd = [sys.executable, "scripts/lidar_8020_sbsr_seeds_train.py",
               "--wrappers", *args.wrappers, "--seeds", *[str(s) for s in half],
               "--output-dir", out]
        print(f"[dispatch] GPU{gpu}: seeds {half} -> {out}", flush=True)
        procs.append(subprocess.Popen(cmd, env=env))
    codes = [p.wait() for p in procs]
    elapsed = (time.time() - started) / 60
    ok = all(c == 0 for c in codes)
    print(f"[dispatch] terminou em {elapsed:.0f} min | códigos: {codes}",
          flush=True)
    notify(args.ntfy, f"dispatch {'OK' if ok else 'FALHOU'} "
           f"({args.wrappers}) em {elapsed:.0f} min, códigos {codes}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
