"""Diebold–Mariano POR HORIZONTE entre duas rodadas (mesma grade de origens).

Avalia dois diretórios de modelos na MESMA grade de origens (mesmo split 80/20)
e compara, horizonte a horizonte, os erros via teste de Diebold–Mariano (HAC
com defasagem h−1, correção HLN). Convenção: DM(A, B) positivo ⇒ o modelo de A
tem perda maior (pior) naquele horizonte.

Uso (default: HP antigos × HP do Optuna, 4 wrappers):
    python studies/dm_vs_hpo.py
    python studies/dm_vs_hpo.py --a pipeline/tmp/trained_8020 \
        --b pipeline/tmp/trained_8020_tcn_cprime \
        --wrappers tcn tcn_bi --out data/results/dm_dilations \
        --a-label "dil. legado" --b-label "dil. C′"

Saída: <out>/dm_matrix.json + <out>/figures/dm_heatmap.png
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
for path in (PROJECT_ROOT, PROJECT_ROOT / "reports" / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import relatorio_resultados as rr  # noqa: E402
from src import common  # noqa: E402
from src.utils import diebold_mariano  # noqa: E402

WRAPPER_META = {
    "lstm": "Seq2Seq_LSTM", "lstm_bi": "Seq2Seq_LSTM_Bidirectional",
    "tcn": "Seq2Seq_TCN", "tcn_bi": "Seq2Seq_TCN_Bidirectional",
}
LABELS = {"Seq2Seq_LSTM": "LSTM", "Seq2Seq_LSTM_Bidirectional": "LSTM-Bi",
          "Seq2Seq_TCN": "TCN", "Seq2Seq_TCN_Bidirectional": "TCN-Bi"}


def error_matrices(models_dir, dataset):
    """{nome_modelo: matriz origin×horizonte de erros (actual_raw − pred)}."""
    rr.MODELS_DIR = models_dir
    print(f"\n=== avaliando {models_dir} ===")
    _results, extras = rr.evaluate_models(dataset)
    out = {}
    for name, ex in extras.items():
        e = ex["all_df"]["actual_raw"] - ex["all_df"]["predicted"]
        out[name] = (ex["all_df"].assign(e=e)
                     .pivot(index="origin", columns="horizon", values="e")
                     .sort_index().sort_index(axis=1).dropna(axis=0))
    return out


def dm_per_horizon(A, B):
    common_idx = A.index.intersection(B.index).sort_values()
    a = A.loc[common_idx].to_numpy(float)
    b = B.loc[common_idx].to_numpy(float)
    dm, p = [], []
    for j in range(a.shape[1]):
        res = diebold_mariano(a[:, j], b[:, j], h=j + 1, power=2)
        dm.append(float(res["dm"]))
        p.append(float(res["p_value"]))
    return dm, p, len(common_idx)


def heatmap(matrix, labels, horizons, out_png, title, a_label, b_label):
    fig, ax = plt.subplots(figsize=(11, max(2.2, 0.7 * len(labels) + 1.4)))
    m = np.array(matrix, dtype=float)
    vmax = np.nanmax(np.abs(m)) or 1.0
    im = ax.imshow(m, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xticks(range(0, len(horizons), 3),
                  [str(h) for h in horizons[::3]])
    ax.set_xlabel("horizonte (passos de 10 min; 36 = 6 h)")
    ax.set_title(title, fontsize=11)
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            if abs(m[i, j]) >= 2.0:
                ax.text(j, i, "*", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax,
                 label=f"DM({a_label}, {b_label})  +  ⇒ {a_label} pior")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--a", default="pipeline/tmp/trained_8020")
    parser.add_argument("--b", default="pipeline/tmp/trained_8020_optuna")
    parser.add_argument("--wrappers", nargs="+",
                        default=["lstm", "lstm_bi", "tcn", "tcn_bi"],
                        choices=list(WRAPPER_META))
    parser.add_argument("--out", default="data/results/dm_vs_hpo")
    parser.add_argument("--a-label", default="antigo")
    parser.add_argument("--b-label", default="hpo")
    args = parser.parse_args()

    dataset = common.load_dataset(rr.DATA_FILE,
                                  keep_raw=("ws40", "ws100", "v40"))
    a_dir = common.resolve(args.a)
    b_dir = common.resolve(args.b)
    errs = {"A": error_matrices(a_dir, dataset),
            "B": error_matrices(b_dir, dataset)}
    out_dir = common.resolve(args.out)

    dm_matrix, sig_matrix, used_labels = [], [], []
    payload = {"horizons": list(range(1, 37)), "pairs": {},
               "a_dir": args.a, "b_dir": args.b,
               "a_label": args.a_label, "b_label": args.b_label}
    for key in args.wrappers:
        name = WRAPPER_META[key]
        if name not in errs["A"] or name not in errs["B"]:
            print(f"[aviso] {name} ausente em uma das rodadas; pulando")
            continue
        dm, p, n = dm_per_horizon(errs["A"][name], errs["B"][name])
        label = LABELS.get(name, name)
        payload["pairs"][label] = {"dm": dm, "p": p, "n_origins": n,
                                   "sig_count": int(sum(pp < 0.05 for pp in p))}
        used_labels.append(label)
        dm_matrix.append(dm)
        sig_matrix.append([1 if pp < 0.05 else 0 for pp in p])
        print(f"[{label}] n={n} | |DM|>=2 em {sum(abs(v) >= 2 for v in dm)} "
              f"horizontes | significativos {sum(pp < 0.05 for pp in p)}/36")
    payload["dm_matrix"] = dm_matrix
    payload["sig_matrix"] = sig_matrix

    out_dir.mkdir(parents=True, exist_ok=True)
    common.write_json(out_dir / "dm_matrix.json", payload)
    heatmap(dm_matrix, used_labels, payload["horizons"],
            out_dir / "figures" / "dm_heatmap.png",
            f"Diebold–Mariano por horizonte: {args.a_label} × {args.b_label} "
            f"(+ ⇒ {args.a_label} pior; * ⇒ p<0,05)",
            args.a_label, args.b_label)
    print(f"\nOK -> {out_dir}")


if __name__ == "__main__":
    main()
