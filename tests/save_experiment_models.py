"""Salva os modelos dos experimentos da sessão em models/experiments/<condição>/.

Cada entrada mantém model.keras + model.json (fonte da verdade das features).
Uso: python tests/save_experiment_models.py
"""
import json
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TMP = PROJECT_ROOT / "pipeline" / "tmp"
DEST = PROJECT_ROOT / "models" / "experiments"

# (condição, base_dir, wrappers, descrição)
RUNS = [
    ("baseline_todas_features", "trained_compare",
     ["Seq2Seq_LSTM", "Seq2Seq_LSTM_Bidirectional", "Seq2Seq_TCN", "Seq2Seq_TCN_Bidirectional"],
     "level 1, todas as features, teacher forcing, gate off, ctx horizon"),
    ("tcnbi_huber", "trained_compare_tcnbi_huber",
     ["Seq2Seq_TCN_Bidirectional"],
     "idem baseline, loss huber no TCN_Bi"),
    ("sem_teacher_forcing", "trained_compare_noteaching",
     ["Seq2Seq_LSTM", "Seq2Seq_TCN_Bidirectional"],
     "decoder direct (sem teacher forcing)"),
    ("sem_pos_dropout", "trained_compare_nopostdrop",
     ["Seq2Seq_TCN"],
     "encoder/decoder post-dropout zerados"),
    ("wavelet_serie_inteira", "trained_compare_wavfull",
     ["Seq2Seq_LSTM", "Seq2Seq_TCN"],
     "wavelet aplicada na série inteira antes do split"),
    ("sem_time_distributed", "trained_compare_nodense",
     ["Seq2Seq_TCN", "Seq2Seq_TCN_Bidirectional"],
     "cabeça Flatten+Dense no lugar da TimeDistributed"),
]


def main():
    copied = []
    for condition, base, wrappers, desc in RUNS:
        src_base = TMP / base
        if not src_base.is_dir():
            print(f"[pulando] {base} não existe")
            continue
        for name in wrappers:
            src = src_base / name
            if not (src / "model.keras").is_file():
                print(f"[pulando] {base}/{name} sem model.keras")
                continue
            dst = DEST / condition / name
            dst.mkdir(parents=True, exist_ok=True)
            for f in ("model.keras", "model.json"):
                shutil.copy2(src / f, dst / f)
            copied.append(f"{condition}/{name}")
            print(f"salvo: experiments/{condition}/{name}")
    index = DEST / "index.json"
    manifest = {"runs": {c: d for c, _, _, d in RUNS}, "modelos": copied}
    json.dump(manifest, open(index, "w"), indent=1)
    print(f"\n{len(copied)} modelos salvos em {DEST} (manifest: index.json)")


if __name__ == "__main__":
    main()
