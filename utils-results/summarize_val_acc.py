#!/usr/bin/env python3
import os
import re

BASE_DIR = "/app/results"
MODEL_DIRS = ["tie-unet", "tie-unet-lstm"]

VARIANTS = ["t_t", "t_t0", "t0_t", "t0_t0"]
VARIANT_LABELS = {"t_t": "t-t", "t_t0": "t-t0", "t0_t": "t0-t", "t0_t0": "t0-t0"}

MODES = ["scratch", "linear", "finetune", "from80", "from100"]
MODE_LABELS = {
    "scratch": "Scratch",
    "linear": "Transferred",
    "finetune": "Finetune",
    "from80": "From 80 Epochs",
    "from100": "From 100 Epochs",
}

RUN_PATTERN = re.compile(r".*_(t0?_t0?|t0?_t|t_t0|t_t)_(scratch|linear|finetune)_\d+")

def read_best_val_acc(path):
    best = None
    with open(path, "r") as f:
        for line in f:
            try:
                v = float(line.strip())
                if best is None or v > best:
                    best = v
            except:
                pass
    return best

def detect_epoch_type(path):
    if "weight_from_epoch_80" in path:
        return "from80"
    if "weight_from_epoch_100" in path:
        return "from100"
    return None

def collect_results(model):
    results = {m: {v: None for v in VARIANTS} for m in MODES}
    root = os.path.join(BASE_DIR, model)

    for dirpath, _, files in os.walk(root):
        if "val_acc.txt" not in files:
            continue

        folder = os.path.basename(dirpath)
        m = RUN_PATTERN.match(folder)
        if not m:
            continue

        variant, mode = m.groups()
        epoch_type = detect_epoch_type(dirpath)
        if epoch_type:
            mode = epoch_type  # override mode to epoch row

        best = read_best_val_acc(os.path.join(dirpath, "val_acc.txt"))
        if best is None:
            continue

        current = results[mode][variant]
        if current is None or best > current:
            results[mode][variant] = best

    return results

def format_table(model_name, results):
    lines = []
    lines.append(f"Model: {model_name}")
    lines.append("=" * len(lines[-1]))
    lines.append("Mode/Variant | " + " | ".join(VARIANT_LABELS[v] for v in VARIANTS))
    lines.append("-" * len(lines[-1]))

    for mode in MODES:
        row = [MODE_LABELS[mode]]
        for variant in VARIANTS:
            v = results[mode][variant]
            row.append("N/A" if v is None else f"{v:.1f}")
        lines.append(" | ".join(row))

    lines.append("")
    return "\n".join(lines)

def main():
    output_file = os.path.join(BASE_DIR, "summary_val_acc_v2.txt")
    blocks = []

    for model in MODEL_DIRS:
        pretty = "TIE-UNET (no LSTM)" if model == "tie-unet" else "TIE-UNET-LSTM"
        results = collect_results(model)
        blocks.append(format_table(pretty, results))

    with open(output_file, "w") as f:
        f.write("\n".join(blocks))

    print(f"Archivo generado: {output_file}")

if __name__ == "__main__":
    main()
