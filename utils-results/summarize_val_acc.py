#!/usr/bin/env python3
import os
import re

BASE_DIR = "/app/results"
MODEL_DIRS = ["tie-unet", "tie-unet-lstm"]

# Columnas (variantes)
VARIANTS = ["t_t", "t_t0", "t0_t", "t0_t0"]
VARIANT_LABELS = {
    "t_t": "t-t",
    "t_t0": "t-t0",
    "t0_t": "t0-t",
    "t0_t0": "t0-t0",
}

# Filas (modos)
MODE_ORDER = [
    "scratch",
    "linear",
    "finetune",
    "linear_from80",
    "finetune_from80",
    "linear_from100",
    "finetune_from100",
]

MODE_LABELS = {
    "scratch": "Scratch",
    "linear": "Transferred",
    "finetune": "Finetune",
    "linear_from80": "Transferred from 80 epochs",
    "finetune_from80": "Finetune from 80 epochs",
    "linear_from100": "Transferred from 100 epochs",
    "finetune_from100": "Finetune from 100 epochs",
}

# Ejemplo de nombre:
# unet_e-ck+_frames_process_30fps_t0_t0_finetune_20251126_072721
RUN_PATTERN = re.compile(
    r".*_(t0_t0|t0_t|t_t0|t_t)_(scratch|linear|finetune)_\d+"
)

def read_best_val_acc(path_txt):
    """Lee val_acc.txt y devuelve el valor máximo (float)."""
    best = None
    with open(path_txt, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                v = float(line)
            except ValueError:
                continue
            if best is None or v > best:
                best = v
    return best

def detect_epoch_stage(dirpath):
    """Devuelve '80', '100' o None dependiendo de la ruta."""
    if "weight_from_epoch_80" in dirpath:
        return "80"
    if "weight_from_epoch_100" in dirpath:
        return "100"
    return None

def collect_results_for_model(model_dir):
    """
    Recorre el árbol de directorios de un modelo (tie-unet o tie-unet-lstm)
    y devuelve:
        results[mode_key][variant] = best_acc
    donde mode_key puede ser:
        scratch, linear, finetune,
        linear_from80, finetune_from80,
        linear_from100, finetune_from100
    """
    root_dir = os.path.join(BASE_DIR, model_dir)

    # Inicializamos con None para las combinaciones conocidas
    results = {m: {v: None for v in VARIANTS} for m in MODE_ORDER}

    for dirpath, _, filenames in os.walk(root_dir):
        if "val_acc.txt" not in filenames:
            continue

        run_folder = os.path.basename(dirpath)
        m = RUN_PATTERN.match(run_folder)
        if not m:
            # Carpetas que no respeten el patrón se ignoran
            continue

        variant, base_mode = m.groups()  # ej. "t0_t0", "finetune"

        # Detectar si viene de weight_from_epoch_80 o 100
        stage = detect_epoch_stage(dirpath)
        if stage is None:
            mode_key = base_mode  # scratch / linear / finetune
        else:
            # Sólo tiene sentido linear / finetune desde 80/100
            mode_key = f"{base_mode}_from{stage}"

        # Si aparece un modo nuevo no previsto, lo creamos dinámicamente
        if mode_key not in results:
            results[mode_key] = {v: None for v in VARIANTS}

        val_acc_path = os.path.join(dirpath, "val_acc.txt")
        best_acc = read_best_val_acc(val_acc_path)
        if best_acc is None:
            continue

        current = results[mode_key].get(variant)
        if current is None or best_acc > current:
            results[mode_key][variant] = best_acc

    return results

def format_table_as_text(model_name, results):
    """
    Genera un bloque de texto con la tabla para un modelo.
    results[mode_key][variant] = accuracy
    """
    lines = []
    lines.append(f"Model: {model_name}")
    lines.append("=" * len(lines[-1]))

    # Encabezado
    header = ["Mode/Variant"]
    header.extend([VARIANT_LABELS[v] for v in VARIANTS])
    lines.append("Mode/Variant | " + " | ".join(header[1:]))
    lines.append("-" * len(lines[-1]))

    # Filas en orden fijo MODE_ORDER
    for mode_key in MODE_ORDER:
        label = MODE_LABELS.get(mode_key, mode_key)
        row_vals = []
        # Si un modo no existe en results, lo tratamos como todo N/A
        mode_dict = results.get(mode_key, {v: None for v in VARIANTS})
        for variant in VARIANTS:
            acc = mode_dict.get(variant)
            if acc is None:
                cell = "N/A"
            else:
                cell = f"{acc:.1f}"  # 1 decimal
            row_vals.append(cell)
        lines.append(label + " | " + " | ".join(row_vals))

    lines.append("")  # línea en blanco
    return "\n".join(lines)

def main():
    output_path = os.path.join(BASE_DIR, "summary_val_acc.txt")
    blocks = []

    for model_dir in MODEL_DIRS:
        if model_dir == "tie-unet":
            pretty_name = "TIE-UNET (no LSTM)"
        elif model_dir == "tie-unet-lstm":
            pretty_name = "TIE-UNET-LSTM"
        else:
            pretty_name = model_dir

        results = collect_results_for_model(model_dir)
        block = format_table_as_text(pretty_name, results)
        blocks.append(block)

    with open(output_path, "w") as f:
        f.write("\n".join(blocks))

    print(f"Resumen guardado en: {output_path}")

if __name__ == "__main__":
    main()
