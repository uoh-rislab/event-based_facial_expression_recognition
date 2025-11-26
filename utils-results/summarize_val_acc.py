#!/usr/bin/env python3
import os
import re

BASE_DIR = "/app/results"
MODEL_DIRS = ["tie-unet", "tie-unet-lstm"]

# Variantes en el orden que quieres como columnas
VARIANTS = ["t_t", "t_t0", "t0_t", "t0_t0"]
VARIANT_LABELS = {
    "t_t": "t-t",
    "t_t0": "t-t0",
    "t0_t": "t0-t",
    "t0_t0": "t0-t0",
}

# Modos en el orden que quieres como filas
MODES = ["scratch", "linear", "finetune"]
MODE_LABELS = {
    "scratch": "Scratch",
    "linear": "Transferred (linear)",
    "finetune": "Finetuning",
}

# Regex para extraer variante y modo del nombre de la carpeta
# Ej: unet_e-ck+_frames_process_30fps_t0_t0_finetune_20251126_072721
RUN_PATTERN = re.compile(
    r".*_(t0?_t0?|t0?_t|t_t0|t_t)_(scratch|linear|finetune)_\d+"
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

def collect_results_for_model(model_dir):
    """
    Recorre el árbol de directorios de un modelo (tie-unet o tie-unet-lstm)
    y devuelve un diccionario:
        results[modo][variante] = mejor_accuracy (float o None)
    """
    root_dir = os.path.join(BASE_DIR, model_dir)

    # Inicializamos con None para todas las combinaciones
    results = {m: {v: None for v in VARIANTS} for m in MODES}

    for dirpath, dirnames, filenames in os.walk(root_dir):
        if "val_acc.txt" not in filenames:
            continue

        run_folder = os.path.basename(dirpath)
        m = RUN_PATTERN.match(run_folder)
        if not m:
            # Si no calza el patrón, lo ignoramos (por si hay carpetas raras)
            continue

        variant, mode = m.groups()  # ej. "t0_t0", "finetune"
        val_acc_path = os.path.join(dirpath, "val_acc.txt")
        best_acc = read_best_val_acc(val_acc_path)

        if best_acc is None:
            continue

        # Guardamos el máximo por combinación (en caso de múltiples runs)
        current = results[mode][variant]
        if current is None or best_acc > current:
            results[mode][variant] = best_acc

    return results

def format_table_as_text(model_name, results):
    """
    Genera un bloque de texto con la tabla para un modelo.
    results: diccionario results[modo][variante] = accuracy
    """
    lines = []
    lines.append(f"Model: {model_name}")
    lines.append("=" * (len(lines[-1])))

    # Encabezado
    header = ["Mode/Variant"]
    header.extend([VARIANT_LABELS[v] for v in VARIANTS])
    lines.append(" | ".join(header))
    lines.append("-" * (len(lines[-1])))

    # Filas
    for mode in MODES:
        row = [MODE_LABELS[mode]]
        for variant in VARIANTS:
            acc = results[mode][variant]
            if acc is None:
                cell = "N/A"
            else:
                cell = f"{acc:.2f}"
            row.append(cell)
        lines.append(" | ".join(row))

    lines.append("")  # línea en blanco al final
    return "\n".join(lines)

def main():
    output_path = os.path.join(BASE_DIR, "summary_val_acc.txt")

    all_blocks = []

    for model_dir in MODEL_DIRS:
        # Nombre amigable para el encabezado
        if model_dir == "tie-unet":
            pretty_name = "TIE-UNET (sin LSTM)"
        elif model_dir == "tie-unet-lstm":
            pretty_name = "TIE-UNET-LSTM"
        else:
            pretty_name = model_dir

        results = collect_results_for_model(model_dir)
        block = format_table_as_text(pretty_name, results)
        all_blocks.append(block)

    with open(output_path, "w") as f:
        f.write("\n".join(all_blocks))

    print(f"Resumen guardado en: {output_path}")

if __name__ == "__main__":
    main()
