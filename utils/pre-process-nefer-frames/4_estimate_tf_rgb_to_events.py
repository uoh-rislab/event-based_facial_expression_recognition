#!/usr/bin/env python3
# 4_estimate_tf_rgb_to_events.py

import os

def read_bbox_txt(path):
    """
    Lee un txt con una línea: x_norm y_norm w_norm h_norm
    y devuelve (x, y, w, h) como floats.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    with open(path, "r") as f:
        line = f.readline().strip()

    if not line:
        raise ValueError(f"Archivo vacío: {path}")

    parts = line.split()
    if len(parts) < 4:
        raise ValueError(f"Se esperaban 4 valores en {path}, encontrados: {len(parts)}")

    x, y, w, h = map(float, parts[:4])
    return x, y, w, h


def main():
    # Rutas (ajusta si cambias nombres)
    events_bbox_path = os.path.join("data", "events_frame_55_bbox.txt")
    rgb_bbox_path    = os.path.join("data", "rgb_frame_50_bbox.txt")

    # Leer bboxes normalizados
    x_e, y_e, w_e, h_e = read_bbox_txt(events_bbox_path)
    x_r, y_r, w_r, h_r = read_bbox_txt(rgb_bbox_path)

    print("[INFO] BBox eventos (norm):", x_e, y_e, w_e, h_e)
    print("[INFO] BBox RGB    (norm):", x_r, y_r, w_r, h_r)

    # Escalas (idealmente muy similares)
    s_w = w_e / w_r
    s_h = h_e / h_r
    s = 0.5 * (s_w + s_h)  # promedio para robustez

    # Translaciones usando esquina superior izquierda
    tx = x_e - s * x_r
    ty = y_e - s * y_r

    print("\n[RESULTADO] Transformación aproximada rgb → eventos (coords normalizadas):")
    print(f"  x_events ≈ {s:.6f} * x_rgb + {tx:.6f}")
    print(f"  y_events ≈ {s:.6f} * y_rgb + {ty:.6f}")

    # Chequeo rápido: aplicar tf al bbox RGB y comparar con eventos
    x_r_hat = s * x_r + tx
    y_r_hat = s * y_r + ty
    w_r_hat = s * w_r
    h_r_hat = s * h_r

    print("\n[CHEQUEO] BBox RGB transformado a espacio eventos:")
    print(f"  x_hat={x_r_hat:.6f}, y_hat={y_r_hat:.6f}, "
          f"w_hat={w_r_hat:.6f}, h_hat={h_r_hat:.6f}")
    print("[CHEQUEO] BBox eventos real:")
    print(f"  x_e  ={x_e:.6f}, y_e  ={y_e:.6f}, "
          f"w_e  ={w_e:.6f}, h_e  ={h_e:.6f}")

    # Guardar tf en txt
    tf_path = os.path.join("data", "rgb_to_events_tf.txt")
    with open(tf_path, "w") as f:
        # Formato: s tx ty
        f.write(f"{s:.9f} {tx:.9f} {ty:.9f}\n")
        f.write("# x_events = s * x_rgb + tx\n")
        f.write("# y_events = s * y_rgb + ty\n")

    print(f"\n[OK] Transformación guardada en: {tf_path}")


if __name__ == "__main__":
    main()
