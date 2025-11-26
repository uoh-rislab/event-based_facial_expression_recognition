#!/usr/bin/env python3
# refined_face_bbox.py

import cv2
import numpy as np
import os

def find_face_bbox_refined(
    img_path,
    perc_x=97,
    perc_y=92,
    expand=1.02,
    enforce_square=True
):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(img_path)

    h, w = img.shape[:2]

    ys, xs = np.where(img > 0)
    if len(xs) < 50:
        print("Muy pocos puntos.")
        return None

    cx = xs.mean()
    cy = ys.mean()

    dx = np.abs(xs - cx)
    dy = np.abs(ys - cy)

    rx = np.percentile(dx, perc_x)
    ry = np.percentile(dy, perc_y)

    mask = (dx <= rx) & (dy <= ry)
    xs_f = xs[mask]
    ys_f = ys[mask]

    if len(xs_f) < 50:
        print("Muy pocos puntos tras refinamiento.")
        return None

    x_min = xs_f.min()
    x_max = xs_f.max()
    y_min = ys_f.min()
    y_max = ys_f.max()

    bw = (x_max - x_min) * expand
    bh = (y_max - y_min) * expand
    cx_box = (x_min + x_max) / 2
    cy_box = (y_min + y_max) / 2

    x_min2 = int(max(0, cx_box - bw/2))
    x_max2 = int(min(w-1, cx_box + bw/2))
    y_min2 = int(max(0, cy_box - bh/2))
    y_max2 = int(min(h-1, cy_box + bh/2))
    # ============================================================
    shift = 15

    # Mover hacia arriba
    y_min2 = max(0, y_min2 - shift)

    # Expandir hacia la derecha
    x_max2 = min(w-1, x_max2 + shift)

    # ============================================================
    # === SHIFT FINAL: Mover el bbox sin modificar el tamaño =====
    # ============================================================

    shift_x = +10   # positivo = mover DERECHA
    shift_y = -30   # negativo = mover ARRIBA

    # Mover todo el bbox
    x_min2 = x_min2 + shift_x
    x_max2 = x_max2 + shift_x
    y_min2 = y_min2 + shift_y
    y_max2 = y_max2 + shift_y

    # Clampear para no salir de la imagen
    if x_min2 < 0:
        x_max2 -= x_min2
        x_min2 = 0

    if y_min2 < 0:
        y_max2 -= y_min2
        y_min2 = 0

    if x_max2 > w - 1:
        overflow = x_max2 - (w - 1)
        x_min2 -= overflow
        x_max2 = w - 1

    if y_max2 > h - 1:
        overflow = y_max2 - (h - 1)
        y_min2 -= overflow
        y_max2 = h - 1


    # ============================================================
    # === Forzar cuadrado recortando por abajo (como antes) ======
    # ============================================================
    if enforce_square:
        bw_final = x_max2 - x_min2
        bh_final = y_max2 - y_min2

        if bh_final > bw_final:
            new_y_max = y_min2 + bw_final
            if new_y_max > h - 1:
                new_y_max = h - 1
                y_min2 = max(0, new_y_max - bw_final)
            y_max2 = int(new_y_max)

    return x_min2, y_min2, x_max2, y_max2


if __name__ == "__main__":
    img_path = "data/events_frame_55.png"

    bbox = find_face_bbox_refined(
        img_path,
        perc_x=96,
        perc_y=92,
        expand=1.02,
        enforce_square=True
    )

    print("Refined BBOX:", bbox)

    if bbox:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        x1, y1, x2, y2 = bbox

        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 2)
        out_img_path = "events_frame_55_bbox_refined_square.png"
        cv2.imwrite(out_img_path, vis)
        print("[OK] Imagen con bbox guardada:", out_img_path)

        # Guardar bbox normalizado
        h, w = img.shape[:2]
        x_norm = x1 / w
        y_norm = y1 / h
        w_norm = (x2 - x1) / w
        h_norm = (y2 - y1) / h

        bbox_txt_path = "data/events_frame_55_bbox.txt"
        with open(bbox_txt_path, "w") as f:
            f.write(f"{x_norm:.6f} {y_norm:.6f} {w_norm:.6f} {h_norm:.6f}\n")

        print("[OK] BBox guardado en:", bbox_txt_path)
