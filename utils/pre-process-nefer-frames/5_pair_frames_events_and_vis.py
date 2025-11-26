#!/usr/bin/env python3
# 5_pair_frames_events_and_vis.py

import os
import cv2
import numpy as np
from glob import glob

# ---------- Utilidades de bbox / tf ----------

def detect_face_bbox_square(img_bgr, face_cascade, scaleFactor=1.1, minNeighbors=5):
    """Devuelve bbox cuadrado (x, y, side) o None si no detecta."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=scaleFactor,
        minNeighbors=minNeighbors
    )
    if len(faces) == 0:
        return None

    # Tomar el rostro más grande
    x, y, w, h = max(faces, key=lambda b: b[2] * b[3])

    # Convertir a cuadrado centrado
    cx, cy = x + w // 2, y + h // 2
    side = max(w, h)
    x_sq = max(cx - side // 2, 0)
    y_sq = max(cy - side // 2, 0)

    H, W = img_bgr.shape[:2]
    if x_sq + side > W:
        side = W - x_sq
    if y_sq + side > H:
        side = H - y_sq

    return int(x_sq), int(y_sq), int(side)


def read_tf_rgb_to_events(tf_path):
    """
    Lee la primera línea no comentada de rgb_to_events_tf.txt
    con formato: s tx ty
    """
    if not os.path.exists(tf_path):
        raise FileNotFoundError(tf_path)

    with open(tf_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 3:
                raise ValueError(f"Línea inválida en {tf_path}: {line}")
            s, tx, ty = map(float, parts[:3])
            return s, tx, ty

    raise ValueError(f"No se encontró línea válida en {tf_path}")


def apply_tf_to_bbox_rgb_norm(x_norm, y_norm, side_norm, s, tx, ty):
    """
    Aplica la tf normalizada rgb -> eventos:
        x_e = s * x_r + tx
        y_e = s * y_r + ty
        side_e = s * side_r
    """
    x_e = s * x_norm + tx
    y_e = s * y_norm + ty
    side_e = s * side_norm
    return x_e, y_e, side_e


def clamp_bbox_square(x, y, side, W, H):
    """Ajusta bbox cuadrado para que quede dentro de [0,W)x[0,H)."""
    x = max(0, min(x, W - 1))
    y = max(0, min(y, H - 1))
    side = max(1, side)

    if x + side > W:
        side = W - x
    if y + side > H:
        side = H - y

    side = max(1, side)
    return int(x), int(y), int(side)


# ---------- Script principal ----------

def main():
    # Rutas base (ajústalas si cambian los nombres)
    samples_root = os.path.join("data", "samples")
    events_dir = os.path.join(samples_root, "events_user00_2022-06-08_12-00-41_cd")
    frames_dir = os.path.join(samples_root, "frames_08")

    out_crops_dir = os.path.join(samples_root, "paired_crops")
    out_full_dir  = os.path.join(samples_root, "paired_full")
    os.makedirs(out_crops_dir, exist_ok=True)
    os.makedirs(out_full_dir, exist_ok=True)

    # Leer tf rgb -> eventos
    tf_path = os.path.join("data", "rgb_to_events_tf.txt")
    s, tx, ty = read_tf_rgb_to_events(tf_path)
    print(f"[INFO] TF rgb->events: s={s:.6f}, tx={tx:.6f}, ty={ty:.6f}")

    # Haar cascade
    face_cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    if not os.path.exists(face_cascade_path):
        raise FileNotFoundError(f"No se encontró Haar cascade en: {face_cascade_path}")
    face_cascade = cv2.CascadeClassifier(face_cascade_path)

    # Listar archivos PNG
    frame_paths = sorted(glob(os.path.join(frames_dir, "frame_*.png")))
    event_paths = sorted(glob(os.path.join(events_dir, "frame_*.png")))

    if not frame_paths or not event_paths:
        print("[ERROR] No se encontraron PNG en alguna de las carpetas.")
        print("Frames dir:", frames_dir)
        print("Events dir:", events_dir)
        return

    num_pairs = min(len(frame_paths), len(event_paths))
    print(f"[INFO] Frames: {len(frame_paths)} | Events: {len(event_paths)} | Pairs: {num_pairs}")

    pair_idx = 0
    for i in range(num_pairs):
        f_path = frame_paths[i]
        e_path = event_paths[i]

        frame = cv2.imread(f_path)
        events = cv2.imread(e_path, cv2.IMREAD_GRAYSCALE)

        if frame is None or events is None:
            print(f"[WARN] No se pudo leer alguna imagen: {f_path}, {e_path}")
            continue

        H_r, W_r = frame.shape[:2]
        H_e, W_e = events.shape[:2]

        # 1) Detectar rostro en RGB con bbox cuadrado
        bbox_rgb = detect_face_bbox_square(frame, face_cascade)
        if bbox_rgb is None:
            print(f"[WARN] No se detectó rostro en {os.path.basename(f_path)}; se salta par.")
            continue

        x_r, y_r, side_r = bbox_rgb

        # 2) BBox RGB normalizado
        x_r_norm = x_r / W_r
        y_r_norm = y_r / H_r
        side_r_norm = side_r / W_r  # asumimos escala uniforme

        # 3) Aplicar tf a eventos
        x_e_norm, y_e_norm, side_e_norm = apply_tf_to_bbox_rgb_norm(
            x_r_norm, y_r_norm, side_r_norm, s, tx, ty
        )

        # 4) Pasar bbox eventos a píxeles
        x_e = int(round(x_e_norm * W_e))
        y_e = int(round(y_e_norm * H_e))
        side_e = int(round(side_e_norm * W_e))  # escala con ancho

        x_e, y_e, side_e = clamp_bbox_square(x_e, y_e, side_e, W_e, H_e)

        # ---------- Crear crops concatenados ----------
        # Crop RGB
        x2_r = x_r + side_r
        y2_r = y_r + side_r
        crop_rgb = frame[y_r:y2_r, x_r:x2_r]

        # Crop eventos (pasar a BGR para concatenar)
        x2_e = x_e + side_e
        y2_e = y_e + side_e
        crop_ev = events[y_e:y2_e, x_e:x2_e]
        crop_ev_bgr = cv2.cvtColor(crop_ev, cv2.COLOR_GRAY2BGR)

        # Redimensionar eventos al tamaño del crop RGB
        ch_r, cw_r = crop_rgb.shape[:2]
        crop_ev_resized = cv2.resize(crop_ev_bgr, (cw_r, ch_r), interpolation=cv2.INTER_NEAREST)

        concat_crops = np.hstack([crop_rgb, crop_ev_resized])

        # Guardar crops concatenados
        out_crop_name = f"pair_{pair_idx:03d}_crops.png"
        out_crop_path = os.path.join(out_crops_dir, out_crop_name)
        cv2.imwrite(out_crop_path, concat_crops)

        # ---------- Crear imágenes completas con bbox ----------
        frame_full_vis = frame.copy()
        events_full_vis = cv2.cvtColor(events, cv2.COLOR_GRAY2BGR)

        # Dibujar bbox en RGB
        cv2.rectangle(frame_full_vis, (x_r, y_r), (x_r + side_r, y_r + side_r), (0, 0, 255), 2)

        # Dibujar bbox en eventos
        cv2.rectangle(events_full_vis, (x_e, y_e), (x_e + side_e, y_e + side_e), (0, 0, 255), 2)

        # Redimensionar eventos completos al tamaño del frame RGB para concatenar
        events_full_resized = cv2.resize(
            events_full_vis,
            (frame_full_vis.shape[1], frame_full_vis.shape[0]),
            interpolation=cv2.INTER_NEAREST
        )

        concat_full = np.hstack([frame_full_vis, events_full_resized])

        out_full_name = f"pair_{pair_idx:03d}_full.png"
        out_full_path = os.path.join(out_full_dir, out_full_name)
        cv2.imwrite(out_full_path, concat_full)

        print(f"[OK] Par {pair_idx:03d} guardado:")
        print("     -", out_crop_path)
        print("     -", out_full_path)

        pair_idx += 1

    print(f"\n[RESUMEN] Pares procesados correctamente: {pair_idx}")


if __name__ == "__main__":
    main()
