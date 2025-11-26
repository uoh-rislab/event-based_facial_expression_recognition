#!/usr/bin/env python3
import os
import cv2
import argparse
from glob import glob

def detect_face_bbox_square(img_bgr, face_cascade, scaleFactor=1.1, minNeighbors=5):
    """Devuelve bbox cuadrado (x, y, side) o None si no detecta."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=scaleFactor, minNeighbors=minNeighbors)
    if len(faces) == 0:
        return None

    # Tomar el rostro más grande
    x, y, w, h = max(faces, key=lambda b: b[2] * b[3])

    # Convertir a cuadrado centrado
    cx, cy = x + w // 2, y + h // 2
    side = max(w, h)
    x_sq = max(cx - side // 2, 0)
    y_sq = max(cy - side // 2, 0)

    # Clamp dentro de la imagen
    H, W = img_bgr.shape[:2]
    if x_sq + side > W:
        side = W - x_sq
    if y_sq + side > H:
        side = H - y_sq

    return int(x_sq), int(y_sq), int(side)

def process_sequence(seq_dir, bbox_txt_path, crop_dir, face_cascade, search_first_n=5):
    """Detecta bbox en primeros N frames y recorta toda la secuencia."""
    frame_paths = sorted(glob(os.path.join(seq_dir, "frame_*.png")))
    if not frame_paths:
        print(f"[WARN] Sin frames en: {seq_dir}")
        return 0, False

    # Intentar detección en los primeros N frames como fallback
    bbox_sq = None
    for fp in frame_paths[:search_first_n]:
        img = cv2.imread(fp)
        if img is None:
            continue
        bbox_sq = detect_face_bbox_square(img, face_cascade)
        if bbox_sq is not None:
            break

    if bbox_sq is None:
        print(f"[WARN] No se detectó rostro en los primeros {min(search_first_n, len(frame_paths))} frames: {seq_dir}")
        return 0, False

    x_sq, y_sq, side = bbox_sq

    # Guardar bbox normalizado (respecto del tamaño del frame usado para detectar)
    img_h, img_w = img.shape[:2]
    x_norm = x_sq / img_w
    y_norm = y_sq / img_h
    w_norm = side / img_w
    h_norm = side / img_h

    os.makedirs(os.path.dirname(bbox_txt_path), exist_ok=True)
    with open(bbox_txt_path, "w") as f:
        f.write(f"{x_norm:.6f} {y_norm:.6f} {w_norm:.6f} {h_norm:.6f}\n")
    print(f"[INFO] BBox guardado: {bbox_txt_path}")

    # Recortar todos los frames con el bbox detectado
    os.makedirs(crop_dir, exist_ok=True)
    saved = 0
    for fp in frame_paths:
        frame = cv2.imread(fp)
        if frame is None:
            continue
        x1, y1 = x_sq, y_sq
        x2, y2 = min(x1 + side, frame.shape[1]), min(y1 + side, frame.shape[0])
        crop = frame[y1:y2, x1:x2]
        out_path = os.path.join(crop_dir, os.path.basename(fp))
        cv2.imwrite(out_path, crop)
        saved += 1

    print(f"[OK] Guardadas {saved} imágenes recortadas en {crop_dir}")
    return saved, True

def main():
    parser = argparse.ArgumentParser(description="Extraer bboxes y recortes faciales en NEFER (rgb_frames).")
    parser.add_argument("--input-root", required=True,
                        help="Raíz de rgb_frames (p.ej. /media/ignacio/KINGSTON/nefer/rgb_frames)")
    parser.add_argument("--bbox-out", required=True,
                        help="Carpeta de salida para .txt de bboxes (se replicará estructura user_xx/seq.txt)")
    parser.add_argument("--crop-out", required=True,
                        help="Carpeta de salida para frames recortados (se replicará estructura user_xx/seq/)")
    parser.add_argument("--search-first-n", type=int, default=5,
                        help="Número de primeros frames para intentar detección si falla el primero.")
    parser.add_argument("--scale-factor", type=float, default=1.1, help="scaleFactor de Haar.")
    parser.add_argument("--min-neighbors", type=int, default=5, help="minNeighbors de Haar.")
    args = parser.parse_args()

    # Haar cascade
    face_cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    if not os.path.exists(face_cascade_path):
        raise FileNotFoundError(f"No se encontró Haar cascade en: {face_cascade_path}")
    face_cascade = cv2.CascadeClassifier(face_cascade_path)

    users = sorted([d for d in os.listdir(args.input_root) if d.startswith("user_") and os.path.isdir(os.path.join(args.input_root, d))])
    total_seqs = 0
    ok_seqs = 0
    total_crops = 0

    for user in users:
        user_dir = os.path.join(args.input_root, user)
        # Cada subcarpeta (02, 03, ..., 22) es una secuencia
        seqs = sorted([s for s in os.listdir(user_dir) if os.path.isdir(os.path.join(user_dir, s))])
        for seq in seqs:
            seq_dir = os.path.join(user_dir, seq)
            # Salidas
            bbox_txt_path = os.path.join(args.bbox_out, user, f"{seq}.txt")
            crop_dir = os.path.join(args.crop_out, user, seq)

            saved, ok = process_sequence(
                seq_dir, bbox_txt_path, crop_dir, face_cascade,
                search_first_n=args.search_first_n
            )
            total_seqs += 1
            total_crops += saved
            ok_seqs += int(ok)

    print(f"\n[RESUMEN] Secuencias procesadas: {total_seqs} | Con rostro: {ok_seqs} | Frames recortados: {total_crops}")

if __name__ == "__main__":
    main()
