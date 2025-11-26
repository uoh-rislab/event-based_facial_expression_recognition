#!/usr/bin/env python3
# detect_rgb_frame_50_bbox.py

import os
import cv2

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

    # Clamp dentro de la imagen
    H, W = img_bgr.shape[:2]
    if x_sq + side > W:
        side = W - x_sq
    if y_sq + side > H:
        side = H - y_sq

    return int(x_sq), int(y_sq), int(side)


def main():
    # Ruta relativa desde tu carpeta actual
    rgb_path = os.path.join("data", "rgb_frame_50.png")
    if not os.path.exists(rgb_path):
        raise FileNotFoundError(rgb_path)

    # Cargar imagen
    img = cv2.imread(rgb_path)
    if img is None:
        raise RuntimeError(f"No se pudo leer la imagen: {rgb_path}")

    H, W = img.shape[:2]

    # Cargar Haar cascade
    face_cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    if not os.path.exists(face_cascade_path):
        raise FileNotFoundError(f"No se encontró Haar cascade en: {face_cascade_path}")
    face_cascade = cv2.CascadeClassifier(face_cascade_path)

    # Detectar bbox cuadrado
    bbox_sq = detect_face_bbox_square(img, face_cascade, scaleFactor=1.1, minNeighbors=5)
    if bbox_sq is None:
        print("[WARN] No se detectó rostro en rgb_frame_50.png")
        return

    x_sq, y_sq, side = bbox_sq
    print(f"[INFO] BBox píxeles: x={x_sq}, y={y_sq}, side={side}")

    # BBox normalizado (estilo NEFER que usabas: x, y, w, h relativos)
    x_norm = x_sq / W
    y_norm = y_sq / H
    w_norm = side / W
    h_norm = side / H
    print(f"[INFO] BBox normalizado: {x_norm:.6f} {y_norm:.6f} {w_norm:.6f} {h_norm:.6f}")

    # Guardar txt (por ejemplo en data/rgb_frame_50_bbox.txt)
    bbox_txt_path = os.path.join("data", "rgb_frame_50_bbox.txt")
    with open(bbox_txt_path, "w") as f:
        f.write(f"{x_norm:.6f} {y_norm:.6f} {w_norm:.6f} {h_norm:.6f}\n")
    print(f"[OK] BBox normalizado guardado en: {bbox_txt_path}")

    # Imagen de depuración
    vis = img.copy()
    x1, y1 = x_sq, y_sq
    x2, y2 = x_sq + side, y_sq + side
    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 2)

    out_debug = os.path.join("", "rgb_frame_50_bbox_debug.png")
    cv2.imwrite(out_debug, vis)
    print(f"[OK] Imagen con bbox guardada en: {out_debug}")


if __name__ == "__main__":
    main()
