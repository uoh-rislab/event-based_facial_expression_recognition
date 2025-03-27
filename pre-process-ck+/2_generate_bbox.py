import cv2
import os
from glob import glob

# Ruta base donde están los frames
input_root = '../data/ck+_frames_30fps'
bbox_output_root = '../output/ck+_frames_process_30fps/bboxes'
crop_output_root = '../output/ck+_frames_process_30fps/cropped_frames'
os.makedirs(bbox_output_root, exist_ok=True)
os.makedirs(crop_output_root, exist_ok=True)

# Cargar Haar cascade para detección de rostro
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Recorrer Train_Set y Test_Set
for split in ['Train_Set', 'Test_Set']:
    split_path = os.path.join(input_root, split)

    for class_dir in os.listdir(split_path):
        class_path = os.path.join(split_path, class_dir)
        if not os.path.isdir(class_path):
            continue

        for subject_dir in os.listdir(class_path):
            subject_path = os.path.join(class_path, subject_dir)
            if not os.path.isdir(subject_path):
                continue

            # Primer frame
            first_frame = os.path.join(subject_path, 'frame_0000.png')
            if not os.path.exists(first_frame):
                print(f"[WARN] No se encontró: {first_frame}")
                continue

            # Leer imagen y convertir a gris
            img = cv2.imread(first_frame)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Detección de rostro
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

            if len(faces) == 0:
                print(f"[WARN] No se detectó rostro en: {first_frame}")
                continue

            # Bounding box más grande
            x, y, w, h = max(faces, key=lambda b: b[2]*b[3])

            # Convertir a bbox cuadrado (1:1)
            cx, cy = x + w // 2, y + h // 2
            side = max(w, h)
            x_square = max(cx - side // 2, 0)
            y_square = max(cy - side // 2, 0)
            x_square = int(x_square)
            y_square = int(y_square)
            side = int(side)

            # Normalizar bbox respecto a tamaño de imagen
            img_h, img_w = img.shape[:2]
            x_norm = x_square / img_w
            y_norm = y_square / img_h
            w_norm = side / img_w
            h_norm = side / img_h

            # Guardar bbox normalizado en .txt
            bbox_txt_path = os.path.join(bbox_output_root, f"{subject_dir}.txt")
            with open(bbox_txt_path, 'w') as f:
                f.write(f"{x_norm:.6f} {y_norm:.6f} {w_norm:.6f} {h_norm:.6f}\n")

            print(f"[INFO] BBox guardado: {bbox_txt_path}")

            # Crear carpeta para imágenes recortadas
            cropped_subject_dir = os.path.join(crop_output_root, split, class_dir, subject_dir)
            os.makedirs(cropped_subject_dir, exist_ok=True)

            # Recorrer todos los frames del sujeto
            frame_paths = sorted(glob(os.path.join(subject_path, 'frame_*.png')))

            for frame_path in frame_paths:
                frame_img = cv2.imread(frame_path)

                # Cortar usando bbox detectado
                x1 = x_square
                y1 = y_square
                x2 = min(x1 + side, frame_img.shape[1])
                y2 = min(y1 + side, frame_img.shape[0])

                cropped = frame_img[y1:y2, x1:x2]

                # Guardar imagen recortada
                frame_name = os.path.basename(frame_path)
                cropped_path = os.path.join(cropped_subject_dir, frame_name)
                cv2.imwrite(cropped_path, cropped)

            print(f"[OK] Guardadas {len(frame_paths)} imágenes recortadas en {cropped_subject_dir}")
