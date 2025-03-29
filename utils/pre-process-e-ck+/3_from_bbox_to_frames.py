import os
import cv2
from glob import glob

# Rutas
bbox_dir = '../../ddbb_cropped/ck+_frames_process_30fps/bboxes'           # .txt con bbox normalizados
input_root = '../../data/e-ck+_events_npy_30fps_t_t'                         # imágenes completas
output_root = '../../ddbb_cropped/e-ck+_frames_process_30fps_t_t'             # solo imágenes recortadas
os.makedirs(output_root, exist_ok=True)

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

            # Leer bbox normalizado desde archivo
            bbox_txt = os.path.join(bbox_dir, f"{subject_dir}.txt")
            if not os.path.exists(bbox_txt):
                print(f"[WARN] BBox no encontrado para {subject_dir}")
                continue

            with open(bbox_txt, 'r') as f:
                line = f.readline().strip()
                x_norm, y_norm, w_norm, h_norm = map(float, line.split())

            # Crear carpeta de salida para recortes
            cropped_subject_dir = os.path.join(output_root, split, class_dir, subject_dir)
            os.makedirs(cropped_subject_dir, exist_ok=True)

            # Recorrer los frames
            frame_paths = sorted(glob(os.path.join(subject_path, 'frame_*.jpg')))
            for frame_path in frame_paths:
                img = cv2.imread(frame_path)
                if img is None:
                    print(f"[WARN] No se pudo leer: {frame_path}")
                    continue

                img_h, img_w = img.shape[:2]

                # Convertir bbox a coordenadas absolutas
                x = int(x_norm * img_w)
                y = int(y_norm * img_h)
                w = int(w_norm * img_w)
                h = int(h_norm * img_h)

                x2 = min(x + w, img_w)
                y2 = min(y + h, img_h)

                cropped = img[y:y2, x:x2]

                # Guardar imagen recortada
                frame_name = os.path.basename(frame_path)
                output_path = os.path.join(cropped_subject_dir, frame_name)
                cv2.imwrite(output_path, cropped)

            print(f"[OK] Recortes guardados para {subject_dir} en {cropped_subject_dir}")
