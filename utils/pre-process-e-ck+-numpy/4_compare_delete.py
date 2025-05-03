import os
import numpy as np
from glob import glob

# Rutas
frames_root = '../../ddbb_cropped/e-ck+_frames_process_30fps'
numpys_root = '../../ddbb_cropped/e-ck+_numpys_process_30fps'

# Obtener nombres válidos desde carpeta de frames (sin extensión)
valid_names = set()

for split in ['Train_Set', 'Test_Set']:
    for class_id in os.listdir(os.path.join(frames_root, split)):
        class_path = os.path.join(frames_root, split, class_id)
        if not os.path.isdir(class_path):
            continue
        for subject in os.listdir(class_path):
            subject_path = os.path.join(class_path, subject)
            if not os.path.isdir(subject_path):
                continue
            for frame_file in os.listdir(subject_path):
                if frame_file.endswith('.png') or frame_file.endswith('.jpg'):
                    name_no_ext = os.path.splitext(frame_file)[0]
                    valid_names.add((split, class_id, subject, name_no_ext))

# Recorrer los npy y eliminar los que no estén en valid_names
for split in ['Train_Set', 'Test_Set']:
    for class_id in os.listdir(os.path.join(numpys_root, split)):
        class_path = os.path.join(numpys_root, split, class_id)
        if not os.path.isdir(class_path):
            continue
        for subject in os.listdir(class_path):
            subject_path = os.path.join(class_path, subject)
            if not os.path.isdir(subject_path):
                continue
            for npy_file in os.listdir(subject_path):
                if not npy_file.endswith('.npy'):
                    continue
                name_no_ext = os.path.splitext(npy_file)[0]
                key = (split, class_id, subject, name_no_ext)
                if key not in valid_names:
                    os.remove(os.path.join(subject_path, npy_file))
                    print(f"[REMOVIDO] {os.path.join(subject_path, npy_file)}")
