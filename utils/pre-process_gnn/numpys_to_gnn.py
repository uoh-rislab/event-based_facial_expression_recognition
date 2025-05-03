import os
import numpy as np
import torch
from glob import glob
from torch_geometric.data import Data
from torch_cluster import radius_graph

# Parámetros
width, height = 346, 260
r = 0.05  # radio en coordenadas normalizadas
input_root = '../../ddbb_cropped/e-ck+_numpys_process_30fps'
output_root = '../../ddbb_graphs/e-ck+_graphs_30fps'
os.makedirs(output_root, exist_ok=True)

# Recorremos particiones
for split in ['Train_Set', 'Test_Set']:
    for class_dir in os.listdir(os.path.join(input_root, split)):
        class_path = os.path.join(input_root, split, class_dir)
        if not os.path.isdir(class_path):
            continue

        for subject_dir in os.listdir(class_path):
            subject_path = os.path.join(class_path, subject_dir)
            if not os.path.isdir(subject_path):
                continue

            # Carpeta de salida
            output_subject_path = os.path.join(output_root, split, class_dir, subject_dir)
            os.makedirs(output_subject_path, exist_ok=True)

            npy_files = sorted(glob(os.path.join(subject_path, 'frame_*.npy')))
            for npy_path in npy_files:
                events = np.load(npy_path)
                if events.shape[0] == 0:
                    continue

                # Extraer características
                x_pix = events[:, 0] / width
                y_pix = events[:, 1] / height
                t = events[:, 2]
                p = events[:, 3]

                t_rel = t - t.min()
                t_norm = t_rel / (t_rel.max() + 1e-6)

                node_features = np.stack([x_pix, y_pix, t_norm, p], axis=1)
                x = torch.tensor(node_features, dtype=torch.float)
                pos = torch.tensor(np.stack([x_pix, y_pix], axis=1), dtype=torch.float)
                edge_index = radius_graph(pos, r=r, loop=False)

                # Crear grafo
                data = Data(x=x, edge_index=edge_index)

                # Guardar
                frame_name = os.path.splitext(os.path.basename(npy_path))[0] + '.pt'
                output_path = os.path.join(output_subject_path, frame_name)
                torch.save(data, output_path)

                #break

            print(f"[OK] Grafos guardados en {output_subject_path}")

            #break
        #break
    #break
