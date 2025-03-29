import os
import shutil

# Ruta raíz del dataset de entrada
for tie in ['t_t', 't_t0', 't0_t', 't0_t0']:


    base_path = f"../../ddbb_cropped/e-ck+_frames_process_30fps_{tie}/"

    # Ruta donde se almacenarán los resultados
    output_path = f"../../ddbb_cropped/e-ck+_frames_lstm_process_30fps_{tie}/"

    # Longitud de la secuencia de frames (sliding window)
    seq_len = 3

    # Mapeo de particiones a carpetas (train, val, test)
    partition_map = {
        "Train_Set": "train", #"Train_Set",
        "Test_Set": "test", #"Test_Set",
    }

    # Iterar sobre todas las particiones (Train_Set, Test_Set, etc.)
    for partition in os.listdir(base_path):
        partition_path = os.path.join(base_path, partition)

        # Verificar si es un directorio válido
        if not os.path.isdir(partition_path):
            continue

        # Obtener el nombre de la partición mapeada (train, val, test)
        partition_name = partition_map.get(partition, None)
        if partition_name is None:
            continue

        # Iterar sobre las clases (1, 2, ..., 7)
        for class_id in os.listdir(partition_path):
            class_path = os.path.join(partition_path, class_id)
            
            # Verificar si es un directorio válido
            if not os.path.isdir(class_path):
                continue
            
            # Crear carpeta de salida para la clase
            class_output_path = os.path.join(output_path, partition_name, f"{class_id}")
            os.makedirs(class_output_path, exist_ok=True)

            # Iterar sobre los sujetos en cada clase
            for subject in os.listdir(class_path):
                subject_path = os.path.join(class_path, subject)
                
                # Verificar si es un directorio válido
                if not os.path.isdir(subject_path):
                    continue
                
                # Obtener todos los frames dentro del sujeto y ordenarlos
                frames = sorted([f for f in os.listdir(subject_path) if f.endswith(".jpg")])
                
                # Crear secuencias con sliding window de longitud 3
                for i in range(len(frames) - seq_len + 1):
                    # Usar el nombre del sujeto como base
                    subject_name = subject  # Ejemplo: S116_006
                    
                    # Agregar índice secuencial para evitar duplicados
                    seq_id = f"seq_{i+1:03d}"  # Ejemplo: seq_001, seq_002
                    
                    # Copiar los 3 frames consecutivos a la carpeta de salida
                    for j in range(seq_len):
                        src_frame_path = os.path.join(subject_path, frames[i + j])
                        frame_num = frames[i + j].split('_')[1].split('.')[0]  # Obtener número de frame
                        # Nombre del archivo corregido
                        dst_frame_name = f"{subject_name}_{seq_id}_frame_{frame_num}.jpg"
                        dst_frame_path = os.path.join(class_output_path, dst_frame_name)
                        shutil.copy(src_frame_path, dst_frame_path)

    print("✅ Creación de secuencias completada sin crear carpetas innecesarias.")
