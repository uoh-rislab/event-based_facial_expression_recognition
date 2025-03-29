import os

def get_relative_paths_no_ext(root_dir):
    relative_paths = set()
    for base, _, files in os.walk(root_dir):
        for file in files:
            filename, _ = os.path.splitext(file)
            rel_dir = os.path.relpath(base, root_dir)
            relative_paths.add(os.path.join(rel_dir, filename))
    return relative_paths

def clean_non_matching_files(reference_dir, target_dir):
    reference_set = get_relative_paths_no_ext(reference_dir)

    for base, _, files in os.walk(target_dir):
        for file in files:
            filename, _ = os.path.splitext(file)
            rel_dir = os.path.relpath(base, target_dir)
            relative_path = os.path.join(rel_dir, filename)

            if relative_path not in reference_set:
                full_path = os.path.join(base, file)
                print(f"Eliminando: {full_path}")
                os.remove(full_path)

# Ejemplo de uso
if __name__ == "__main__":
    reference = '../../ddbb_cropped/e-ck+_frames_process_30fps'
    target = '../../ddbb_cropped/e-ck+_frames_process_30fps_t_t'
    clean_non_matching_files(reference, target)
