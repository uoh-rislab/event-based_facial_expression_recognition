import cv2
import os
from glob import glob

# Paths
input_root = '../data/ck+_videos/'
output_root = '../output/ck+_frames_30fps'

# Recorre Train_Set y Test_Set
for split in ['Train_Set', 'Test_Set']:
    split_path = os.path.join(input_root, split)

    for class_dir in os.listdir(split_path):
        class_path = os.path.join(split_path, class_dir)
        if not os.path.isdir(class_path):
            continue

        for video_path in glob(os.path.join(class_path, '*.avi')):
            base_name = os.path.splitext(os.path.basename(video_path))[0]
            relative_path = os.path.relpath(video_path, input_root)
            output_dir = os.path.join(output_root, os.path.dirname(relative_path), base_name)
            os.makedirs(output_dir, exist_ok=True)

            # Abrir video
            cap = cv2.VideoCapture(video_path)

            # Ignorar el primer frame
            cap.read()

            frame_idx = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                frame_name = f'frame_{frame_idx:04d}.png'
                cv2.imwrite(os.path.join(output_dir, frame_name), gray_frame)

                frame_idx += 1

            cap.release()