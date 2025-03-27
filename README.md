# Event-based Facial Expression Recognition

## Vision Transformer (ViT) for Facial Expression Recognition with Event-Based Cameras

This repository contains a training script (`vit_train.py`) for a Vision Transformer (`vit_b_16`) model using grayscale images generated from an event-based camera. The goal is to classify facial expressions in the CK+ dataset, preprocessed into frame sequences.

### Dataset Structure

The dataset must be organized as follows:

```
output/e-ck+_frames_process_100fps/
├── Train_Set/
│   ├── 0/  # Class: Anger
│   ├── 1/  # Class: Contempt
│   ├── 2/  # Class: Disgust
│   ├── 3/  # Class: Fear
│   ├── 4/  # Class: Happy
│   ├── 5/  # Class: Sadness
│   └── 6/  # Class: Surprise
└── Test_Set/
    ├── 0/
    ├── 1/
    └── ...
```

Each subfolder contains `.png` grayscale event frames corresponding to a specific facial expression class.

### Installation

Make sure you have Python 3.7+ installed and install the required dependencies:

```bash
pip install torch torchvision tqdm matplotlib seaborn scikit-learn tensorboard
```

### Training

Run the training script:

```bash
python vit_train.py
```

A result directory will be automatically created under:

```
results/vit_e-ckplus_100fps_<timestamp>/
```

This directory will contain:

- `best_model_vit.pth`: Best model based on validation accuracy
- `train_loss.png`, `accuracy.png`: Training/validation plots
- `confusion_matrix_best_model.png`: Final normalized confusion matrix
- Raw and normalized `.txt` confusion matrix files
- TensorBoard logs
- Training metrics and hyperparameters in `.txt` and `.json` formats

### Hyperparameters

- Model: `vit_b_16`
- Input size: 224x224 pixels (grayscale converted to RGB if pretrained)
- Pretrained: ImageNet (`use_pretrained=True`)
- Optimizer: AdamW
- Learning rate: 1e-4
- Epochs: 20
- Batch size: 4
- Augmentations: Random crop, flip, rotation

### TensorBoard

Launch TensorBoard to monitor training:

```bash
tensorboard --logdir results/vit_e-ckplus_100fps_<timestamp>/tensorboard
```

### Output Example

```
results/
└── vit_e-ckplus_100fps_YYYYMMDD_HHMMSS/
    ├── best_model_vit.pth
    ├── train_loss.txt
    ├── train_acc.txt
    ├── val_acc.txt
    ├── accuracy.png
    ├── train_loss.png
    ├── confusion_matrix_best_model.png
    ├── confusion_matrix_raw.txt
    ├── confusion_matrix_normalized.txt
    ├── hyperparameters.json
    ├── model_architecture.txt
    └── tensorboard/
```

## License

This project is intended for academic and research use.
