#!/usr/bin/env python3
import os
import argparse
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

import numpy as np
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import json
from PIL import Image

# ============================================================
# 1) Dataset de secuencias: root_dir/{1..7}/S010_004_0001.jpg...
# ============================================================
class SequenceDataset(Dataset):
    """
    Dataset para secuencias de frames:
    root_dir/
      1/
        S010_004_0001.jpg
        S010_004_0002.jpg
        ...
      2/
      ...

    Devuelve:
      images: [seq_len, C, H, W]
      label:  índice [0..num_classes-1]
    """
    def __init__(self, root_dir, transform=None, seq_len=3):
        self.root_dir = root_dir
        self.transform = transform
        self.seq_len = seq_len
        self.samples = []

        # clases (carpetas) ordenadas
        class_names = [
            d for d in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, d))
        ]
        class_names = sorted(class_names)
        self.class_names = class_names
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(class_names)}

        for class_name in class_names:
            class_path = os.path.join(root_dir, class_name)
            class_idx = self.class_to_idx[class_name]

            # agrupar por sujeto/expresión, ej: S010_004
            seq_dict = {}
            for frame_name in sorted(os.listdir(class_path)):
                if not frame_name.lower().endswith((".jpg", ".png")):
                    continue
                subject_name = "_".join(frame_name.split("_")[:2])  # S010_004
                if subject_name not in seq_dict:
                    seq_dict[subject_name] = []
                seq_dict[subject_name].append(os.path.join(class_path, frame_name))

            for _, frame_paths in seq_dict.items():
                if len(frame_paths) >= self.seq_len:
                    for i in range(len(frame_paths) - self.seq_len + 1):
                        seq_frames = frame_paths[i:i + self.seq_len]
                        self.samples.append((seq_frames, class_idx))

        print(f"[SequenceDataset] root={root_dir} | classes={self.class_names} | samples={len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        frame_paths, label = self.samples[idx]
        images = []
        for fp in frame_paths:
            img = Image.open(fp).convert("RGB")
            if self.transform is not None:
                img = self.transform(img)
            images.append(img)
        images = torch.stack(images, dim=0)  # [seq_len, C, H, W]
        return images, label


# ============================================================
# 2) Pix2Pix U-Net (igual que tu script original)
# ============================================================
class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, batchnorm=True):
        super(EncoderBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
        self.bn = nn.BatchNorm2d(out_channels) if batchnorm else None
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        x = self.lrelu(x)
        return x


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=True):
        super(DecoderBlock, self).__init__()
        self.deconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout(0.5) if dropout else None
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, skip_connection):
        x = self.deconv(x)
        x = self.bn(x)
        if self.dropout is not None:
            x = self.dropout(x)
        x = torch.cat([x, skip_connection], dim=1)
        x = self.relu(x)
        return x


class Pix2PixGenerator(nn.Module):
    def __init__(self, input_channels=3, output_channels=3):
        super(Pix2PixGenerator, self).__init__()

        # Encoder
        self.e1 = EncoderBlock(input_channels, 64, batchnorm=False)
        self.e2 = EncoderBlock(64, 128)
        self.e3 = EncoderBlock(128, 256)
        self.e4 = EncoderBlock(256, 512)
        self.e5 = EncoderBlock(512, 512)
        self.e6 = EncoderBlock(512, 512)
        self.e7 = EncoderBlock(512, 512)

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )

        # Decoder (no lo usamos para el encoder, pero dejamos completo)
        self.d1 = DecoderBlock(512,   512)
        self.d2 = DecoderBlock(1024,  512)
        self.d3 = DecoderBlock(1024,  512)
        self.d4 = DecoderBlock(1024,  512, dropout=False)
        self.d5 = DecoderBlock(1024,  256, dropout=False)
        self.d6 = DecoderBlock(512,   128, dropout=False)
        self.d7 = DecoderBlock(256,   64,  dropout=False)

        self.out_conv = nn.ConvTranspose2d(128, output_channels, kernel_size=4, stride=2, padding=1)
        self.out_activation = nn.Tanh()

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(e1)
        e3 = self.e3(e2)
        e4 = self.e4(e3)
        e5 = self.e5(e4)
        e6 = self.e6(e5)
        e7 = self.e7(e6)

        b  = self.bottleneck(e7)

        d1 = self.d1(b,  e7)
        d2 = self.d2(d1, e6)
        d3 = self.d3(d2, e5)
        d4 = self.d4(d3, e4)
        d5 = self.d5(d4, e3)
        d6 = self.d6(d5, e2)
        d7 = self.d7(d6, e1)

        out_image = self.out_activation(self.out_conv(d7))
        return out_image


def load_pix2pix_generator_weights(
    generator: nn.Module,
    ckpt_path: str,
    map_location: str = "cpu"
):
    """
    Carga pesos desde un .pth de tu reconstructor.
    Soporta checkpoints guardados con DataParallel ('module.*').
    """
    state = torch.load(ckpt_path, map_location=map_location)

    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    new_state = {}
    for k, v in state.items():
        if k.startswith("module."):
            k = k[len("module."):]
        new_state[k] = v

    missing, unexpected = generator.load_state_dict(new_state, strict=False)
    print(f"[load_pix2pix_generator_weights] Missing keys: {missing}")
    print(f"[load_pix2pix_generator_weights] Unexpected keys: {unexpected}")
    return generator


class UNetEncoder(nn.Module):
    def __init__(
        self,
        input_channels: int = 3,
        pretrained_generator_path: str = None,
        device: str = "cpu",
    ):
        super().__init__()
        self.encoder_unet = Pix2PixGenerator(
            input_channels=input_channels,
            output_channels=3  # no importa para el encoder
        )

        if pretrained_generator_path is not None:
            print(f"[UNetEncoder] Loading pretrained generator from: {pretrained_generator_path}")
            load_pix2pix_generator_weights(
                self.encoder_unet,
                pretrained_generator_path,
                map_location=device
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.encoder_unet.e1(x)
        e2 = self.encoder_unet.e2(e1)
        e3 = self.encoder_unet.e3(e2)
        e4 = self.encoder_unet.e4(e3)
        e5 = self.encoder_unet.e5(e4)
        e6 = self.encoder_unet.e6(e5)
        e7 = self.encoder_unet.e7(e6)
        b  = self.encoder_unet.bottleneck(e7)  # (N, 512, 1, 1) para 256x256
        return b


# ============================================================
# 3) UNet + LSTM Classifier (SOLO LSTM)
# ============================================================
class UNetLSTMClassifier(nn.Module):
    """
    UNet encoder + LSTM sobre secuencias de frames.
    Espera entradas de tamaño [batch, seq_len, C, H, W].
    """
    def __init__(
        self,
        num_classes: int = 7,
        input_channels: int = 3,
        pretrained_generator_path: str = None,
        device: str = "cpu",
        image_size: int = 256,
        seq_len: int = 3,
        lstm_hidden_dim: int = 512,
        lstm_num_layers: int = 1,
    ):
        super().__init__()

        self.seq_len = seq_len

        # Encoder U-Net
        self.encoder = UNetEncoder(
            input_channels=input_channels,
            pretrained_generator_path=pretrained_generator_path,
            device=device,
        )

        # Inferir dimensión del bottleneck
        with torch.no_grad():
            dummy = torch.randn(1, input_channels, image_size, image_size)
            feat  = self.encoder(dummy)          # [1, 512, 1, 1]
            feat_flat = feat.view(1, -1)         # [1, 512]
            feat_dim = feat_flat.size(1)
        print(f"[UNetLSTMClassifier] Bottleneck feature dim: {feat_dim}")

        # LSTM sobre secuencia de vectores (uno por frame)
        self.lstm = nn.LSTM(
            input_size=feat_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_num_layers,
            batch_first=True
        )

        # Clasificador final (usar 'classifier' para ser compatible con train_classifier)
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, seq_len, C, H, W]
        """
        B, T, C, H, W = x.shape
        assert T == self.seq_len, f"Esperaba seq_len={self.seq_len} pero llegó {T}"

        # Reorganizar para procesar todos los frames de golpe
        x_reshaped = x.view(B * T, C, H, W)      # [B*T, C, H, W]

        # Pasar por encoder U-Net frame a frame
        feat = self.encoder(x_reshaped)          # [B*T, 512, 1, 1]
        feat = feat.view(B, T, -1)               # [B, T, feat_dim]

        # Secuencia por LSTM
        lstm_out, _ = self.lstm(feat)            # [B, T, hidden_dim]

        # Tomar la última salida temporal
        last_out = lstm_out[:, -1, :]            # [B, hidden_dim]

        # Clasificación
        logits = self.classifier(last_out)       # [B, num_classes]
        return logits


# ============================================================
# 4) Entrenamiento (scratch / linear / finetune)
# ============================================================
def train_classifier(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    n_epochs: int,
    lr_encoder: float,
    lr_fc: float,
    mode: str,
    output_dir: str,
    class_names=None
):
    os.makedirs(output_dir, exist_ok=True)

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()

    # Configurar optimizador según modo
    if mode == "scratch":
        print("[train_classifier] MODE = scratch (encoder + LSTM + classifier desde cero).")
        optimizer = torch.optim.Adam(model.parameters(), lr=lr_fc)

    elif mode == "linear":
        print("[train_classifier] MODE = linear (encoder congelado, LSTM + classifier entrenables).")
        for p in model.encoder.parameters():
            p.requires_grad = False
        # Entrenamos LSTM + classifier
        fc_params = list(model.lstm.parameters()) + list(model.classifier.parameters())
        optimizer = torch.optim.Adam(fc_params, lr=lr_fc)

    elif mode == "finetune":
        print("[train_classifier] MODE = finetune (encoder + LSTM + classifier, "
              "LR más bajo en encoder).")
        encoder_params = [p for p in model.encoder.parameters() if p.requires_grad]
        head_params = list(model.lstm.parameters()) + list(model.classifier.parameters())
        optimizer = torch.optim.Adam(
            [
                {"params": encoder_params, "lr": lr_encoder},
                {"params": head_params,    "lr": lr_fc},
            ]
        )
    else:
        raise ValueError(f"Modo no reconocido: {mode}")

    best_val_acc = 0.0
    best_ckpt    = os.path.join(output_dir, f"best_unet_lstm_classifier_{mode}.pth")

    train_losses, train_accuracies, val_accuracies = [], [], []

    for epoch in range(n_epochs):
        # ---------- TRAIN ----------
        model.train()
        running_loss = 0.0
        correct, total = 0, 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{n_epochs}")
        for imgs, labels in pbar:
            imgs   = imgs.to(device)    # [B, seq_len, C, H, W]
            labels = labels.to(device)  # [B]

            optimizer.zero_grad()
            logits = model(imgs)
            loss   = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * imgs.size(0)
            _, preds = torch.max(logits, 1)
            total   += labels.size(0)
            correct += (preds == labels).sum().item()

            acc = 100.0 * correct / max(total, 1)
            pbar.set_postfix(loss=running_loss / max(total, 1), acc=f"{acc:.2f}%")

        train_loss = running_loss / max(total, 1)
        train_acc  = 100.0 * correct / max(total, 1)
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)

        # ---------- VAL ----------
        model.eval()
        val_loss = 0.0
        val_correct, val_total = 0, 0

        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs   = imgs.to(device)
                labels = labels.to(device)

                logits = model(imgs)
                loss   = criterion(logits, labels)

                val_loss   += loss.item() * imgs.size(0)
                _, preds    = torch.max(logits, 1)
                val_total  += labels.size(0)
                val_correct += (preds == labels).sum().item()

        val_loss /= max(val_total, 1)
        val_acc   = 100.0 * val_correct / max(val_total, 1)
        val_accuracies.append(val_acc)

        print(
            f"[Epoch {epoch+1}/{n_epochs}] "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%"
        )

        # Guardar mejor modelo
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_ckpt)
            print(f"  -> New best model saved to: {best_ckpt} (Val Acc: {val_acc:.2f}%)")

    # Guardar curvas de entrenamiento
    np.savetxt(os.path.join(output_dir, "train_loss.txt"), train_losses, fmt="%.4f")
    np.savetxt(os.path.join(output_dir, "train_acc.txt"),  train_accuracies, fmt="%.2f")
    np.savetxt(os.path.join(output_dir, "val_acc.txt"),    val_accuracies, fmt="%.2f")

    # Plot loss
    plt.figure()
    plt.plot(train_losses, label="Train Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss (UNet+LSTM classifier)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "train_loss.png"))
    plt.close()

    # Plot accuracies (0-1)
    train_acc_norm = [0.0] + [x/100.0 for x in train_accuracies]
    val_acc_norm   = [0.0] + [x/100.0 for x in val_accuracies]
    epochs = list(range(len(train_acc_norm)))

    plt.figure(figsize=(8,6))
    plt.plot(epochs, train_acc_norm, label="Train Acc", color="blue")
    plt.plot(epochs, val_acc_norm,   label="Val Acc",   color="orange")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (0-1)")
    plt.title("Accuracy per Epoch (Train vs Val)")
    plt.grid(True)
    plt.ylim(0,1)
    plt.xlim(0,len(train_acc_norm)-1)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "accuracy.png"))
    plt.close()

    # -------- Confusion Matrix con el mejor modelo --------
    model.load_state_dict(torch.load(best_ckpt, map_location=device))
    model.to(device)
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs   = imgs.to(device)
            labels = labels.to(device)
            logits = model(imgs)
            _, preds = torch.max(logits, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds)
    cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True)

    np.savetxt(os.path.join(output_dir, "confusion_matrix_raw.txt"), cm, fmt="%d")
    np.savetxt(os.path.join(output_dir, "confusion_matrix_normalized.txt"), cm_norm, fmt="%.2f")

    if class_names is None:
        class_names = [str(i) for i in range(cm.shape[0])]

    plt.figure(figsize=(8,6))
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Normalized Confusion Matrix (UNet+LSTM classifier)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confusion_matrix_best_model.png"))
    plt.close()

    print(f"✅ Training finished. Best Val Acc: {best_val_acc:.2f}%")
    print(f"✅ All results saved under: {output_dir}")


# ============================================================
# 5) Utilities pix2pix
# ============================================================
def infer_tie_from_data_dir(data_dir: str) -> str:
    """
    Extrae el sufijo t_t, t0_t, t_t0, t0_t0 a partir de un data_dir tipo:
    '../input/e-ck+_frames_lstm_process_30fps_t_t'
    """
    base = os.path.basename(os.path.normpath(data_dir))
    # asume patrón e-ck+_frames_lstm_process_30fps_<tie>
    if "30fps_" in base:
        return base.split("30fps_")[1]
    else:
        return base.split("_")[-1]


def find_pix2pix_run_dir(pix2pix_root: str, tie: str) -> str:
    """
    Busca automáticamente en pix2pix_root el último run cuyo nombre termine en 'X1_<tie>'.
    Ej: 20251030_181352_pix2pix_unet_faces_events_ck_X1_t0_t
    """
    if not os.path.isdir(pix2pix_root):
        raise FileNotFoundError(f"pix2pix_root no existe: {pix2pix_root}")

    candidates = [
        d for d in os.listdir(pix2pix_root)
        if d.endswith(f"X1_{tie}") and os.path.isdir(os.path.join(pix2pix_root, d))
    ]
    if not candidates:
        raise FileNotFoundError(f"No se encontró ningún run en '{pix2pix_root}' que termine en 'X1_{tie}'")

    dir_name = sorted(candidates)[-1]
    return os.path.join(pix2pix_root, dir_name)


def build_ckpt_from_epoch(pix2pix_root: str, tie: str, epoch: int) -> str:
    """
    Construye automáticamente el path al .pth:
    <pix2pix_run_dir>/epoch_00XX/model_0000XX.pth
    """
    run_dir   = find_pix2pix_run_dir(pix2pix_root, tie)
    epoch_dir = os.path.join(run_dir, f"epoch_{epoch:04d}")
    ckpt_path = os.path.join(epoch_dir, f"model_{epoch:06d}.pth")

    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"No existe el checkpoint esperado: {ckpt_path}")
    print(f"[build_ckpt_from_epoch] Usando checkpoint: {ckpt_path}")
    return ckpt_path


# ============================================================
# 6) run_for_one_data_dir (SOLO LSTM: data_dir/train y data_dir/test)
# ============================================================
def run_for_one_data_dir(args, data_dir: str, device: torch.device):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = os.path.basename(os.path.normpath(data_dir))
    out_dir   = os.path.join(args.output_root, f"unet_lstm_{base_name}_{args.mode}_{timestamp}")
    os.makedirs(out_dir, exist_ok=True)

    # Transforms
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),
    ])

    test_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),
    ])

    train_dir = os.path.join(data_dir, "train")
    test_dir  = os.path.join(data_dir, "test")

    if not (os.path.isdir(train_dir) and os.path.isdir(test_dir)):
        raise FileNotFoundError(
            f"Esperaba subcarpetas 'train' y 'test' dentro de {data_dir} (dataset LSTM)."
        )

    train_dataset = SequenceDataset(train_dir, transform=train_transform, seq_len=args.seq_len)
    test_dataset  = SequenceDataset(test_dir,  transform=test_transform,  seq_len=args.seq_len)
    class_names   = train_dataset.class_names
    num_classes   = len(class_names)

    print(f"[{base_name}] Found {num_classes} classes: {class_names}")
    print(f"[{base_name}] Train samples: {len(train_dataset)} | Test samples: {len(test_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,  num_workers=2)
    test_loader  = DataLoader(test_dataset,  batch_size=args.batch_size, shuffle=False, num_workers=2)

    # -------- elegir pretrained_path según modo --------
    if args.mode == "scratch":
        pretrained_path = None
    elif args.mode in ["linear", "finetune"]:
        if args.pretrained_generator is not None:
            pretrained_path = args.pretrained_generator
        else:
            if args.pretrained_epoch is None:
                raise ValueError(
                    f"Modo '{args.mode}' requiere --pretrained_generator o --pretrained_epoch"
                )
            tie = infer_tie_from_data_dir(data_dir)
            pretrained_path = build_ckpt_from_epoch(
                pix2pix_root=args.pix2pix_root,
                tie=tie,
                epoch=args.pretrained_epoch
            )
    else:
        raise ValueError

    # -------- Modelo --------
    model = UNetLSTMClassifier(
        num_classes=num_classes,
        input_channels=3,
        pretrained_generator_path=pretrained_path,
        device=str(device),
        image_size=256,
        seq_len=args.seq_len,
        lstm_hidden_dim=args.lstm_hidden_dim,
        lstm_num_layers=args.lstm_num_layers,
    )

    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        print(f"Using DataParallel over {torch.cuda.device_count()} GPUs.")
        model = nn.DataParallel(model)

    # Guardar hiperparámetros
    hyperparams = {
        "data_dir": data_dir,
        "mode": args.mode,
        "pretrained_generator": pretrained_path,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr_encoder": args.lr_encoder,
        "lr_fc": args.lr_fc,
        "image_size": "256x256",
        "classes": class_names,
        "architecture": "UNetEncoder + LSTM",
        "seq_len": args.seq_len,
        "lstm_hidden_dim": args.lstm_hidden_dim,
        "lstm_num_layers": args.lstm_num_layers,
    }
    with open(os.path.join(out_dir, "hyperparameters.json"), "w") as f:
        json.dump(hyperparams, f, indent=4)

    # Entrenamiento
    train_classifier(
        model=model,
        train_loader=train_loader,
        val_loader=test_loader,
        device=device,
        n_epochs=args.epochs,
        lr_encoder=args.lr_encoder,
        lr_fc=args.lr_fc,
        mode=args.mode,
        output_dir=out_dir,
        class_names=class_names,
    )


# ============================================================
# 7) main
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="UNet-encoder + LSTM classifier for e-ck+ sequences.")
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Carpeta con subcarpetas 'train' y 'test' (secuencias LSTM).")
    parser.add_argument("--mode", choices=["scratch", "linear", "finetune"], default="scratch")
    parser.add_argument("--pretrained_generator", type=str, default=None,
                        help="Ruta al .pth del GENERATOR pix2pix entrenado (para modos linear/finetune).")
    parser.add_argument("--pretrained_epoch", type=int, default=None,
                        help="Época del pix2pix a usar (si no se pasa --pretrained_generator).")
    parser.add_argument("--pix2pix_root", type=str, default="../input/pix2pix",
                        help="Carpeta raíz de los runs de pix2pix.")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr_encoder", type=float, default=1e-4)
    parser.add_argument("--lr_fc",      type=float, default=1e-3)
    parser.add_argument("--output_root", type=str, default="../results/tie-unet-lstm",
                        help="Carpeta raíz donde se guardarán los resultados.")
    parser.add_argument("--gpu", type=str, default=None, help="IDs de GPU, ej: '0' o '0,1'.")
    parser.add_argument("--auto_all_ties", action="store_true",
                        help="Si se activa, recorre automáticamente todas las carpetas e-ck+_frames_lstm_process_30fps_*")
    # LSTM
    parser.add_argument("--seq_len", type=int, default=3,
                        help="Longitud de la secuencia temporal (para LSTM).")
    parser.add_argument("--lstm_hidden_dim", type=int, default=512,
                        help="Dimensión oculta del LSTM.")
    parser.add_argument("--lstm_num_layers", type=int, default=1,
                        help="Número de capas LSTM apiladas.")
    args = parser.parse_args()

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
        print(f"CUDA_VISIBLE_DEVICES = {os.environ['CUDA_VISIBLE_DEVICES']}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(args.output_root, exist_ok=True)

    if args.auto_all_ties:
        input_root = "../input"
        dirs = [
            os.path.join(input_root, d)
            for d in os.listdir(input_root)
            if d.startswith("e-ck+_frames_lstm_process_30fps_")
        ]
        dirs = sorted(dirs)
        print(f"Se encontraron {len(dirs)} variantes LSTM: {dirs}")

        for data_dir in dirs:
            print(f"\n========== Ejecutando para {data_dir} ==========\n")
            run_for_one_data_dir(args, data_dir, device)

    else:
        if args.data_dir is None:
            raise ValueError("Debes especificar --data_dir o usar --auto_all_ties")
        run_for_one_data_dir(args, args.data_dir, device)


if __name__ == "__main__":
    main()
