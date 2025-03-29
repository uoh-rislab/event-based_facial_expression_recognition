# ===========================
# 1️⃣ Importar Librerías
# ===========================
import os
import json
import torch
import numpy as np
import seaborn as sns
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.models import vit_b_16
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
from datetime import datetime
from torchvision.models import vit_b_16
from PIL import Image
import yaml

# ===========================
# 2️⃣ Definir Modelo ViT + LSTM
# ===========================
class ViT_LSTM_Classifier(nn.Module):
    def __init__(self, embed_dim=768, hidden_dim=512, num_classes=7, num_layers=1):
        super(ViT_LSTM_Classifier, self).__init__()
        
        # Cargar modelo ViT-B/16 desde torchvision (preentrenado)
        self.vit = vit_b_16(weights=pretrained_weights)
        self.vit.heads = nn.Identity()  # Eliminar la capa de clasificación

        
        # LSTM para procesar secuencias de embeddings de ViT
        self.lstm = nn.LSTM(input_size=embed_dim, hidden_size=hidden_dim, 
                            num_layers=num_layers, batch_first=True)
        
        # Capa de clasificación final
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, images):
        """
        Entrada: images -> [batch_size, seq_len, 3, 224, 224]
        """
        batch_size, seq_len, c, h, w = images.size()
        
        vit_outputs = []
        # Procesar cada imagen individualmente con ViT desde torchvision
        for i in range(seq_len):
            img_i = images[:, i, :, :, :]  # Obtener imagen en la posición i
            vit_out = self.vit(img_i)  # Obtener embedding
            vit_outputs.append(vit_out)

        # Concatenar embeddings para crear secuencia [batch_size, seq_len, embed_dim]
        vit_outputs = torch.stack(vit_outputs, dim=1)
        
        # Procesar la secuencia de embeddings con LSTM
        lstm_out, _ = self.lstm(vit_outputs)
        
        # Usar la última salida de la secuencia para clasificación
        out = lstm_out[:, -1, :]  # Tomar la última salida temporal
        
        # Predicción final
        logits = self.fc(out)
        return logits


# ===========================
# 3️⃣ Definir Dataset Personalizado
# ===========================
class SequenceDataset(Dataset):
    def __init__(self, root_dir, transform=None, seq_len=3):
        self.root_dir = root_dir
        self.transform = transform
        self.seq_len = seq_len
        self.samples = []

        # Obtener secuencias de todas las clases
        for class_name in os.listdir(root_dir):
            class_path = os.path.join(root_dir, class_name)
            if not os.path.isdir(class_path):
                continue

            # Agrupar secuencias de un mismo sujeto
            seq_dict = {}
            for frame_name in sorted(os.listdir(class_path)):
                if frame_name.endswith(".jpg"):
                    subject_name = "_".join(frame_name.split("_")[:2])  # Ejemplo: S010_004
                    if subject_name not in seq_dict:
                        seq_dict[subject_name] = []
                    seq_dict[subject_name].append(os.path.join(class_path, frame_name))

            # Añadir secuencias válidas al dataset
            for seq_name, frame_paths in seq_dict.items():
                if len(frame_paths) >= seq_len:
                    for i in range(len(frame_paths) - seq_len + 1):
                        seq_frames = frame_paths[i:i + seq_len]

                        # Verificar si el nombre de la clase es un número válido
                        if not class_name.isdigit():
                            print(f"⚠️ Carpeta inválida detectada: {class_path}. Ignorando.")
                            continue

                        # Extraer la clase como entero
                        label = int(class_name)

                        self.samples.append((seq_frames, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        frame_paths, label = self.samples[idx]
        
        # Cargar y transformar cada imagen
        images = []
        for frame_path in frame_paths:
            img = Image.open(frame_path).convert("RGB")
            if self.transform:
                img = self.transform(img)
            images.append(img)
        
        # Apilar imágenes para crear tensor de secuencia [seq_len, 3, 224, 224]
        images = torch.stack(images, dim=0)
        return images, label


# ===========================
# 9️⃣ Función de Entrenamiento
# ===========================
def train_model(model, train_loader, val_loader, num_epochs=10):
    train_losses, train_accuracies, val_accuracies = [], [], []

    for epoch in range(num_epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")

        for images, labels in progress_bar:
            images, labels = images.to(device), labels.to(device)

            # Restar 1 para convertir etiquetas de [1, 7] a [0, 6]
            labels = labels - 1

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            acc = 100 * correct / total
            progress_bar.set_postfix(loss=running_loss / total, acc=f"{acc:.2f}%")

        # Calcular métricas de entrenamiento
        train_loss = running_loss / total
        train_acc = 100 * correct / total
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)

        # ===========================
        # 🔍 Validación al Final de Cada Época
        # ===========================
        model.eval()
        val_correct, val_total = 0, 0
        val_running_loss = 0.0

        with torch.no_grad():
            for val_images, val_labels in val_loader:
                val_images, val_labels = val_images.to(device), val_labels.to(device)

                # Restar 1 para convertir etiquetas de [1, 7] a [0, 6]
                val_labels = val_labels - 1

                val_outputs = model(val_images)
                val_loss = criterion(val_outputs, val_labels)

                val_running_loss += val_loss.item()
                _, val_predicted = torch.max(val_outputs, 1)
                val_total += val_labels.size(0)
                val_correct += (val_predicted == val_labels).sum().item()

        val_acc = 100 * val_correct / val_total
        val_loss_avg = val_running_loss / val_total
        val_accuracies.append(val_acc)

        print(f"✅ Validation Accuracy after Epoch {epoch+1}: {val_acc:.2f}%")

        # Guardar el mejor modelo si la precisión mejora
        global best_val_acc
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print(f"✅ Nuevo mejor modelo guardado en: {save_path} (val acc: {val_acc:.2f}%)")

    # ===========================
    # 🎨 Guardar Gráficos y Métricas Solo al Final
    # ===========================
    np.savetxt(os.path.join(output_dir, "train_loss.txt"), train_losses, fmt='%.4f')
    np.savetxt(os.path.join(output_dir, "train_acc.txt"), train_accuracies, fmt='%.2f')
    np.savetxt(os.path.join(output_dir, "val_acc.txt"), val_accuracies, fmt='%.2f')

    # Graficar pérdida del entrenamiento
    plt.figure()
    plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss per Epoch")
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(output_dir, "train_loss.png"))

    # Graficar precisión de entrenamiento y validación
    plt.figure()
    plt.plot(range(1, num_epochs + 1), np.array(train_accuracies) / 100.0, label='Train Accuracy')
    plt.plot(range(1, num_epochs + 1), np.array(val_accuracies) / 100.0, label='Val Accuracy')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (0-1)")
    plt.title("Accuracy per Epoch (Train vs Val)")
    plt.grid(True)
    plt.ylim(0, 1)  # Limitar el eje y entre 0 y 1
    plt.legend()
    plt.savefig(os.path.join(output_dir, "accuracy.png"))

    print("✅ Gráficos de métricas guardados exitosamente después del entrenamiento.")



# ===========================
# 4️⃣ Transformaciones y Data Augmentation
# ===========================
train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
    #transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                     std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
    #transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                     std=[0.229, 0.224, 0.225])
])


# ===========================
# 📄 Cargar Configuración desde YAML
# ===========================
with open("yaml/config_asus.yaml", "r") as f:
    config = yaml.safe_load(f)

# Parámetros de configuración
num_epochs = config["num_epochs"]
batch_size = config["batch_size"]
data_dir_base = config["data_dir_base"]
num_classes = config["num_classes"]
learning_rate = config["learning_rate"]
learning_rate = float(config["learning_rate"])  # Asegurar que sea float
pretrained_weights = config["pretrained_weights"]
fps = config["fps"]
ties = config["ties"]


# ===========================
# 6️⃣ Bucle Principal para Variantes tie
# ===========================
for tie in ties:
    print(f"🏁 Entrenando modelo para tie: {tie}...")

    # Directorio de datos para la variante actual
    data_dir = f"{data_dir_base}/e-ck+_frames_lstm_process_{fps}fps_{tie}/"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"results/{timestamp}_vit_lstm_e-ckplus_{fps}fps_{tie}"
    os.makedirs(output_dir, exist_ok=True)

    # ===========================
    # 7️⃣ Cargar Dataset y DataLoader
    # ===========================
    train_dataset = SequenceDataset(root_dir=os.path.join(data_dir, 'train'), transform=train_transform)
    val_dataset = SequenceDataset(root_dir=os.path.join(data_dir, 'test'), transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)


    # ===========================
    # 8️⃣ Configurar Modelo, Pérdida y Optimizador
    # ===========================
    model = ViT_LSTM_Classifier(num_classes=num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    best_val_acc = 0.0
    save_path = os.path.join(output_dir, 'best_model_vit_lstm.pth')


    # ===========================
    # 🔥 Entrenamiento del Modelo
    # ===========================
    train_model(model, train_loader, val_loader, num_epochs=num_epochs)

    # ===========================
    # 🎯 Matriz de Confusión
    # ===========================
    class_names = ['Anger', 'Contempt', 'Disgust', 'Fear', 'Happy', 'Sadness', 'Surprise']

    model.load_state_dict(torch.load(save_path))
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)

            # Restar 1 para convertir etiquetas de [1, 7] a [0, 6]
            labels = labels - 1

            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds)
    cm_norm = cm.astype('float') / cm.sum(axis=1, keepdims=True)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Normalized Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"))

    # Guardar Matrices
    np.savetxt(os.path.join(output_dir, "confusion_matrix_raw.txt"), cm, fmt='%d')
    np.savetxt(os.path.join(output_dir, "confusion_matrix_normalized.txt"), cm_norm, fmt='%.2f')

    # ===========================
    # 💾 Guardar Hiperparámetros
    # ===========================
    hyperparams = {
        "fps": fps,
        "tie": tie,
        "data_dir": data_dir,
        "batch_size": batch_size,
        "num_epochs": 20,
        "num_classes": 7,
        "optimizer": "AdamW",
        "learning_rate": 1e-4,
        "image_size": "224x224",
        "augmentation": {
            "RandomResizedCrop": "scale=(0.8, 1.0)",
            "RandomHorizontalFlip": 0.5,
            "RandomRotation": 15,
            "ColorJitter": "brightness=0.2, contrast=0.2"
        },
        "architecture": "vit_b_16 + LSTM"
    }

    with open(os.path.join(output_dir, "hyperparameters.json"), "w") as f:
        json.dump(hyperparams, f, indent=4)

    # Guardar arquitectura del modelo
    with open(os.path.join(output_dir, "model_architecture.txt"), "w") as f:
        f.write(str(model))

    print(f"✅ Finalizado para tie: {tie}\n\n")
