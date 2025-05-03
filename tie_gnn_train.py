import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, utils
from torchvision.models import vit_b_16
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np
from datetime import datetime
import os
import json

from torch_geometric.data import Data
from torch_geometric.utils import grid
from PIL import Image

from torch_geometric.loader import DataLoader as GeoDataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.nn import GATv2Conv

from sklearn.utils.class_weight import compute_class_weight
from torch_geometric.nn import global_add_pool

# Configuración
fps = 30
img_size = 224

class ImageToGraphDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None, image_size=img_size):
        self.root_dir = root_dir
        self.transform = transform
        self.image_size = image_size
        self.classes = sorted(os.listdir(root_dir))
        self.filepaths = []
        for class_index, class_name in enumerate(self.classes):
            class_path = os.path.join(root_dir, class_name)
            for root, _, files in os.walk(class_path):
                for filename in files:
                    if filename.lower().endswith((".png", ".jpg", ".jpeg")):
                        self.filepaths.append((os.path.join(root, filename), class_index))


    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        img_path, label = self.filepaths[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)  # [3, H, W]

        c, h, w = image.shape
        image_flat = image.view(c, -1).permute(1, 0)  # [H*W, C]

        # Detectar eventos: píxeles donde al menos un canal sea distinto a 0
        mask = (image_flat != 0).any(dim=1)  # [H*W]
        x = image_flat[mask]  # [N_active, C]

        # Obtener coordenadas (y, x) originales
        coords = torch.stack(torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij"), dim=-1).view(-1, 2)
        coords = coords[mask]  # [N_active, 2]

        # Construir grafo por vecindad espacial
        from torch_geometric.nn import radius_graph
        edge_index = radius_graph(coords.float(), r=2.0, loop=False)

        return Data(x=x, edge_index=edge_index, y=torch.tensor(label, dtype=torch.long))


class GATv2Net(nn.Module):
    def __init__(self, in_channels=3, hidden_channels=128, num_classes=7, heads=4):
        super().__init__()

        self.gat1 = GATv2Conv(in_channels, hidden_channels, heads=heads, concat=True)
        self.proj1 = nn.Linear(hidden_channels * heads, hidden_channels)

        self.gat2 = GATv2Conv(hidden_channels, hidden_channels)
        self.gat3 = GATv2Conv(hidden_channels, hidden_channels)
        self.gat4 = GATv2Conv(hidden_channels, hidden_channels)
        self.gat5 = GATv2Conv(hidden_channels, hidden_channels)

        self.bn1 = nn.BatchNorm1d(hidden_channels * heads)
        self.bn2 = nn.BatchNorm1d(hidden_channels)
        self.bn3 = nn.BatchNorm1d(hidden_channels)
        self.bn4 = nn.BatchNorm1d(hidden_channels)
        self.bn5 = nn.BatchNorm1d(hidden_channels)

        self.dropout = nn.Dropout(0.2)
        self.lin = nn.Linear(hidden_channels, num_classes)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x1 = torch.relu(self.bn1(self.gat1(x, edge_index)))
        x1_proj = self.dropout(self.proj1(x1))

        x2 = torch.relu(self.bn2(self.gat2(x1_proj, edge_index))) + x1_proj
        x3 = torch.relu(self.bn3(self.gat3(x2, edge_index))) + x2
        x4 = torch.relu(self.bn4(self.gat4(x3, edge_index))) + x3
        x5 = torch.relu(self.bn5(self.gat5(x4, edge_index))) + x4

        x = global_add_pool(x5, batch)
        return self.lin(x)



for tie in ['t_t', 't_t0', 't0_t', 't0_t0']:

    #data_dir = f'output/e-ck+_frames_process_{fps}fps_{tie}/'  # carpeta que contiene Train_Set y Test_Set
    data_dir = f'input/e-ck+_frames_process_{fps}fps_{tie}/'  

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"results/tie-gnn/gnn_e-ckplus_{fps}fps_{tie}_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(output_dir, "tensorboard"))

    batch_size = 8
    num_epochs = 20
    num_classes = 7  # Ajusta al número real de clases
    use_pretrained = True  # True = usa ViT con pesos ImageNet
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    normalize_values = {
        't_t':    {'mean': [0.1065, 0.2633, 0.0600], 'std': [0.1305, 0.1285, 0.1172]},
        't_t0':   {'mean': [0.0457, 0.0467, 0.0490], 'std': [0.1154, 0.1147, 0.1098]},
        't0_t':   {'mean': [0.0961, 0.2531, 0.0532], 'std': [0.1194, 0.1199, 0.1106]},
        't0_t0':  {'mean': [0.0406, 0.0468, 0.0479], 'std': [0.1099, 0.1100, 0.1055]},
    }

    mean_std = normalize_values[tie]

    # Transformaciones y Data Augmentation
    if use_pretrained:
        train_transform = transforms.Compose([
            #transforms.Grayscale(num_output_channels=3),
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            #transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3)
            #transforms.Normalize(mean=mean_std['mean'], std=mean_std['std'])
        ])

        test_transform = transforms.Compose([
            #transforms.Grayscale(num_output_channels=3),
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3)
            #transforms.Normalize(mean=mean_std['mean'], std=mean_std['std'])
        ])
    else:
        train_transform = transforms.Compose([
            #transforms.Grayscale(num_output_channels=1),
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

        test_transform = transforms.Compose([
            #transforms.Grayscale(num_output_channels=1),
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

    # Dataset y loaders
    train_dataset = ImageToGraphDataset(os.path.join(data_dir, 'Train_Set'), transform=test_transform, image_size=28)
    test_dataset = ImageToGraphDataset(os.path.join(data_dir, 'Test_Set'), transform=test_transform, image_size=28)

    train_loader = GeoDataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = GeoDataLoader(test_dataset, batch_size=batch_size, shuffle=False)


    # Modelo
    #model = GCN(in_channels=3, hidden_channels=64, num_classes=num_classes).to(device)
    model = GATv2Net(in_channels=3, hidden_channels=64, num_classes=num_classes, heads=4).to(device)
    
    best_val_acc = 0.0
    save_path = os.path.join(output_dir, 'best_model_vit.pth')

    # Función de pérdida y optimizador
    #criterion = nn.CrossEntropyLoss()
    
    # Extraer las etiquetas manualmente desde tu dataset personalizado
    train_targets = [label for _, label in train_dataset.filepaths]
    class_labels = np.unique(train_targets)

    weights = compute_class_weight('balanced', classes=class_labels, y=train_targets)
    class_weights_tensor = torch.tensor(weights, dtype=torch.float).to(device)

    # Y luego úsalo en la función de pérdida
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor) 

    optimizer = optim.AdamW(model.parameters(), lr=1e-4)

    train_losses, train_accuracies, val_accuracies = [], [], []


    # Entrenamiento
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct, total = 0, 0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")

        for data in progress_bar:
            data = data.to(device)
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, data.y)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += data.y.size(0)
            correct += (predicted == data.y).sum().item()

            acc = 100 * correct / total
            progress_bar.set_postfix(loss=running_loss / total, acc=f"{acc:.2f}%")

        train_loss = running_loss / total
        train_acc = 100 * correct / total
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)

        # Validación usando test_loader
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for data in test_loader:
                data = data.to(device)
                val_outputs = model(data)
                _, val_predicted = torch.max(val_outputs, 1)

                val_total += data.y.size(0)
                val_correct += (val_predicted == data.y).sum().item()


        val_acc = 100 * val_correct / val_total
        val_accuracies.append(val_acc)
        print(f"Validation Accuracy: {val_acc:.2f}%\n")

        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Accuracy/train", train_acc, epoch)
        writer.add_scalar("Accuracy/val", val_acc, epoch)

        # Guardar el mejor modelo
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print(f"Nuevo mejor modelo guardado en: {save_path} (val acc: {val_acc:.2f}%)")


    # Guardar métricas
    np.savetxt(os.path.join(output_dir, "train_loss.txt"), train_losses, fmt='%.4f')
    np.savetxt(os.path.join(output_dir, "train_acc.txt"), train_accuracies, fmt='%.2f')
    np.savetxt(os.path.join(output_dir, "val_acc.txt"), val_accuracies, fmt='%.2f')

    # Graficar métricas
    plt.figure()
    plt.plot(train_losses, label='Train Loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(output_dir, "train_loss.png"))

    # Normalizar los valores de accuracy al rango [0, 1]
    train_accuracies_norm = [0.0] + [x / 100 for x in train_accuracies]
    val_accuracies_norm = [0.0] + [x / 100 for x in val_accuracies]

    # Crear la lista de epochs considerando el punto (0, 0)
    epochs = list(range(len(train_accuracies_norm)))

    # Crear la figura y plotear los valores normalizados
    plt.figure(figsize=(8, 6))  # Tamaño más profesional
    plt.plot(epochs, train_accuracies_norm, label='Train Accuracy', color='blue')
    plt.plot(epochs, val_accuracies_norm, label='Val Accuracy', color='orange')

    # Configuración del gráfico
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (0-1)")
    plt.title("Accuracy per Epoch (Train vs Val)")
    plt.grid(True)
    plt.ylim(0, 1)  # Limitar el eje y entre 0 y 1
    plt.xlim(0, len(train_accuracies_norm) - 1)  # Ajustar eje x
    plt.legend()
    plt.tight_layout()

    # Guardar la imagen en el directorio de salida
    plt.savefig(os.path.join(output_dir, "accuracy.png"))

    # Confusion Matrix Final
    class_names = ['Anger', 'Contempt', 'Disgust', 'Fear', 'Happy', 'Sadness', 'Surprise']
    model.load_state_dict(torch.load(save_path))
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            outputs = model(data)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(data.y.cpu().numpy())


    cm = confusion_matrix(all_labels, all_preds)
    cm_norm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
    np.savetxt(os.path.join(output_dir, "confusion_matrix_raw.txt"), cm, fmt='%d')
    np.savetxt(os.path.join(output_dir, "confusion_matrix_normalized.txt"), cm_norm, fmt='%.2f')

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Normalized Confusion Matrix (Best Validation Model)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confusion_matrix_best_model.png"))
    #plt.show()

    model_output_path = os.path.join(output_dir, "best_model_vit.pth")
    if os.path.exists(save_path):
        os.rename(save_path, model_output_path)

    print(f"Resultados completos guardados en: {output_dir}")
    writer.close()

    # Guardar hiperparámetros
    hyperparams = {
        "data_dir": data_dir,
        "batch_size": batch_size,
        "num_epochs": num_epochs,
        "num_classes": num_classes,
        "use_pretrained": use_pretrained,
        "optimizer": "AdamW",
        "learning_rate": 1e-4,
        "image_size": "224x224",
        "augmentation": {
            "RandomResizedCrop": "scale=(0.8, 1.0)",
            "RandomHorizontalFlip": 0.5,
            "RandomRotation": 15
        },
        "architecture": "vit_b_16",
        "pretrained_weights": "IMAGENET1K_V1" if use_pretrained else "None"
    }

    with open(os.path.join(output_dir, "hyperparameters.json"), "w") as f:
        json.dump(hyperparams, f, indent=4)

    # Guardar arquitectura del modelo
    with open(os.path.join(output_dir, "model_architecture.txt"), "w") as f:
        f.write(str(model))
