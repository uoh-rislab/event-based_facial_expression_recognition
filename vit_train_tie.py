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

# Configuración
fps = 30
#tie = 't_t'

for tie in ['t_t', 't_t0', 't0_t', 't0_t0']:

    data_dir = f'output/e-ck+_frames_process_{fps}fps_{tie}/'  # carpeta que contiene Train_Set y Test_Set

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"results/vit_e-ckplus_{fps}fps_{tie}_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(output_dir, "tensorboard"))

    batch_size = 4
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
            transforms.Resize((256, 256)),
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            #transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3)
            #transforms.Normalize(mean=mean_std['mean'], std=mean_std['std'])
        ])

        test_transform = transforms.Compose([
            #transforms.Grayscale(num_output_channels=3),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3)
            #transforms.Normalize(mean=mean_std['mean'], std=mean_std['std'])
        ])
    else:
        train_transform = transforms.Compose([
            #transforms.Grayscale(num_output_channels=1),
            transforms.Resize((256, 256)),
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

        test_transform = transforms.Compose([
            #transforms.Grayscale(num_output_channels=1),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

    # Dataset y loaders
    train_dataset = datasets.ImageFolder(os.path.join(data_dir, 'Train_Set'), transform=train_transform)
    test_dataset = datasets.ImageFolder(os.path.join(data_dir, 'Test_Set'), transform=test_transform)


    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)


    # Modelo
    if use_pretrained:
        model = vit_b_16(weights='IMAGENET1K_V1')
    else:
        model = vit_b_16(weights=None)
        model.conv_proj = nn.Conv2d(1, 768, kernel_size=16, stride=16)

    # Detectar si es Sequential o Linear
    if isinstance(model.heads, nn.Sequential):
        in_features = model.heads[0].in_features
    else:
        in_features = model.heads.in_features

    # Reemplazar la cabeza
    model.heads = nn.Linear(in_features, num_classes)
    model.to(device)

    best_val_acc = 0.0
    save_path = os.path.join(output_dir, 'best_model_vit.pth')

    # Función de pérdida y optimizador
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)

    train_losses, train_accuracies, val_accuracies = [], [], []


    # Entrenamiento
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct, total = 0, 0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")

        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

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
            for val_inputs, val_labels in test_loader:
                val_inputs, val_labels = val_inputs.to(device), val_labels.to(device)
                val_outputs = model(val_inputs)
                _, val_predicted = torch.max(val_outputs, 1)
                val_total += val_labels.size(0)
                val_correct += (val_predicted == val_labels).sum().item()

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
            print(f"✅ Nuevo mejor modelo guardado en: {save_path} (val acc: {val_acc:.2f}%)")


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

    plt.figure()
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(val_accuracies, label='Val Accuracy')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Accuracy per Epoch")
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(output_dir, "accuracy.png"))

    # Confusion Matrix Final
    class_names = ['Anger', 'Contempt', 'Disgust', 'Fear', 'Happy', 'Sadness', 'Surprise']
    model.load_state_dict(torch.load(save_path))
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

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

    print(f"✅ Resultados completos guardados en: {output_dir}")
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