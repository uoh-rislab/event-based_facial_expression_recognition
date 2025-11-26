# ===========================
#  Entrenamiento y Evaluación con GNN y Augmentación
# ===========================

import os
import json
import torch
import numpy as np
import seaborn as sns
import torch.nn.functional as F
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm
from glob import glob
from sklearn.metrics import confusion_matrix, accuracy_score
from torch.nn import Linear, Dropout, BatchNorm1d
from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATv2Conv, global_mean_pool
from torch_geometric.nn import GCNConv

# Función de pérdida con Label Smoothing
class LabelSmoothingCrossEntropy(torch.nn.Module):
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, pred, target):
        n_classes = pred.size(1)
        log_preds = F.log_softmax(pred, dim=1)
        with torch.no_grad():
            true_dist = torch.zeros_like(log_preds)
            true_dist.fill_(self.smoothing / (n_classes - 1))
            true_dist.scatter_(1, target.unsqueeze(1), 1.0 - self.smoothing)
        return torch.mean(torch.sum(-true_dist * log_preds, dim=1))

# Dataset con graph augmentation
class EventGraphDataset(Dataset):
    def __init__(self, root_dir, split, augment=False):
        super().__init__()
        self.graph_paths = []
        self.labels = []
        self.label_map = {}
        self.augment = augment

        base_dir = os.path.join(root_dir, split)
        classes = sorted(os.listdir(base_dir))
        for i, cls in enumerate(classes):
            self.label_map[cls] = i
            for subject in os.listdir(os.path.join(base_dir, cls)):
                subject_path = os.path.join(base_dir, cls, subject)
                for graph_file in glob(os.path.join(subject_path, '*.pt')):
                    self.graph_paths.append(graph_file)
                    self.labels.append(i)

    def len(self):
        return len(self.graph_paths)

    def get(self, idx):
        data = torch.load(self.graph_paths[idx])
        data.y = torch.tensor([self.labels[idx]], dtype=torch.long)
        if self.augment:
            data = self.augment_graph(data)
        return data

    def augment_graph(self, data):
        if torch.rand(1).item() < 0.3:
            noise = 0.05 * torch.randn_like(data.x)
            data.x = data.x + noise
        return data

# Modelo GNN
class GNN(torch.nn.Module):
    def __init__(self, in_dim=4, hidden_dim=64, out_dim=7):
        super().__init__()
        self.conv1 = GATv2Conv(in_dim, hidden_dim, heads=2, concat=True)
        self.bn1 = BatchNorm1d(hidden_dim * 2)
        self.conv2 = GATv2Conv(hidden_dim * 2, hidden_dim, heads=2, concat=True)
        self.bn2 = BatchNorm1d(hidden_dim * 2)
        self.lin1 = Linear(hidden_dim * 2, hidden_dim)
        self.lin2 = Linear(hidden_dim, out_dim)
        self.dropout = Dropout(p=0.3)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.bn1(self.conv1(x, edge_index)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.conv2(x, edge_index)))
        x = self.dropout(x)
        x = global_mean_pool(x, batch)
        x = F.relu(self.lin1(x))
        return self.lin2(x)

class DeepGNN(torch.nn.Module):
    def __init__(self, in_dim=4, hidden_dim=64, out_dim=7):
        super().__init__()
        self.conv1 = GATv2Conv(in_dim, hidden_dim, heads=2, concat=True)
        self.bn1 = BatchNorm1d(hidden_dim * 2)
        
        self.conv2 = GATv2Conv(hidden_dim * 2, hidden_dim, heads=2, concat=True)
        self.bn2 = BatchNorm1d(hidden_dim * 2)

        self.conv3 = GATv2Conv(hidden_dim * 2, hidden_dim, heads=2, concat=True)
        self.bn3 = BatchNorm1d(hidden_dim * 2)

        self.conv4 = GATv2Conv(hidden_dim * 2, hidden_dim, heads=2, concat=True)
        self.bn4 = BatchNorm1d(hidden_dim * 2)

        self.lin1 = Linear(hidden_dim * 2, hidden_dim)
        self.lin2 = Linear(hidden_dim, out_dim)
        self.dropout = Dropout(p=0.3)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.bn1(self.conv1(x, edge_index)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.conv2(x, edge_index)))
        x = self.dropout(x)
        x = F.relu(self.bn3(self.conv3(x, edge_index)))
        x = self.dropout(x)
        x = F.relu(self.bn4(self.conv4(x, edge_index)))
        x = self.dropout(x)
        x = global_mean_pool(x, batch)
        x = F.relu(self.lin1(x))
        return self.lin2(x)


# Configuración
root_dir = 'input/e-ck+_graphs_30fps'
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
results_dir = os.path.join("results", "gnn_augmented", f"gnn_e-ckplus_{timestamp}")
os.makedirs(results_dir, exist_ok=True)
log_path = os.path.join(results_dir, "training_log.txt")

train_dataset = EventGraphDataset(root_dir, 'Train_Set', augment=True)
test_dataset = EventGraphDataset(root_dir, 'Test_Set', augment=False)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

model = DeepGNN(in_dim=4, out_dim=len(train_dataset.label_map))
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
loss_fn = LabelSmoothingCrossEntropy(smoothing=0.1)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

class_names = ['Anger', 'Contempt', 'Disgust', 'Fear', 'Happy', 'Sadness', 'Surprise']
best_acc = 0.0
train_losses, train_accuracies, test_accuracies = [], [], []

patience = 10
epochs_no_improve = 0

for epoch in range(1, 151):
    torch.cuda.empty_cache()
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}")
    for batch in progress_bar:
        batch = batch.to(device)
        out = model(batch)
        loss = loss_fn(out, batch.y.view(-1))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total_loss += loss.item()
        preds = out.argmax(dim=1)
        correct += (preds == batch.y.view(-1)).sum().item()
        total += batch.y.size(0)
        progress_bar.set_postfix(loss=total_loss / total, acc=f"{100 * correct / total:.2f}%")

        del loss, out, preds, batch
        torch.cuda.empty_cache()

    train_losses.append(total_loss / total)
    train_accuracies.append(100 * correct / total)
    scheduler.step()

    # Evaluación
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            out = model(batch)
            pred = out.argmax(dim=1)
            y_true.extend(batch.y.view(-1).cpu().tolist())
            y_pred.extend(pred.cpu().tolist())

    acc = accuracy_score(y_true, y_pred)
    test_accuracies.append(100 * acc)
    log_line = f"Epoch {epoch} - Loss: {total_loss:.4f} - Train Acc: {train_accuracies[-1]:.2f}% - Test Acc: {acc:.4f}"
    print(log_line)

    with open(log_path, "a") as f:
        f.write(log_line + "\n")

    if acc > best_acc:
        best_acc = acc
        epochs_no_improve = 0  # reset counter
        torch.save(model.state_dict(), os.path.join(results_dir, "best_model.pth"))

        # Guardar arquitectura
        with open(os.path.join(results_dir, "model_architecture.txt"), "w") as f:
            f.write(str(model))

        # Matriz de confusión
        cm = confusion_matrix(y_true, y_pred)
        cm_norm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.title("Normalized Confusion Matrix")
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, "confusion_matrix.png"))
        plt.close()

        np.savetxt(os.path.join(results_dir, "confusion_matrix_raw.txt"), cm, fmt='%d')
        np.savetxt(os.path.join(results_dir, "confusion_matrix_normalized.txt"), cm_norm, fmt='%.2f')

    else:
        epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print(f"🛑 Early stopping triggered at epoch {epoch}")
            break

# Guardar métricas
np.savetxt(os.path.join(results_dir, "train_loss.txt"), train_losses, fmt='%.4f')
np.savetxt(os.path.join(results_dir, "train_acc.txt"), train_accuracies, fmt='%.2f')
np.savetxt(os.path.join(results_dir, "test_acc.txt"), test_accuracies, fmt='%.2f')

plt.figure()
plt.plot(train_losses, label='Train Loss')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.grid(True)
plt.legend()
plt.savefig(os.path.join(results_dir, "train_loss.png"))

plt.figure()
plt.plot(np.array(train_accuracies) / 100.0, label='Train Accuracy')
plt.plot(np.array(test_accuracies) / 100.0, label='Test Accuracy')
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.ylim(0, 1)
plt.title("Accuracy per Epoch")
plt.grid(True)
plt.legend()
plt.savefig(os.path.join(results_dir, "accuracy.png"))

print(f"Finalizado. Mejor precisión: {best_acc:.4f}")
