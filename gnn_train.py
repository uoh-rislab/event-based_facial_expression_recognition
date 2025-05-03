from torch_geometric.data import Dataset, DataLoader
import torch
import os
from glob import glob
from torch_geometric.data import Data

from torch.nn import Linear, ReLU
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool

from torch_geometric.loader import DataLoader
import torch
from sklearn.metrics import accuracy_score


class EventGraphDataset(Dataset):
    def __init__(self, root_dir, split):
        super().__init__()
        self.graph_paths = []
        self.labels = []
        self.label_map = {}  # clase → int

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
        return data

class GNN(torch.nn.Module):
    def __init__(self, in_dim=4, hidden_dim=64, out_dim=7):
        super().__init__()
        self.conv1 = GATv2Conv(in_dim, hidden_dim, heads=2, concat=True)
        self.conv2 = GATv2Conv(hidden_dim * 2, hidden_dim, heads=2, concat=True)
        self.lin1 = Linear(hidden_dim * 2, hidden_dim)
        self.lin2 = Linear(hidden_dim, out_dim)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = global_mean_pool(x, batch)
        x = F.relu(self.lin1(x))
        return self.lin2(x)

train_dataset = EventGraphDataset('ddbb_graphs/e-ck+_graphs_30fps', 'Train_Set')
test_dataset = EventGraphDataset('ddbb_graphs/e-ck+_graphs_30fps', 'Test_Set')

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

model = GNN(in_dim=4, out_dim=len(train_dataset.label_map))
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

for epoch in range(1, 21):
    model.train()
    total_loss = 0
    for batch in train_loader:
        batch = batch.to(device)
        out = model(batch)
        loss = F.cross_entropy(out, batch.y.view(-1))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total_loss += loss.item()

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
    print(f"Epoch {epoch} - Loss: {total_loss:.4f} - Test Acc: {acc:.4f}")

