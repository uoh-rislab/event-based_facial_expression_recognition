from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch
import warnings
warnings.filterwarnings("ignore")

# Transformación solo con ToTensor para no alterar los valores
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Carga tu dataset
fps = 30


for tie in ['t_t', 't_t0', 't0_t', 't0_t0']:

    data_dir = f'output/e-ck+_frames_process_{fps}fps_{tie}/'  # carpeta que contiene Train_Set y Test_Set

    dataset = datasets.ImageFolder(data_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4)

    mean = 0.
    std = 0.
    nb_samples = 0.

    for data, _ in loader:
        batch_samples = data.size(0)
        data = data.view(batch_samples, data.size(1), -1)  # (B, C, H*W)
        mean += data.mean(2).sum(0)
        std += data.std(2).sum(0)
        nb_samples += batch_samples

    mean /= nb_samples
    std /= nb_samples

    print("Representacion: ", tie)
    print("Mean:", mean)
    print("Std:", std)
