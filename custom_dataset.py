from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

class CustomDataset(Dataset):
    def __init__(self, data_file, labels_file, transform=None):
        self.data = np.load(data_file)
        self.labels = np.load(labels_file)
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.data[:, idx].reshape(300, 300, 3)  # Reshape the individual image to its original shape
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)


        return image, label