import torch
from torch.utils.data import Dataset
from torchvision import transforms
import os
from PIL import Image


class PointCloudSequences(Dataset):

    def __init__(self, path, transform):
        self.path = path
        self.transform = transform
        self.folders = self._load_folders()
        self.labels = self._load_labels()

    def _load_labels(self):
        labels_file = os.path.join(self.path, 'labels.txt')
        labels = []
        with open(labels_file, 'r') as f:
            for line in f.readlines():
                labels.append([int(x) for x in line.split()])

        labels = torch.LongTensor(labels)
        return labels

    def _load_folders(self):
        folders = []
        for name in os.listdir(self.path):
            p = os.path.join(self.path, name)
            if os.path.isdir(p):
                folders += [p]

        folders.sort()
        return folders

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        folder = self.folders[index]
        images = []
        image_names = os.listdir(folder)
        image_names.sort()
        for name in image_names:
            image = Image.open(os.path.join(folder, name)).convert('RGB')
            image = self.transform(image)
            images.append(image)

        images = torch.stack(images)
        labels = self.labels[index]
        sample = (images, labels)

        return sample



# if __name__ == '__main__':
#     tf = transforms.Compose([
#         transforms.Scale(200),
#         transforms.ToTensor(),
#     ])
#     dataset = PointCloudSequences('./data/clouds/train/', tf)
#
#     print(dataset[3])
#
