from torch.utils import data
from torchvision import transforms as T
from PIL import Image
import os
import numpy as np
import pickle


def save(a, filename):
    with open(f'{filename}.pickle', 'wb') as handle:
        pickle.dump(a, handle, protocol=pickle.HIGHEST_PROTOCOL)

def load(filename):
    with open(f'{filename}.pickle', 'rb') as handle:
        b = pickle.load(handle)

    return b


def crop_top(img, percent=0.15):
    offset = int(img.shape[0] * percent)
    return img[offset:]

def central_crop(img):
    size = min(img.shape[0], img.shape[1])
    offset_h = int((img.shape[0] - size) / 2)
    offset_w = int((img.shape[1] - size) / 2)
    return img[offset_h:offset_h + size, offset_w:offset_w + size]


class Covid(data.Dataset):
    """Dataset class for the Covid dataset."""

    def __init__(self, image_dir, transform, mode):
        """Initialize and preprocess the Covid dataset."""
        self.image_dir = image_dir
        self.transform = transform
        self.mode = mode
        self.datasetA = []
        self.datasetB = []
        self.preprocess()

        if mode == 'train':
            self.num_images = len(self.datasetA) + len(self.datasetB)
        else:
            self.num_images = max(len(self.datasetA), len(self.datasetB))

    def preprocess(self):
        if self.mode in ['train', 'test2'] :
            pos = load(os.path.join("data", "covid", "train_pos"))
            neg = load(os.path.join("data", "covid", "train_neg"))
            neg_mixed = load(os.path.join("data", "covid", "train_neg_mixed"))

            self.datasetA = pos + neg_mixed
            self.datasetB = neg
        else:
            self.datasetA = load(os.path.join("data", "covid", "test_pos"))
            self.datasetB = load(os.path.join("data", "covid", "test_neg"))

        print('Finished preprocessing the COVID dataset...')

    def __getitem__(self, index):
        """Return one image and its corresponding attribute label."""
        datasetA = self.datasetA
        datasetB = self.datasetB
        
        filenameA = datasetA[index%len(datasetA)]
        filenameB = datasetB[index%len(datasetB)]

        if self.mode in ['train']:
            imageA = Image.open(os.path.join(self.image_dir, 'train', filenameA)).convert("RGB")
            imageB = Image.open(os.path.join(self.image_dir, 'train', filenameB)).convert("RGB")
        else:
            imageA = Image.open(os.path.join(self.image_dir, 'test', filenameA)).convert("RGB")
            imageB = Image.open(os.path.join(self.image_dir, 'test', filenameB)).convert("RGB")

        imageA = np.array(imageA)
        imageA = crop_top(imageA, 0.08)
        imageA = central_crop(imageA)

        imageB = np.array(imageB)
        imageB = crop_top(imageB, 0.08)
        imageB = central_crop(imageB)

        imageA = Image.fromarray(imageA)
        imageB = Image.fromarray(imageB)

        return self.transform(imageA), self.transform(imageB)

    def __len__(self):
        """Return the number of images."""
        return self.num_images



class TestValid(data.Dataset):
    """Dataset class for the Covid dataset."""

    def __init__(self, image_dir, transform, mode):
        """Initialize and preprocess the Covid dataset."""
        self.image_dir = image_dir
        self.transform = transform
        self.mode = mode
        self.datasetA = []
        self.datasetB = []
        self.preprocess()

        if "ano" in self.mode:
            self.num_images = len(self.datasetA)
        elif "hea" in self.mode:
            self.num_images = len(self.datasetB)

    def preprocess(self):
        self.datasetA = load(os.path.join("data", "covid", "test_pos"))
        self.datasetB = load(os.path.join("data", "covid", "test_neg"))

        print(f'Finished preprocessing the COVID dataset for {self.mode} ...')

    def __getitem__(self, index):
        """Return one image and its corresponding attribute label."""
        if "ano" in self.mode:
            dataset = self.datasetA
        else:
            dataset = self.datasetB

        filename = dataset[index%len(dataset)]
        image = Image.open(os.path.join(self.image_dir, 'test', filename)).convert("RGB")

        image = np.array(image)
        image = crop_top(image, 0.08)
        image = central_crop(image)

        image = Image.fromarray(image)

        return self.transform(image)

    def __len__(self):
        """Return the number of images."""
        return self.num_images


def get_loader(image_dir, image_size=256, batch_size=1, dataset='Covid', mode='train', num_workers=1):
    """Build and return a data loader."""
    transform = []
    if mode == 'train':
        transform.append(T.RandomHorizontalFlip())
    transform.append(T.Resize(image_size))
    transform.append(T.ToTensor())
    transform.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
    transform = T.Compose(transform)

    if dataset == 'Covid':
        dataset = Covid(image_dir, transform, mode)
    elif dataset == 'TestValid':
        dataset = TestValid(image_dir, transform, mode)
    else:
        print("Dataset not found!")
        exit()

    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=(mode=='train'),
                                  num_workers=num_workers)
    return data_loader
