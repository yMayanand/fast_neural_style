import cv2
import os
import numpy as np

from torchvision import transforms
import torch

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])

IMAGENET_MEAN_255 = np.array([123.675, 116.28, 103.53])
IMAGENET_STD_NEUTRAL = np.array([1, 1, 1])

def gram_matrix(x):
    """computes gram matrix of feature map shape (b, ch, h, w)"""
    b, ch, h, w = x.shape
    features = x.view(b, ch, h * w)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t) / (ch * h * w)
    return gram

def load_image(img_path, shape=None):
    """load image given its path"""
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if shape is not None:
        img = cv2.resize(img, shape)
    return img

def preprocess_image(image, size):
    """preprocesses image and prepares it to be fed to the model"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(size),
        transforms.CenterCrop(size),
        transforms.Lambda(lambda x: (x - x.min())/(x.max() - x.min())),
        transforms.Lambda(lambda x: x.mul(255)),
    ])
    img = transform(image)
    return img

def save_image(filename, data):
    """saves image after training"""
    img = data.permute(1, 2, 0).numpy().astype(np.uint8)
    cv2.imwrite(filename, img[:, :, ::-1]) # converts rgb to bgr due to opencv constraint

def postprocess_image(image):
    """postprocesses images after training"""
    image = image.squeeze(0)
    image = image.cpu().detach().clone().numpy()
    image = image.transpose(1, 2, 0)
    image = (image * IMAGENET_STD_NEUTRAL) + IMAGENET_MEAN_255
    image = image.clip(0, 255)
    return image


class Dataset:
    def __init__(self, files, root_dir, shape=256, preprocess=True):
        self.preprocess = preprocess
        self.shape = shape
        self.files = files
        self.root_dir = root_dir

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        path = os.path.join(self.root_dir, fname)
        image = load_image(path)
        if self.preprocess:
            image = preprocess_image(image, self.shape)

        return image

class AverageMeter:
    def __init__(self):
        self.reset()
    
    def update(self, val):
        self.val = val
        self.sum += val
        self.count += 1
        self.avg = self.sum / self.count

    def reset(self):
        self.val = 0
        self.sum = 0
        self.count = 0
        self.avg = 0
        

def save_model(model, model_dir):
    os.makedirs(model_dir, exist_ok=True)
    store = {'weights': model.state_dict()}
    fname = 'model.pt'
    torch.save(store, os.path.join(model_dir, fname))
    
def normalize_batch(batch):
    # normalize using imagenet mean and std
    mean = batch.new_tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
    std = batch.new_tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
    batch = batch.div_(255.0)
    return (batch - mean) / std

def quantize_model(model):
    model.qconfig =  torch.quantization.get_default_qconfig("fbgemm")
    model = torch.quantization.prepare(model)
    model = torch.quantization.convert(model)
    return model