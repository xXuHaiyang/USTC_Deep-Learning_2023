import os

from PIL import Image
import numpy as np

import torch
import torch.nn as nn

import torchvision.transforms as transforms

def image_loader(image_name, imsize):
    
    loader = transforms.Compose(
        [
            transforms.Resize((imsize, imsize)),
            transforms.ToTensor()
        ]
    )

    image = Image.open(image_name)
    image = loader(image)
    image = image.unsqueeze(0)
    
    return image

def image_loader_gray(image_name, imsize):
    
    loader = transforms.Compose(
        [
            transforms.Resize((imsize, imsize)),
            transforms.ToTensor()
        ]
    )

    image = Image.open(image_name).convert('L')
    image = np.array(image)
    image = np.array([image, image, image], dtype=np.uint8)
    image = Image.fromarray(image.transpose(1, 2, 0))
    image = loader(image)
    image = image.unsqueeze(0)
    
    return image

def save_image(tensor, src, fname='transferred.png'):
    
    to_PIL = transforms.ToPILImage()

    img = tensor.detach().cpu().clone()
    img = img.squeeze(0)
    img = to_PIL(img)
    
    src_img = Image.open(src)
    src_img_tensor = transforms.ToTensor()(src_img)
    imsize = src_img_tensor.shape
    img = img.resize((imsize[2], imsize[1]))

    out_path = os.path.join('transferred', fname)
    if not os.path.exists('transferred'):
        os.mkdir('transferred')

    img.save(out_path)

class ContentLoss(nn.Module):
    def __init__(self, target, weight):
        super(ContentLoss, self).__init__()
        self.target = target.detach() * weight
        self.weight = weight
        self.criterion = nn.MSELoss()

    def forward(self, input):
        output = input.clone()
        content = input 
        content *= self.weight
        self.loss = self.criterion(content, self.target)
        return output

    def backward(self, retain_graph=True):
        self.loss.backward(retain_graph=retain_graph)
        return self.loss

class GramMatrix(nn.Module):

    def forward(self, input):
        
        B, C, H, W = input.shape
        features = input.reshape(B * C, H * W)
        
        # compute gram matrix G and normalize
        G = features @ features.T / (B * C * H * W)

        return G

class StyleLoss(nn.Module):

    def __init__(self, target, weight):
        super(StyleLoss, self).__init__()
        self.target = target.detach() * weight
        self.weight = weight
        self.gram = GramMatrix()
        self.criterion = nn.MSELoss()

    def forward(self, input):
        output = input.clone()
        style = self.gram(input)
        style *= self.weight
        self.loss = self.criterion(style, self.target)
        return output

    def backward(self, retain_graph=True):
        self.loss.backward(retain_graph=retain_graph)
        return self.loss