import os
import copy
import argparse

from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models

from utils import image_loader, image_loader_gray, save_image, ContentLoss, GramMatrix, StyleLoss

import ipdb

def get_style_model_and_losses(cnn, 
                               style_img, content_img, 
                               style_weight, content_weight, 
                               content_layers, style_layers,
                               device):

    content_losses = []
    style_losses = []

    model = nn.Sequential().to(device)
    gram = GramMatrix().to(device)

    i = 1
    cnn = copy.deepcopy(cnn)
    for layer in list(cnn):
        if isinstance(layer, nn.Conv2d):
            name = "conv_" + str(i)
            model.add_module(name, layer)
            if name in content_layers:
                # add content loss:
                target = model(content_img).clone()
                content_loss = ContentLoss(target, content_weight)
                model.add_module("content_loss_" + str(i), content_loss)
                content_losses.append(content_loss)
            if name in style_layers:
                # add style loss:
                target_feature = model(style_img).clone()
                target_feature_gram = gram(target_feature)
                style_loss = StyleLoss(target_feature_gram, style_weight)
                model.add_module("style_loss_" + str(i), style_loss)
                style_losses.append(style_loss)
        if isinstance(layer, nn.ReLU):
            name = "relu_" + str(i)
            model.add_module(name, layer)
            if name in content_layers:
                # add content loss:
                target = model(content_img).clone()
                content_loss = ContentLoss(target, content_weight)
                model.add_module("content_loss_" + str(i), content_loss)
                content_losses.append(content_loss)
            if name in style_layers:
                # add style loss:
                target_feature = model(style_img).clone()
                target_feature_gram = gram(target_feature)
                style_loss = StyleLoss(target_feature_gram, style_weight)
                model.add_module("style_loss_" + str(i), style_loss)
                style_losses.append(style_loss)
            i += 1
        if isinstance(layer, nn.MaxPool2d):
            name = "pool_" + str(i)
            model.add_module(name, layer)

    return model, style_losses, content_losses


def run_style_transfer(input_img, model, style_losses, content_losses, args):

    input_param = nn.Parameter(input_img.data)
    optimizer = optim.LBFGS([input_param])

    run = [0]
    while run[0] <= args.epoch:
        
        def closure():
            input_param.data.clamp_(0, 1)

            optimizer.zero_grad()
            model(input_param)
            style_loss = 0
            content_loss = 0

            for sl in style_losses:
                style_loss += sl.backward()
            for cl in content_losses:
                content_loss += cl.backward()

            run[0] += 1
            if run[0] % 50 == 0:
                print(f"{run[0]}/{args.epoch}: ")
                print('Style Loss: {:4f} Content Loss: \{:4f}'.format
                    (style_loss.item(), content_loss.item()))

            loss = style_loss + content_loss
            return loss
        
        optimizer.step(closure)

    input_param.data.clamp_(0, 1)

    return input_param.data


def args_parser():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--content', '-c', type=str, required=True, help='The path to the Content image')
    parser.add_argument('--style', '-s', type=str, required=True, help='The path to the style image')
    parser.add_argument('--epoch', '-e', type=int, default=300, help='The number of epoch')
    parser.add_argument('--content_weight', '-c_w', type=int, default=1, help='The weight of content loss')
    parser.add_argument('--style_weight', '-s_w', type=int, default=1000, help='The weight of style loss')
    parser.add_argument('--imsize', '-i', type=int, default=1536, help='The size of image')

    return parser

def main():
    
    # set random determinstic
    torch.manual_seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    parser = args_parser()
    args = parser.parse_args()
    
    # load images
    print("Loading images ...")
    style_img = image_loader(args.style, args.imsize).to(device)
    content_img = image_loader(args.content, args.imsize).to(device)
    input_img = torch.randn(content_img.shape).to(device)
    
    assert style_img.shape == content_img.shape, \
        "We need to import style and content images of the same size"
    
    # prepare model and losses
    print("Create model ...")
    cnn = models.vgg19(pretrained=True).to(device).features
    content_layers_default = ['conv_4']
    style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']
    model, style_losses, content_losses = get_style_model_and_losses(cnn, 
                                                                     style_img, content_img, 
                                                                     args.style_weight, args.content_weight,
                                                                     content_layers_default, style_layers_default, 
                                                                     device)

    # run the model
    print("Begin style transfer ...")
    input_img.requires_grad_(True)
    model.requires_grad_(False)
    style_transfer_img = run_style_transfer(input_img, model, style_losses, content_losses, args)
    
    # save the style transfer image
    name_content, ext = os.path.splitext(os.path.basename(args.content))
    name_style, _ = os.path.splitext(os.path.basename(args.style))
    fname = name_content + '-' + name_style + ext
    save_image(style_transfer_img, args.content, fname=fname)
    print(f"Save Image in ./transferred/{fname}")

if __name__ == "__main__":
    main()