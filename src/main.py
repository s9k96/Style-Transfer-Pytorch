import torch
import torch.nn.functional as F
from torchvision import transforms as tf
from torchvision import models, utils
from torch import optim

from datetime import datetime
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import config

vgg = models.vgg19(pretrained=True).features

for param in vgg.parameters():
    param.requires_grad_(False)
vgg.to(config.DEVICE)

mean = (0.485, 0.465, 0.406)
std = (0.229, 0.224, 0.224)

LAYERS_OF_INTEREST = {
    '0' : 'conv1_1',
    '5' : 'conv2_1',
    '10' : 'conv3_1',
    '19' : 'conv4_1',
    '21' : 'conv4_2',
    '28' : 'conv5_1'
}

def transformations(img):
    tasks = tf.Compose([
        tf.Resize(256), 
        tf.ToTensor(), 
        tf.Normalize(mean,std)
    ])
    img = tasks(img)
    img = img.unsqueeze(0)      # Adding batch dimension to the tensor
    return img

def tensor_to_img(tensor):
    '''
    helper function to convert tenror to numpy matrix for plotting through matplotlib.
    '''
    img = tensor.clone().detach()
    img = img.cpu().numpy().squeeze()
    img = img.transpose(1,2,0)
    img *=np.array(std) + np.mean(mean)
    img = img.clip(0,1)
    return img

def apply_model_and_extract_features(image, model):
    features = {}
    x = image

    for name, layer in vgg._modules.items():
        x = layer(x)
        
        if name in LAYERS_OF_INTEREST:
            features[LAYERS_OF_INTEREST[name]] = x
    return features


def calculate_gram_matrix(tensor):
    _, channels, width, height = tensor.size()
    tensor = tensor.view(channels, height*width)
    
    gram_matrix = torch.mm(tensor, tensor.t())
    gram_matrix = gram_matrix.div(channels*width*height)
    return gram_matrix


def main():

    content_img = Image.open(config.CONTENT_IMAGE).convert('RGB')
    style_img = Image.open(config.STYLE_IMAGE).convert('RGB')

    content_img = transformations(content_img).to(config.DEVICE)
    style_img = transformations(style_img).to(config.DEVICE)


    content_img_features = apply_model_and_extract_features(content_img, vgg)
    style_img_features = apply_model_and_extract_features(style_img, vgg)

    style_features_gram_matrix = {layer: calculate_gram_matrix(style_img_features[layer]) for layer in style_img_features}

    weights = dict(zip(LAYERS_OF_INTEREST.values(), [1.0, 0.75, 0.35, 0.25, 0.15]))

    target = content_img.clone().requires_grad_(True).to(config.DEVICE)
    optimizer = optim.Adam([target], lr = 0.003)

    for i in range(1, config.EPOCH):
        start_time = datetime.now()
        target_features = apply_model_and_extract_features(target, vgg)
        content_loss = F.mse_loss(target_features['conv4_2'], content_img_features['conv4_2'])  
        
        style_loss = 0
        for layer in weights:
            target_feature = target_features[layer]
            
            target_gram_matrix = calculate_gram_matrix(target_feature)
            style_gram_matrix = style_features_gram_matrix[layer]
            
            layer_loss = F.mse_loss(target_gram_matrix, style_gram_matrix)
            layer_loss = layer_loss*weights[layer]
            
            style_loss = style_loss+ layer_loss
            
        total_loss = 1000000*style_loss + content_loss       # Since style loss is much lower than content loss, we can amplify it so they both contribute equally. 
        
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        if i%100 == 0:
            print("Epoch {}: ContentLoss: {:4f} StyleLoss: {:4f} TotalLoss: {:4f} time/epoch: {}".format(i, content_loss, style_loss, total_loss, datetime.now()-start_time))


    utils.save_image(target, '../images/output.jpg')

if __name__ == '__main__':
    main()
    