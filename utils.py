# TO:DO : improve recover_image

import torch
import torchvision as tv
from torchvision import transforms
from torch.utils.data import DataLoader

import numpy as np
from PIL import Image

import matplotlib.pyplot as plt


def gram_matrix(x):
    """returns gram matrix of the image x"""
    a, b, c, d = x.size()  # a=batch size(=1), b=number of feature maps, (c,d)=dimensions of a f. map (N=c*d)

    features = x.view(a , b, c * d)  # resise F_XL into \hat F_XL

    G = torch.bmm(features, features.transpose(1,2))  # compute the gram product Batch matrix multiplication

    # we 'normalize' the values of the gram matrix by dividing by the number of element in each feature maps.
    return G.div(a * b * c * d)


def get_numpy_image_to_plot(img):
    """recover image after de-normalizing
    takes tensor as input and converts to numpy array -> [1, H, W, RGB]
    """

    if isinstance(img, torch.Tensor):
        img = img.to('cpu').numpy()
    
    #Denormalize
    img = img * np.array([0.229, 0.224, 0.225]).reshape((1, 3, 1, 1)) + np.array([0.485, 0.456, 0.406]).reshape((1, 3, 1, 1)) 
    img = img.transpose(0,2,3,1) * 255.  # [batch, H, W, RGB] * scale up to 255
    img = img.clip(0,255).astype(np.uint8) # clip leftovers and convert to unsigned int for plt.imshow
    return img


def get_batch_tensor_from_image(image, image_size= (512,512), device ='cpu'):
    """
    return normalized tensor of required image_size with batch dim  - > [1,3,image_size] [1,3,512,512]
    """

    img = Image.open(image).convert('RGB')

    with torch.no_grad():
        req_transform = transforms.Compose([transforms.CenterCrop(image_size), transforms.ToTensor(),
                                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        img_tensor = req_transform(img).unsqueeze(0).to(device)

    return img_tensor

def save_debug_image(tensor_orig, tensor_transformed, filename):
    """saves image at current state of the model for intermediate viewing"""
    assert tensor_orig.size() == tensor_transformed.size()
    result = Image.fromarray(get_numpy_image_to_plot(tensor_transformed.cpu().numpy())[0])
    orig = Image.fromarray(get_numpy_image_to_plot(tensor_orig.cpu().numpy())[0])
    new_im = Image.new('RGB', (result.size[0] * 2 + 5, result.size[1]))
    new_im.paste(orig, (0,0))
    new_im.paste(result, (result.size[0] + 5,0))
    new_im.save(filename)

def convert_to_size_and_save(img_src, filename, img_size= (256,256)):
    """ write to specific size and output_name"""
    import cv2
    img = cv2.imread(img_src, cv2.IMREAD_UNCHANGED)[:,:,::-1]
    img_ = cv2.resize(img, img_size)
    cv2.imwrite(filename, img_[:,:,::-1])
    return

def resize_to_fixed(img_src, filename):
    """ write to specific size and output_name"""
    import cv2
    img = cv2.imread(img_src, cv2.IMREAD_UNCHANGED)[:,:,::-1]
    shape = img.shape
    x = shape[0]
    y = shape[1]
    z = min(x, y)
    new_shape = (y*256//z, x*256//z)
    img_ = cv2.resize(img, new_shape)
    cv2.imwrite(filename, img_[:,:,::-1])
    return

def post_processing(cost_curves_filename="./artifacts/loss_details_per_epoch.txt"):
    """given the filename of [cost_curves_filename] (txt file)
    returns history - dictionary(keys- epochs) of dictionaries(keys - lossese)
    
    loss_dict = {"epoch":count,
                         "agg_content_loss": agg_content_loss.item()/10,
                         "agg_style_loss": agg_style_loss.item()/10,
                         "agg_reg_loss": agg_reg_loss.item()/10,
                         "total_loss": (agg_content_loss.item() +agg_style_loss.item() +agg_reg_loss.item() )/ 10}
                      
    
    """
    with open(cost_curves_filename ,"r") as f:             
        losses = f.read().splitlines()
    history = {}
    for num,item in enumerate(losses):
        temp_dict = eval(item)
        history[num] = temp_dict
    return history

def get_loss(lines):
    """gets loss from list of strings in dict form as in Johnson et al code"""
    x = []
    content_loss = []
    style_loss = []
    tv_reg_loss = []
    total_loss = []
    for line in lines:
        val = eval(line)
        x.append(val['epoch'])
        content_loss.append(val['agg_content_loss'])
        style_loss.append(val['agg_style_loss'])
        tv_reg_loss.append(val['agg_reg_loss'])
        total_loss.append(val['total_loss'])
    return x, content_loss, style_loss, tv_reg_loss, total_loss
        
def plot_loss(x, content_loss, style_loss, tv_reg_loss, total_loss):
    """returns plots with loss input as list"""
    fig, axes = plt.subplots(nrows=1, ncols=4, sharey=False, figsize=(20, 5))
    axes[0].plot(x[::20], content_loss[::20])
    axes[0].set_xlim(10,40000)
    axes[0].set_xticks(x[999::1000])
    axes[0].set_title('Content Loss')
    axes[0].set_xlabel('Iterations')
    axes[0].set_ylabel('Loss')
    axes[1].plot(x[::20], style_loss[::20])
    axes[1].set_xlim(10,40000)
    axes[1].set_ylim(0, 20)
    axes[1].set_xticks(x[999::1000])
    axes[1].set_title('Style Loss')
    axes[1].set_xlabel('Iterations')
    axes[1].set_ylabel('Loss')
    axes[2].plot(x[::20], tv_reg_loss[::20])
    axes[2].set_xlim(10,40000)
    axes[2].set_xticks(x[999::1000])
    axes[2].set_title('TV Regularization Loss')
    axes[2].set_xlabel('Iterations')
    axes[2].set_ylabel('Loss')
    axes[3].plot(x[::20], total_loss[::20])
    axes[3].set_ylim(0, 60)
    axes[3].set_xlim(10,40000)
    axes[3].set_xticks(x[999::1000])
    axes[3].set_title('Total Loss')
    axes[3].set_xlabel('Iterations')
    axes[3].set_ylabel('Loss')
    fig.show()
    
def get_plots(filename):
    """plots loss for losses in given file"""
    fd = open(filename)
    lines = fd.readlines()
    fd.close()
    plot_loss(*get_loss(lines))