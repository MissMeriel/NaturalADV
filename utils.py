import pickle
import sys
import os
print(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(f"{os.path.dirname(os.path.realpath(__file__))}/../")
sys.path.append(f"{os.path.dirname(os.path.realpath(__file__))}/DAVE2")
from DAVE2pytorch import *
import numpy as np
import PIL.Image
from matplotlib import pylab as P
import torch
from torchvision import models, transforms
from pathlib import Path
# import pandas as pd
import random, string
import cv2
import time
# from PIL import Image
import io 
import torch.nn.functional as F
# from torch.utils.data import DataLoader, TensorDataset
# import torch.optim as optim
from torchvision.transforms import Compose, ToPILImage, ToTensor, Resize, Lambda, Normalize


def randstr():
    return ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(6))


def timestr():
    localtime = time.localtime()
    return "{}_{}-{}_{}".format(localtime.tm_mon, localtime.tm_mday, localtime.tm_hour, localtime.tm_min)


class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
            return super().find_class(module, name)


def unpickle_for_image(img_filename, args):
    p = Path(img_filename)
    xrai_parent = args.xrais
    val_parent = args.dataset
    pickle_filename = xrai_parent + "/" + p.parts[-2] + "/" + p.stem + ".pickle"
    # print(f"{img_filename=}\n{p.parts=}")
    # print(f"{pickle_filename=}\n")
    with open(pickle_filename, "rb") as f:
        # unpickled_metas = pickle.load(f) becomes...
        unpickled_metas = CPU_Unpickler(f).load()
        return unpickled_metas


def ShowImage(im, title='', ax=None, outfile="./output_xrai/", id=None, save=False):
    if ax is None:
        P.close("all")
        P.figure()
    P.axis('off')
    P.imshow(im)
    P.title(title, fontdict={"size": 18})
    if save:
        P.savefig(f"{outfile}/showimg-{id}-{randstr()}.jpg")

def ShowGrayscaleImage(im, title='', ax=None, outfile="./output_xrai/", save=False):
    if ax is None:
        P.close("all")
        P.figure()
    P.axis('off')
    P.imshow(im, cmap=P.cm.gray, vmin=0, vmax=1)
    P.title(title, fontdict={"size": 18})
    P.colorbar()
    if save:
        P.savefig(f"{outfile}/grayscale-{randstr()}.jpg")

def ShowHeatMap(im, title, ax=None, outfile="./output_xrai/", id=None, save=False):
    if ax is None:
        P.close("all")
        P.figure()
    P.axis('off')
    a = P.imshow(im, cmap='inferno')
    P.title(title, fontdict={"size": 18})
    P.colorbar()
    if save:
        P.savefig(f"{outfile}/heatmap-{id}-{randstr()}.jpg")
    return a

def ShowOverlay(image, overlay, title, ax=None, outfile="./output_xrai/", id=None, save=False):
    # np.array uint8
    if ax is None:
        P.close("all")
        P.figure()
    super_imposed_img = cv2.addWeighted(image, 0.5, overlay, 0.5, 0)
    P.imshow(np.array(super_imposed_img), cmap='inferno')
    P.title(title, fontdict={"size": 18})
    if save:
        P.savefig(f"{outfile}/overlay-heatmap-{randstr()}-{id}.jpg")

transformer = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
def PreprocessImages(images):
    # assumes input is 4-D, with range [0,255]
    #
    # torchvision have color channel as first dimension
    # with normalization relative to mean/std of ImageNet:
    #    https://pytorch.org/vision/stable/models.html
    images = np.array(images)
    images = images/255
    images = np.transpose(images, (0,3,1,2))
    images = torch.tensor(images, dtype=torch.float32)
    # images = transformer.forward(images)
    return images.requires_grad_(True)

# def ShowImage(im, title='', ax=None, outfile="./output_xrai/", id=None):
#     if ax is None:
#         P.close("all")
#         P.figure()
#     P.axis('off')
#     P.imshow(im)
#     P.title(title, fontdict={"size": 18})
#     P.savefig(f"{outfile}/showimg-{id}-{randstr()}.jpg")


# def ShowGrayscaleImage(im, title='', ax=None, outfile="./output_xrai/"):
#     if ax is None:
#         P.close("all")
#         P.figure()
#     P.axis('off')
#     P.imshow(im, cmap=P.cm.gray, vmin=0, vmax=1)
#     P.title(title, fontdict={"size": 18})
#     P.colorbar()
#     P.savefig(f"{outfile}/grayscale-{randstr()}.jpg")


# def ShowHeatMap(im, title, ax=None, outfile="./output_xrai/", id=None):
#     if ax is None:
#         P.close("all")
#         P.figure()
#     P.axis('off')
#     a = P.imshow(im, cmap='inferno')
#     P.title(title, fontdict={"size": 18})
#     P.colorbar()
#     P.savefig(f"{outfile}/heatmap-{id}-{randstr()}.jpg")
#     return a


# def ShowOverlay(image, overlay, title, ax=None, outfile="./output_xrai/", id=None):
#     # np.array uint8
#     if ax is None:
#         P.close("all")
#         P.figure()
#     super_imposed_img = cv2.addWeighted(image, 0.5, overlay, 0.5, 0)
#     P.imshow(np.array(super_imposed_img), cmap='inferno')
#     P.title(title, fontdict={"size": 18})
#     P.savefig(f"{outfile}/overlay-heatmap-{randstr()}-{id}.jpg")


def LoadImage(file_path, resize=(299, 299)):
    im = PIL.Image.open(file_path)
    im = im.resize(resize)
    return np.asarray(im), im


transformer = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
def PreprocessImages(images, device=torch.device("cpu")):
    # assumes input is 4-D, with range [0,255]
    #
    # torchvision have color channel as first dimension
    # with normalization relative to mean/std of ImageNet:
    #    https://pytorch.org/vision/stable/models.html
    images = np.array(images)
    images = images/255
    images = np.transpose(images, (0,3,1,2))
    images = torch.tensor(images, dtype=torch.float32)
    # images = transformer.forward(images)
    images = images.to(device)
    return images.requires_grad_(True)


def get_idx(img_name):
    img_name = Path(img_name).stem
    idx = int(img_name.replace("sample-base-", ""))
    return idx


def uniformify(img_name):
    if ".jpg" not in img_name and os.path.exists(img_name + ".jpg"):
        return img_name + ".jpg"
    else:
        return img_name
