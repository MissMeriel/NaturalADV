import string
import random
import time
import numpy as np
import cv2
from PIL import Image, ImageFont, ImageDraw
import skimage
import kornia
import torch
import torch.utils.data as data
import torch.nn.functional as F
from torch import nn
import torchvision
import warnings
from functools import wraps
import torch.utils.data as data
import argparse


call_count = 6


def ignore_warnings(f):
    @wraps(f)
    def inner(*args, **kwargs):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("ignore")
            response = f(*args, **kwargs)
        return response
    return inner


def randstr():
    return ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(6))


def timestr():
    localtime = time.localtime()
    return "{}_{}-{}_{}".format(localtime.tm_mon, localtime.tm_mday, localtime.tm_hour, localtime.tm_min)



def get_clear_billboard(img: Image, mask: Image):
    keypoints = get_qr_corners_from_colorseg_image(mask)[0]
    # print(f"{keypoints=}")
    left = min(keypoints[0][0], keypoints[3][0])
    top = min(keypoints[0][1], keypoints[1][1])
    right = max(keypoints[2][0], keypoints[1][0])
    bottom = max(keypoints[2][1], keypoints[3][1])
    dims = (int(bottom - top), int(right - left))
    if dims[0] > dims[1]:
        diff = dims[0] - dims[1]
        right = right + diff
    elif dims[1] > dims[0]:
        diff = dims[1] - dims[0]
        bottom = bottom + diff
    dims = (int(bottom - top), int(right - left))
    right = right - (dims[0] % 5)
    bottom = bottom - (dims[1] % 5)
    pert = img.crop((left, top, right, bottom))
    # warp back into a square?
    print(f"CLEAR BILLBOARD COORDS={(left, top, right, bottom)} {(bottom - top, right - left)}")
    print(f"CLEAR BILLBOARD SIZE={(int(bottom - top), int(right - left))}")
    # pert.save("./test_get_clear_billboard.jpg")
    return pert, (int(bottom - top), int(right - left))


def get_clear_billboard2(img: Image, mask: np.array):
    keypoints = mask
    img.save("./clear-bb-test.jpg")
    # print(f"{keypoints=}")
    # print(f"{keypoints.shape=}")
    # keypoints = keypoints[0]
    left = min(keypoints[0][0], keypoints[3][0])
    top = min(keypoints[0][1], keypoints[1][1])
    right = max(keypoints[2][0], keypoints[1][0])
    bottom = max(keypoints[2][1], keypoints[3][1])
    pert = img.crop((left, top, right, bottom))
    # warp back into a square?
    pert.save("./test_get_clear_billboard.jpg")
    return pert


def add_text_to_image(img: torch.Tensor, t1: str):
    font = ImageFont.truetype("/p/sdbb/natadv/font/ShareTechMono-Regular.ttf", 20)
    # indices, channels, heights, widths = img.shape
    # x_offset = 0
    im = Image.fromarray(np.uint8(img[0].numpy().transpose(1,2,0) * 255))
    draw = ImageDraw.Draw(im)
    draw.text((0, 0), t1, (255,0,0), font=font)
    t1 = torch.Tensor((np.array(im) / 255).transpose(2,0,1))[None]
    # print(torch.min(t1), torch.max(t1))
    return t1.float()


def save_images(img1, img2, t1, t2, outpath="./sample.jpg"):
    font = ImageFont.truetype("/p/sdbb/natadv/font/ShareTechMono-Regular.ttf", 20)
    indices, channels, heights, widths = zip(*(i.shape for i in [img1, img2]))
    total_width = sum(widths)
    max_height = max(heights)
    new_im = Image.new('RGB', (total_width, max_height))
    x_offset = 0
    for im, t in zip([img1, img2], [t1, t2]):
        im = Image.fromarray(im[0].transpose(1,2,0))
        draw = ImageDraw.Draw(im)
        # ImageDraw.Draw(im).text((0, 0), t, (0, 0, 0))
        draw.text((0, 0), t, (255,0,0), font=font)
        new_im.paste(im, (x_offset,0))
        x_offset += im.size[0]
    new_im.save(outpath)


def add_perturbed_billboard(img, bb, qr_corners, debug=False, results_dir="results"):
    img = overlay_transparent(np.array(img), bb, np.asarray(qr_corners), debug=debug, results_dir=results_dir )
    return img

@ignore_warnings
def overlay_transparent(img1, img2, corners, debug=False, results_dir="results"):
    global call_count
    orig = torch.from_numpy(img1)[None] / 255 #.permute(0, 3, 1, 2) / 255.0
    pert = torch.from_numpy(img2)[None].permute(0, 3, 1, 2) / 255.0
    
    _, c, h, w = _, *pert_shape = pert.shape
    _, *orig_shape = orig.shape
    if debug:
        print(f"{pert.shape=}")
        print(f"{orig_shape=}")
        print(f"{corners.shape=}")
    if len(corners.shape) == 2:
        patch_coords = corners[None] #np.transpose(corners, (1, 0, 2))[None]
    else:
        patch_coords = corners
    if debug:
        print(f"{patch_coords.shape=}")
    src_coords = np.tile(
        np.array(
            [
                [
                    [0.0, 0.0],
                    [w - 1.0, 0.0],
                    [0.0, h - 1.0],
                    [w - 1.0, h - 1.0],
                ]
            ]
        ),
        (len(patch_coords), 1, 1),
    )
    if debug:
        print(f"{src_coords.shape=}")
    src_coords = torch.from_numpy(src_coords).float()
    patch_coords = torch.from_numpy(patch_coords).float()
    
    # build the transforms to and from image patches
    try:
        perspective_transforms = kornia.geometry.transform.imgwarp.get_perspective_transform(src_coords, patch_coords[0])
    except:
        perspective_transforms = kornia.geometry.transform.imgwarp.get_perspective_transform(src_coords, patch_coords)

    perturbation_warp = kornia.geometry.transform.warp_perspective(
        pert,
        perspective_transforms,
        dsize=orig_shape[1:],
        mode="nearest",
        align_corners=True
    )
    mask_patch = torch.ones(1, *pert_shape)
    warp_masks = kornia.geometry.transform.warp_perspective(
        mask_patch, perspective_transforms, dsize=orig_shape[1:],
        mode="nearest",
        align_corners=True
    )
    perturbed_img = orig * (1 - warp_masks)
    perturbed_img += perturbation_warp * warp_masks
    if debug:
        torchvision.utils.save_image(perturbed_img, f"./{results_dir}/{call_count:03d}.jpg")
    return src_coords, patch_coords, (perturbed_img.permute(0, 2, 3, 1).numpy()[0] * 255).astype(np.uint8)


def mask2bw(image: Image, debug=False, results_dir="results"):
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image = cv2.GaussianBlur(image, (3,3), 0)
    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    light_color = (50, 190, 0)  # (50, 235, 235) #(0, 200, 0)
    dark_color = (90, 256, 256)  # (70, 256, 256) #(169, 256, 256)
    mask = cv2.inRange(hsv_image, light_color, dark_color)
    image = cv2.bitwise_and(image, image, mask=mask)
    if debug:
        cv2.imwrite(f"./{results_dir}/{call_count:03d}-{random_id}-bitwiseand.jpg", image)
    # convert image to inverted greyscale
    R, G, B = image[:, :, 0], image[:, :, 1], image[:, :, 2]
    imgGray = np.uint8(0.2989 * R + 0.5870 * G + 0.1140 * B)
    imgGrayInverted = np.uint8(255 - imgGray)
    ret, thresh = cv2.threshold(imgGrayInverted, 200, 256, cv2.THRESH_BINARY)
    thresh = np.uint8(255 - thresh)
    return thresh


# Shi-Tomasi corner detector: https://docs.opencv.org/4.10.0/d8/dd8/tutorial_good_features_to_track.html
@ignore_warnings
def get_qr_corners_from_colorseg_image(image: Image, debug=False, results_dir="results"):
    global call_count
    random_id = '000' #randstr()
    thresh = mask2bw(image)
    if debug:
        cv2.imwrite(f"./{results_dir}/{call_count:03d}-{random_id}-thresh.jpg", thresh)
    
    # Parameters for Shi-Tomasi algorithm
    maxCorners = max(4, 1)
    qualityLevel = 0.0001 #0.01
    minDistance = 10
    blockSize = 3
    gradientSize = 3
    useHarrisDetector = False
    k = 0.04
    # copy = np.copy(image)
 
    # Apply corner detection
    corners = cv2.goodFeaturesToTrack(thresh, maxCorners, qualityLevel, minDistance, None, \
        blockSize=blockSize, gradientSize=gradientSize, useHarrisDetector=useHarrisDetector, k=k)

    def sortClockwise(approx):
        approx = np.squeeze(approx)
        if debug:
            print(f"{approx=}\n{approx.shape=}")
        xs = [a[0] for a in approx]
        ys = [a[1] for a in approx]
        center = [int(sum(xs) / len(xs)), int(sum(ys) / len(ys))]
        def sortFxnX(e):
            return e[0]
        def sortFxnY(e):
            return e[1]
        approx = list(approx)
        approx.sort(key=sortFxnX)
        midpt = int(len(approx) / 2)
        leftedge = list(approx[:midpt])
        rightedge = list(approx[midpt:])
        leftedge.sort(key=sortFxnY)
        rightedge.sort(key=sortFxnY)
        approx = np.array([leftedge[0], leftedge[1], rightedge[1], rightedge[0]])
        return approx, leftedge, rightedge, center

    corners, leftedge, rightedge, center = sortClockwise(corners)
    if debug:
        # Draw corners detected
        print('** Number of corners detected:', corners.shape[0], f"{corners.shape=}")
        radius = 2
        colors = [(0,0,255), (0,255,0), (255,0,0), (100, 100, 100), (100, 22, 40)]
        for i in range(corners.shape[0]):
            cv2.circle(copy, (int(corners[i,0]), int(corners[i,1])), radius, colors[i], cv2.FILLED)
        cv2.imwrite(f"./{results_dir}/{call_count:03d}-{random_id}-Shi-Tomasi.jpg", copy)
    keypoints = [[tuple(corners[0]), tuple(corners[3]),
                tuple(corners[1]), tuple(corners[2])]]
    call_count += 1
    return keypoints


class DataSequence(data.Dataset):
    def __init__(self, img_array, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.img_arr = img_array
        self.transform = transform

    def __len__(self):
        return len(self.img_arr)

    def __getitem__(self, idx):
        sample = np.array(self.img_arr[idx])
        if self.transform:
            sample = self.transform(sample)

        return sample