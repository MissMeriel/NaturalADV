import string
import random
import time
import numpy as np
import cv2
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw 
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

print(f"{cv2.__version__=}")
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


def save_images(img1, img2, t1, t2, outpath="./sample.jpg"):
    font = ImageFont.truetype("./font/ShareTechMono-Regular.ttf", 20)
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


def add_perturbed_billboard(img, bb, qr_corners):
    img = overlay_transparent(np.array(img), bb, np.asarray(qr_corners))
    return img


def overlay_transparent(img1, img2, corners):
    global call_count
    orig = torch.from_numpy(img1)[None] / 255 #.permute(0, 3, 1, 2) / 255.0
    pert = torch.from_numpy(img2)[None].permute(0, 3, 1, 2) / 255.0
    print(f"{pert.shape=}")
    _, c, h, w = _, *pert_shape = pert.shape
    _, *orig_shape = orig.shape
    print(f"{orig_shape=}")
    print(f"{corners.shape=}")
    patch_coords = np.transpose(corners, (1, 0, 2))[None]
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
    src_coords = torch.from_numpy(src_coords).float()
    patch_coords = torch.from_numpy(patch_coords).float()
    print(f"{src_coords.shape=}\n{patch_coords.shape=}")
    # build the transforms to and from image patches
    perspective_transforms = kornia.geometry.transform.imgwarp.get_perspective_transform(src_coords, patch_coords[0])

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
    torchvision.utils.save_image(perturbed_img, f"./deletetestout/{call_count:03d}.jpg")
    return (perturbed_img.permute(0, 2, 3, 1).numpy()[0] * 255).astype(np.uint8)



# Shi-Tomasi corner detector: https://docs.opencv.org/4.10.0/d8/dd8/tutorial_good_features_to_track.html
@ignore_warnings
def get_qr_corners_from_colorseg_image6(image):
    global call_count
    random_id = '000' #randstr()
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image = cv2.GaussianBlur(image, (3,3), 0)
    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    light_color = (50, 190, 0)  # (50, 235, 235) #(0, 200, 0)
    dark_color = (90, 256, 256)  # (70, 256, 256) #(169, 256, 256)
    mask = cv2.inRange(hsv_image, light_color, dark_color)
    image = cv2.bitwise_and(image, image, mask=mask)
    cv2.imwrite(f"./deletetestout6/{call_count:03d}-{random_id}-bitwiseand.jpg", image)
    # convert image to inverted greyscale
    R, G, B = image[:, :, 0], image[:, :, 1], image[:, :, 2]
    imgGray = np.uint8(0.2989 * R + 0.5870 * G + 0.1140 * B)
    imgGrayInverted = np.uint8(255 - imgGray)
    ret, thresh = cv2.threshold(imgGrayInverted, 200, 256, cv2.THRESH_BINARY)
    thresh = np.uint8(255 - thresh)
    cv2.imwrite(f"./deletetestout6/{call_count:03d}-{random_id}-thresh.jpg", thresh)
    maxCorners = max(4, 1)
 
    # Parameters for Shi-Tomasi algorithm
    qualityLevel = 0.01
    minDistance = 10
    blockSize = 3
    gradientSize = 3
    useHarrisDetector = False
    k = 0.04
 
    # Copy the source image
    copy = np.copy(image)
 
    # Apply corner detection
    corners = cv2.goodFeaturesToTrack(thresh, maxCorners, qualityLevel, minDistance, None, \
        blockSize=blockSize, gradientSize=gradientSize, useHarrisDetector=useHarrisDetector, k=k)


    def sortClockwise(approx):
        approx = np.squeeze(approx)
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
    # Draw corners detected
    print('** Number of corners detected:', corners.shape[0], f"{corners.shape=}")
    radius = 4
    colors = [(0,0,255), (0,255,0), (255,0,0), (100, 100, 100), (100, 22, 40)]
    for i in range(corners.shape[0]):
        cv2.circle(copy, (int(corners[i,0]), int(corners[i,1])), radius, colors[i], cv2.FILLED)
    
    # Show what you got
    cv2.imwrite(f"./deletetestout6/{call_count:03d}-{random_id}-Shi-Tomasi.jpg", copy)
    keypoints = [[tuple(corners[0]), tuple(corners[3]),
                tuple(corners[1]), tuple(corners[2])]]
    call_count += 1
    return keypoints

# cornerHarris_demo: https://docs.opencv.org/3.4/d4/d7d/tutorial_harris_detector.html
@ignore_warnings
def get_qr_corners_from_colorseg_image5(image):
    global call_count
    random_id = '000' #randstr()
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image = cv2.GaussianBlur(image, (3,3), 0)
    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    light_color = (50, 190, 0)  # (50, 235, 235) #(0, 200, 0)
    dark_color = (90, 256, 256)  # (70, 256, 256) #(169, 256, 256)
    mask = cv2.inRange(hsv_image, light_color, dark_color)
    image = cv2.bitwise_and(image, image, mask=mask)
    cv2.imwrite(f"./deletetestout5/{call_count:03d}-{random_id}-bitwiseand.jpg", image)
    # convert image to inverted greyscale
    R, G, B = image[:, :, 0], image[:, :, 1], image[:, :, 2]
    imgGray = np.uint8(0.2989 * R + 0.5870 * G + 0.1140 * B)
    imgGrayInverted = np.uint8(255 - imgGray)
    ret, thresh = cv2.threshold(imgGrayInverted, 200, 256, cv2.THRESH_BINARY)
    thresh = np.uint8(255 - thresh)
    cv2.imwrite(f"./deletetestout5/{call_count:03d}-{random_id}-thresh.jpg", thresh)
    # Detector parameters
    blockSize = 2
    apertureSize = 3
    k = 0.04
    # Detecting corners
    dst = cv2.cornerHarris(thresh, blockSize, apertureSize, k)
    print(f"{dst=}")
    threshold_val = 100
    # Normalizing
    dst_norm = np.empty(dst.shape, dtype=np.float32)
    cv2.normalize(dst, dst_norm, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    dst_norm_scaled = cv2.convertScaleAbs(dst_norm)
    # Drawing a circle around corners
    for i in range(dst_norm.shape[0]):
        for j in range(dst_norm.shape[1]):
            if int(dst_norm[i,j]) > threshold_val:
                cv2.circle(dst_norm_scaled, (j,i), 5, (0), 2)
    cv2.imwrite(f"./deletetestout4/{call_count:03d}-{random_id}-cornerHarris.jpg", dst_norm_scaled)
    keypoints = [[tuple(dst[0]), tuple(dst[3]),
                tuple(dst[1]), tuple(dst[2])]]
    call_count += 1
    return keypoints

@ignore_warnings
def get_qr_corners_from_colorseg_image4(image):
    global call_count
    random_id = '000' #randstr()
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image = cv2.GaussianBlur(image, (3,3), 0)
    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    light_color = (50, 190, 0)  # (50, 235, 235) #(0, 200, 0)
    dark_color = (90, 256, 256)  # (70, 256, 256) #(169, 256, 256)
    mask = cv2.inRange(hsv_image, light_color, dark_color)
    image = cv2.bitwise_and(image, image, mask=mask)
    cv2.imwrite(f"./deletetestout4/{call_count:03d}-{random_id}-bitwiseand.jpg", image)
    # convert image to inverted greyscale
    R, G, B = image[:, :, 0], image[:, :, 1], image[:, :, 2]
    imgGray = np.uint8(0.2989 * R + 0.5870 * G + 0.1140 * B)
    imgGrayInverted = np.uint8(255 - imgGray)
    ret, thresh = cv2.threshold(imgGrayInverted, 200, 256, cv2.THRESH_BINARY)
    cv2.imwrite(f"./deletetestout4/{call_count:03d}-{random_id}-thresh.jpg", thresh)
    # Convert image to gray and blur it
    # src_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # src_gray = cv2.blur(src_gray, (3,3))
    # cv2.createTrackbar('Canny thresh:', source_window, 100, 255, thresh_callback)
    threshold = 100
    # Detect edges using Canny
    canny_output = cv2.Canny(thresh, threshold, 256)
    cv2.imwrite(f"./deletetestout4/{call_count:03d}-{random_id}-canny_output.jpg", canny_output)
    # Find contours
    contours, _ = cv2.findContours(canny_output, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # Find the convex hull object for each contour
    hull_list = []
    for i in range(len(contours)):
        hull = cv2.convexHull(contours[i])
        print(f"hull {i}={hull}")
        hull_list.append(hull)
    # Draw contours + hull results
    drawing = np.zeros((canny_output.shape[0], canny_output.shape[1], 3), dtype=np.uint8)
    for i in range(len(contours)):
        color = (random.randint(100,256), random.randint(100,256), random.randint(100,256))
        cv2.drawContours(drawing, contours, i, color)
        cv2.drawContours(drawing, hull_list, i, color)
    # Show in a window
    cv2.imwrite(f"./deletetestout4/{call_count:03d}-{random_id}-Contours.jpg", drawing)
    call_count += 1
    return drawing


# uses convex hull
@ignore_warnings
def get_qr_corners_from_colorseg_image3(image):
    global call_count
    image = np.array(image)
    random_id = randstr()
    image = cv2.GaussianBlur(image, (3,3), 0)
    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    light_color = (50, 230, 0)  # (50, 235, 235) #(0, 200, 0)
    dark_color = (90, 256, 256)  # (70, 256, 256) #(169, 256, 256)
    mask = cv2.inRange(hsv_image, light_color, dark_color)
    image = cv2.bitwise_and(image, image, mask=mask)
    cv2.imwrite(f"./deletetestout3/{call_count:03d}-{random_id}-bitwiseand.jpg", image)
    # convert image to inverted greyscale
    R, G, B = image[:, :, 0], image[:, :, 1], image[:, :, 2]
    imgGray = np.uint8(0.2989 * R + 0.5870 * G + 0.1140 * B)
    imgGrayInverted = np.uint8(255 - imgGray)
    ret, thresh = cv2.threshold(imgGrayInverted, 150, 255, cv2.THRESH_BINARY)
    cv2.imwrite(f"./deletetestout3/{call_count:03d}-{random_id}-inverted-uint8.jpg", imgGrayInverted)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    drawing = np.zeros((contours.shape[0], contours.shape[1], 3), dtype=np.uint8)

    hull = cv2.convexHull(contours[0])
    temp = cv2.drawContours(image, [hull], 0, (255,255,255), 3)
    cv2.imwrite(f"./deletetestout3/{call_count:03d}-{random_id}-drawhullcontours.jpg", temp)
    epsilon = 0.1 * cv2.arcLength(np.float32(contours[1]), True)
    approx = cv2.approxPolyDP(contours[0], epsilon, True)
    temp = cv2.drawContours(image, [approx], 0, (255,255,255), 3)
    cv2.imwrite(f"./deletetestout3/{call_count:03d}-{random_id}-drawcontours2.jpg", temp)
    keypoints = [[tuple(approx[0]), tuple(approx[3]),
                tuple(approx[1]), tuple(approx[2])]]
    call_count += 1
    return keypoints

# uses convex hull
@ignore_warnings
def get_qr_corners_from_colorseg_image2(image):
    global call_count
    image = np.array(image)
    random_id = randstr()
    image = cv2.GaussianBlur(image, (3,3), 0)
    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    print(f"{hsv_image.shape=}")
    # unique_vals = np.unique(hsv_image, axis=0)
    # print(f"axis=0 {unique_vals=} \n{unique_vals.shape=} \n{unique_vals[0].shape=}")
    # unique_vals = np.unique(hsv_image, axis=1)
    # print(f"axis=1 {unique_vals=} \n{unique_vals.shape=} \n{unique_vals[0].shape=}")
    # unique_vals = np.unique(hsv_image, axis=2)
    # print(f"axis=2 {unique_vals=} \n{unique_vals.shape=} \n{unique_vals[0].shape=}")
    # light_color = (50, 230, 0)  # (50, 235, 235) #(0, 200, 0)
    # dark_color = (90, 256, 256)  # (70, 256, 256) #(169, 256, 256)
    light_color = (50, 230, 0)  # (50, 235, 235) #(0, 200, 0)
    dark_color = (65, 256, 256)  # (70, 256, 256) #(169, 256, 256)
    mask = cv2.inRange(hsv_image, light_color, dark_color)
    image = cv2.bitwise_and(image, image, mask=mask)
    cv2.imwrite(f"./deletetestout2/{call_count:03d}-{random_id}-bitwiseand.jpg", image)
    # convert image to inverted greyscale
    R, G, B = image[:, :, 0], image[:, :, 1], image[:, :, 2]
    imgGray = np.uint8(0.2989 * R + 0.5870 * G + 0.1140 * B)
    imgGrayInverted = np.uint8(255 - imgGray)
    ret, thresh = cv2.threshold(imgGrayInverted, 150, 255, cv2.THRESH_BINARY)
    cv2.imwrite(f"./deletetestout2/{call_count:03d}-{random_id}-inverted-uint8.jpg", imgGrayInverted)
    cv2.imwrite(f"./deletetestout2/{call_count:03d}-{random_id}-thresh.jpg", thresh)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(f"{type(contours)=}", f"{type(contours[0])=}", f"{len(contours)=}")
    print([f"{type(contours[i])=} {contours[i].shape=}" for i in range(len(contours))])
    temp = cv2.drawContours(image, contours, -1, (0,255,0), 3)
    # contours = sorted(contours, key=lambda x: len(x))
    cv2.imwrite(f"./deletetestout2/{call_count:03d}-{random_id}-drawcontours.jpg", temp)
    hull = cv2.convexHull(contours[0])
    temp = cv2.drawContours(image, [hull], 0, (255,255,255), 3)
    cv2.imwrite(f"./deletetestout2/{call_count:03d}-{random_id}-drawhullcontours.jpg", temp)
    epsilon = 0.1 * cv2.arcLength(np.float32(contours[1]), True)
    approx = cv2.approxPolyDP(contours[0], epsilon, True)
    temp = cv2.drawContours(image, [approx], 0, (255,255,255), 3)
    cv2.imwrite(f"./deletetestout2/{call_count:03d}-{random_id}-drawcontours2.jpg", temp)
    keypoints = [[tuple(approx[0]), tuple(approx[3]),
                tuple(approx[1]), tuple(approx[2])]]
    call_count += 1
    return keypoints

# uses contour detection
@ignore_warnings
def get_qr_corners_from_colorseg_image(image):
    global call_count
    image = np.array(image)
    random_id = randstr()
    image = cv2.GaussianBlur(image, (3,3), 0)
    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    light_color = (50, 230, 0)  # (50, 235, 235) #(0, 200, 0)
    dark_color = (90, 256, 256)  # (70, 256, 256) #(169, 256, 256)
    mask = cv2.inRange(hsv_image, light_color, dark_color)
    image = cv2.bitwise_and(image, image, mask=mask)
    cv2.imwrite(f"./deletetestout/{call_count:03d}-{random_id}-bitwiseand.jpg", image)
    # convert image to inverted greyscale
    R, G, B = image[:, :, 0], image[:, :, 1], image[:, :, 2]
    imgGray = 0.2989 * R + 0.5870 * G + 0.1140 * B
    imgGrayInverted = np.uint8(255 - imgGray)
    cv2.imwrite(f"./deletetestout/{call_count:03d}-{random_id}-inverted-uint8.jpg", imgGrayInverted)
    ret, thresh = cv2.threshold(imgGrayInverted, 150, 255, cv2.THRESH_BINARY)
    cv2.imwrite(f"./deletetestout/{call_count:03d}-{random_id}-thresh.jpg", thresh)
    contours, hierarchy = cv2.findContours(image=thresh, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)
    print(f"{len(contours)=}")


    if contours == []: # or np.array(contours).shape[0] < 2:
        print(f"{contours=}, returning early placeholder")
        call_count += 1
        return [[[0, 0], [0, 0], [0, 0], [0, 0]]], None
    else:
        epsilon = 0.1 * cv2.arcLength(np.float32(contours[1]), True)
        approx = cv2.approxPolyDP(np.float32(contours[1]), epsilon, True)
        contours = np.array([c[0] for c in contours[1]])
        approx = [c[0] for c in approx][:4]
        # hull = cv2.convexHull(contours[0])
        # temp = cv2.drawContours(image, [approx], 0, (255,255,255), 3)
        # cv2.imwrite(f"./deletetestout/{call_count:03d}-{random_id}-convexhull.jpg", temp)
        # print("convex hull has ",len(hull),"points")
        # contours = contours.reshape((contours.shape[0], 2))
        if len(approx) < 4:
            print(f"{contours.shape=}, {approx=}, returning early placeholder")
            return [[[0, 0], [0, 0], [0, 0], [0, 0]]], None
        
        def sortClockwise(approx):
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
            approx = [leftedge[0], leftedge[1], rightedge[1], rightedge[0]]
            return approx, leftedge, rightedge, center

        approx, le, re, center = sortClockwise(approx)
        for i,c in enumerate(le):
            cv2.circle(image, tuple([int(x) for x in c]), radius=1, color=(100 + i*20, 0, 0), thickness=2) # blue
        for i,c in enumerate(re):
            cv2.circle(image, tuple([int(x) for x in c]), radius=1, color=(0, 0, 100 + i*20), thickness=2)  # blue
        cv2.circle(image, tuple(center), radius=1, color=(203,192,255), thickness=2)  # lite pink
        if len(approx) > 3:
            cv2.circle(image, tuple([int(x) for x in approx[0]]), radius=1, color=(0, 255, 0), thickness=2) # green
            cv2.circle(image, tuple([int(x) for x in approx[2]]), radius=1, color=(0, 0, 255), thickness=2) # red
            cv2.circle(image, tuple([int(x) for x in approx[3]]), radius=1, color=(255, 255, 255), thickness=2) # white
            cv2.circle(image, tuple([int(x) for x in approx[1]]), radius=1, color=(147,20,255), thickness=2)# pink
        cv2.imwrite(f"./deletetestout/{call_count:03d}-{random_id}-bbedges.jpg", image)
        keypoints = [[tuple(approx[0]), tuple(approx[3]),
                      tuple(approx[1]), tuple(approx[2])]]
        call_count += 1
        return keypoints, image

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