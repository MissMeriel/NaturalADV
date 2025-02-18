import sys
sys.path.append("../DAVE2-Keras")
import shutil
# https://kornia.readthedocs.io/en/latest/losses.html
from kornia.losses import psnr_loss
from kornia.losses import WelschLoss
from kornia.metrics import psnr
import torch
from torch.autograd import Variable
from torch import optim
import cv2
import numpy as np
import os
import pickle
import torchvision
from torchvision.transforms import Compose, ToPILImage, ToTensor, Resize, Lambda, Normalize
import torch.nn as nn
import torch.nn.functional as F
import PIL
from PIL import Image, ImageFilter
from pathlib import Path
import copy

from DAVE2pytorch import *
from utils import *
import argparse


def parse_args():
    parser = argparse.ArgumentParser(
                    prog='DM2CLEAR_PSNR',
                    description='What the program does',
                    epilog='Text at the bottom of help')


    parser.add_argument('-s', '--pertsize', type=int, default=100)
    parser.add_argument('-i', '--iters', type=int, default=100)
    parser.add_argument('-sw', '--criterionweight', type=float, default=1.)
    parser.add_argument('-pw', '--predweight', type=float, default=1.)
    parser.add_argument('-o', '--origpertfile', type=str, default="/p/sdbb/natadv/candidate_dm_bbs-clean/pert_billboard-sdbb-5-400-15-9RMDC6-NEW.jpg")      # option that takes a value
    parser.add_argument('-d', '--diffusionimage', type=str, default="CLEAR")
    parser.add_argument('-r', '--resultsparentdir', type=str, default="./results/")
    parser.add_argument('-b', '--blur', action='store_true')  # on/off flag
    return parser.parse_args()


def get_natural_pert(PERT_ID):

    SD_PERTS = {
        "MONDRIAN": "/p/sdbb/pert_billboard-0024/output-runwayml-stable-diffusion-v1-5/test-prompt-04-mondrian-museumbrochure.jpg",
        "ROUSSEAU": "/p/sdbb/pert_billboard-0024/output-runwayml-stable-diffusion-v1-5/test-prompt-03-henri rousseau-addictive.jpg",
        "GTA": "/p/sdbb/pert_billboard-0024/output-runwayml-stable-diffusion-v1-5-disabled-xformers/test-prompt-01-videogameadvertisement-detailed-grandtheftauto.jpg",
        "RACING": "/p/sdbb/pert_billboard-0024/output-runwayml-stable-diffusion-v1-5/test-prompt-01-videogame-advertisement-detailed-exciting-racing.jpg",
        "MUSEUM": "/p/sdbb/pert_billboard-0024/output-kandinsky-community-kandinsky-2-2-decoder/test-prompt-02-museum-detailed-colorful-artistic-modern art.jpg",
    }

    if PERT_ID == "CLEAR":
        sd_pert = get_clear_billboard(img_arr_pil[-1], mask_arr_pil[-1])
    else:
        sd_pert_file = SD_PERTS[PERT_ID]
        sd_pert = Image.open(sd_pert_file)
    return sd_pert


@ignore_warnings
def main(args):
    rid = randstr()
    blur = args.blur
    PERT_SIZE = (args.pertsize, args.pertsize)
    SD_IMG = args.diffusionimage
    iterations = args.iters

    # results_dir = f"./{args.resultsparentdir}/dm2clear-SIZE{PERT_SIZE[0]}-BLUR{blur}-iters{iterations}-{args.criterionweight}psnr-{args.predweight}pred-{SD_IMG}-{rid}/"
    results_dir = f"./{args.resultsparentdir}/naturalpatch{SD_IMG}-Welsch-{args.criterionweight}welsch-{args.predweight}pred-SIZE{PERT_SIZE[0]}-BLUR{blur}-iters{iterations}-{rid}/"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir, exist_ok=True)
        shutil.copy(__file__, f"{results_dir}/{Path(__file__).name}")
        shutil.copy("./utils.py", f"{results_dir}/utils.py")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # load trained DAVE2
    dave2 = torch.load("/p/sdbb/natadv/weights/model-DAVE2v3-lr1e4-100epoch-batch64-lossMSE-82Ksamples-INDUSTRIALandHIROCHIandUTAH-135x240-noiseflipblur.pt", map_location=torch.device('cpu'))
    dave2 = dave2.eval().to(device)
    print(dave2)

    masks_parent_dir = "/p/sdbb/natadv/input/sdbb-0001/masks/"
    images_parent_dir = "/p/sdbb/natadv/input/sdbb-0001/images/"
    paths_masks = [i for i in os.listdir(masks_parent_dir) if "sample-annot-0" in i]
    paths_images = [i for i in os.listdir(images_parent_dir) if "sample-0" in i]

    paths_masks = sorted(paths_masks, key=lambda x: int(x.split("-")[-1][:-4])) #[:30]
    paths_images = sorted(paths_images, key=lambda x: int(x.split("-")[-1][:-4]))[:len(paths_masks)]
    # indices = [0, 1, 2, 4, 5, 6, 7, 21, 22, 28, 30, -5, -4, -3, -2, -1]
    # indices=[1, 2, 3, 4, 5, 6, 7, 20, 21, 22, 23, 24, 25, 26, 29, 31, 48, 50, 61, 65, 67, 68, 69, 70]
    indices=[0, 1, 2, 20, 21, 22, 29, 67]
    paths_masks = [paths_masks[i] for i in indices]
    paths_images = [paths_images[i] for i in indices]
    
    img_arr_pil = [Image.open(images_parent_dir+i) for i in paths_images]
    mask_arr_pil = [Image.open(masks_parent_dir+i) for i in paths_masks]

    sd_pert = get_natural_pert(SD_IMG)
    original_pert = Image.open(args.origpertfile)

    if not PERT_SIZE:
        PERT_SIZE = sd_pert.size
    else:
        sd_pert = sd_pert.resize(PERT_SIZE, resample=Image.BICUBIC)
    original_pert = original_pert.resize(PERT_SIZE, resample=Image.NEAREST)

    if blur:
        # PERT_SIZE = (450, 450)
        sd_pert = sd_pert.resize(PERT_SIZE, resample=Image.BICUBIC)
        original_pert = original_pert.resize(PERT_SIZE, resample=Image.NEAREST)
        original_pert = original_pert.filter(ImageFilter.GaussianBlur(radius = 3))
    
    transform = Compose([ToTensor()])
    torch.autograd.set_detect_anomaly(True)

    sd_pert = get_clear_billboard(img_arr_pil[-1], mask_arr_pil[-1])
    if not PERT_SIZE:
        PERT_SIZE = sd_pert.size
    else:
        sd_pert = sd_pert.resize(PERT_SIZE, resample=Image.BICUBIC)
    original_pert = original_pert.resize(PERT_SIZE, resample=Image.NEAREST)
    print(f"{PERT_SIZE=}")
    if blur:
        # PERT_SIZE = (450, 450)
        sd_pert = sd_pert.resize(PERT_SIZE, resample=Image.BICUBIC)
        original_pert = original_pert.resize(PERT_SIZE, resample=Image.NEAREST)
        original_pert = original_pert.filter(ImageFilter.GaussianBlur(radius = 3)) 


    img_arr = np.transpose(np.array(img_arr_pil), (0, 3, 1, 2))
    # print(f"{img_arr.shape=}")
    # print(f"{img_arr.dtype=}")

    mask_arr = np.transpose(np.array(mask_arr_pil), (0, 3, 1, 2))
    # img_arr = transform(img_arr)
    # print(f"{img_arr.shape=}")
    # print(f"{len(mask_arr_pil)=}")
    img_patches, all_src_coords, all_patch_coords = [], [], []
    for i, mask in enumerate(mask_arr_pil):
        # print(f"\n{masks_parent_dir + paths_masks[i]}")
        # print(f"{mask.size=}")
        corners = get_qr_corners_from_colorseg_image(mask, debug=False, results_dir=results_dir)
        img_patches.append(corners)
        src_coords, patch_coords, image_pert = add_perturbed_billboard(img_arr[i], np.array(sd_pert), np.array(corners), debug=False, results_dir=results_dir)
        all_src_coords.append(src_coords)
        all_patch_coords.append(patch_coords)
        # print(f"{type(image_pert)=}\n{image_pert.shape=}")
        # cv2.imwrite(f"./{results_dir}/{paths_masks[i].replace('.jpg', '-pert.jpg')}", image_pert)
        # Image.fromarray(image_pert).save(f"./{results_dir}/{paths_masks[i].replace('.jpg', '-pert.jpg')}")

    all_src_coords = np.array(all_src_coords)
    all_src_coords = torch.tensor(np.squeeze(all_src_coords))
    all_patch_coords = torch.tensor(np.squeeze(np.array(all_patch_coords)))
    # print(f"{all_src_coords.shape=}\n{type(all_src_coords)=}")
    # print(f"{all_patch_coords.shape=}\n{type(all_patch_coords)=}")
    perspective_transforms = kornia.geometry.transform.get_perspective_transform(all_src_coords, all_patch_coords) #.to(device)

    dataset = DataSequence(img_arr, transform=transform)
    data_loader = data.DataLoader(dataset, batch_size=len(dataset))
    # shape = next(iter(data_loader)).shape.permute(1,2,0)
    orig_shape = img_arr.shape[2:]
    pert_shape = np.transpose(np.array(sd_pert), (2, 0, 1))[None].shape
    # print(f"{orig_shape=}\n{pert_shape=}")
    mask_patch = torch.ones(len(perspective_transforms), *pert_shape).squeeze().float().to(device)
    # print(f"{len(perspective_transforms)=}\n{pert_shape=}\n{mask_patch.shape=}")
    # perturbation = (transform(sd_pert)[None]).float().to(device)
    perturbation = (transform(original_pert)[None]).float().to(device)
    perturbation = torch.clamp(perturbation, min=0., max=1.)
    perturbation_start = torch.clone(perturbation)
    perturbation_sd_start = (transform(sd_pert)[None]).float().to(device)
    perturbation.requires_grad_() # = True
    # print(f"{perturbation.shape=}")
    perturbation_orig = (transform(original_pert)[None]).float().to(device)
    # print(f"{perturbation_orig.shape=}")
    initial_psnr_value = torch.mean(psnr(perturbation_sd_start, perturbation_orig, 2.0))
    print("INITIAL PSNR:", initial_psnr_value)
    
    perturbation_warp_orig = kornia.geometry.transform.warp_perspective(
        torch.vstack([perturbation_orig for _ in range(len(perspective_transforms))]),
        perspective_transforms,
        dsize=orig_shape,
        mode="nearest",
        align_corners=True
    )

    # print(f"\nwarp_masks warp_perspective\n{mask_patch.shape=}\n{perspective_transforms.shape=}\n{orig_shape=}")
    warp_masks = kornia.geometry.transform.warp_perspective(
        mask_patch, 
        perspective_transforms, 
        dsize=orig_shape, #np.transpose(orig_shape, (0, 2,3,1)),
        mode="bilinear",
        align_corners=True
    )

    # get the effect of the unaltered SD patch
    perturbation_warp_sd_start = kornia.geometry.transform.warp_perspective(
        torch.vstack([perturbation_sd_start for _ in range(len(perspective_transforms))]),
        perspective_transforms,
        dsize=orig_shape,
        mode="bilinear",
        align_corners=True
    )

    imgs = next(iter(data_loader)).to(device)
    imgs = torch.permute(imgs, (0, 2, 3, 1))
    imgs_pert_sd_start = imgs * (1 - warp_masks)
    imgs_pert_sd_start += perturbation_warp_sd_start * warp_masks
    imgs_pert_sd_start = torch.clamp(imgs_pert_sd_start, min=0., max=1.)
    preds_pert_sd_start = dave2(imgs_pert_sd_start).detach().numpy()
    # print(f"PREDICTIONS: preds_sd_start mean, var: {np.mean(preds_pert_sd_start):.4f}, {np.var(preds_pert_sd_start):.4f}")
    imgs_pert_orig = imgs * (1 - warp_masks)
    imgs_pert_orig += perturbation_warp_orig * warp_masks
    imgs_pert_orig = torch.clamp(imgs_pert_orig, min=0., max=1.)
    preds_pert_orig = dave2(imgs_pert_orig).detach().numpy()
    # optimizer = optim.Adam([perturbation], lr=1e-2)
    optimizer = optim.SGD([perturbation], lr=1e-2, momentum=0.9)
    psnr_value = initial_psnr_value
    criterion = WelschLoss(reduction="mean")
    mse_loss = nn.MSELoss()

    for it in range(iterations):
        optimizer.zero_grad()
        dave2.zero_grad()
        perturbation = perturbation.detach()
        perturbation = torch.clamp(perturbation, min=0., max=1.)
        # perturbation.requires_grad = True
        perturbation.requires_grad_()
        imgs = next(iter(data_loader)).to(device)
        # print(f"{imgs.shape=}")
        imgs = torch.permute(imgs, (0, 2, 3, 1)) #.size()
        # imgs = torch.clamp(imgs + torch.rand(imgs.shape) / 20, 0, 1)
        # print(f"\nperturbation_warp warp_perspective\n{imgs.shape=}\n{perspective_transforms.shape=}\n{orig_shape=}")
        perturbation_warp_sd = kornia.geometry.transform.warp_perspective(
            torch.vstack([perturbation for _ in range(len(perspective_transforms))]),
            perspective_transforms,
            dsize=orig_shape,
            mode="bilinear",
            align_corners=True
        )

        # print(f"{warp_masks.shape=}\n{imgs.shape=}")
        imgs_pert_sd = imgs * (1 - warp_masks)
        imgs_pert_sd += perturbation_warp_sd * warp_masks
        imgs_pert_sd = torch.clamp(imgs_pert_sd, min=0., max=1.)
        loss_out = -args.criterionweight * criterion(perturbation_sd_start, perturbation) - args.predweight * mse_loss(dave2(imgs_pert_sd), dave2(imgs_pert_orig))
        loss_out.backward()
        optimizer.step()
        criterion_value = criterion(perturbation_sd_start, perturbation)
        print(f"ITER {it:03d} loss={loss_out.item():.4f} pred loss={mse_loss(dave2(imgs_pert_sd), dave2(imgs_pert_orig)):.3f} welsch loss={criterion_value:.3f} mean grad={torch.mean(perturbation.grad).item()}")
        perturbation = torch.clamp(perturbation + torch.sign(perturbation.grad) / 100, 0., 1.)
        # print("CURRENT criterion:", criterion_value)
    
    print(f"{perturbation.shape=}")
    print(f"{perturbation_orig.shape=}")
    grid = torchvision.utils.make_grid(torch.cat([perturbation_sd_start, perturbation, perturbation_orig]))
    print(f"cat {grid.shape=}")
    torchvision.utils.save_image(grid, f"./{results_dir}/pert_change_after_{it:03d}iters-1.jpg")


    dataset1 = DataSequence(imgs)
    dataset2 = DataSequence(imgs_pert_orig)
    dataset3 = DataSequence(imgs_pert_sd.detach())
    dl1 = data.DataLoader(dataset1, batch_size=1)
    dl2 = data.DataLoader(dataset2, batch_size=1)
    dl3 = data.DataLoader(dataset3, batch_size=1)
    dl1_iter = iter(dl1)
    dl2_iter = iter(dl2)
    dl3_iter = iter(dl3)
    imgs_labelled_orig, imgs_labelled_pert_orig, imgs_labelled_pert_sd = [], [], []
    preds_orig, preds_pert_orig, preds_pert_sd = [], [], []
    for it, ims in enumerate(zip(dl1_iter, dl2_iter, dl3_iter)):
        # print(it, [i1.shape for i1 in ims])
        i1, i2, i3 = ims
        p1 = dave2(i1).item()
        p2 = dave2(i2).item()
        p3 = dave2(i3).item()
        preds_orig.append(p1)
        preds_pert_orig.append(p2)
        preds_pert_sd.append(p3)
        i1_labelled = add_text_to_image(i1, f"{p1:.3f}")
        imgs_labelled_orig.append(i1_labelled)
        i2_labelled = add_text_to_image(i2, f"{p2:.3f}")
        imgs_labelled_pert_orig.append(i2_labelled)
        i3_labelled = add_text_to_image(i3, f"{p3:.3f}")
        imgs_labelled_pert_sd.append(i3_labelled)

    grid = torchvision.utils.make_grid(torch.cat(imgs_labelled_orig))
    torchvision.utils.save_image(grid, f"./{results_dir}/FINAL-{rid}-imgs_labelled_orig.jpg")
    grid = torchvision.utils.make_grid(torch.cat(imgs_labelled_pert_orig))
    torchvision.utils.save_image(grid, f"./{results_dir}/FINAL-{rid}-imgs_labelled_pert_orig.jpg")
    grid = torchvision.utils.make_grid(torch.cat(imgs_labelled_pert_sd))
    torchvision.utils.save_image(grid, f"./{results_dir}/FINAL-{rid}-imgs_labelled_pert_sd.jpg")
    cp = copy.deepcopy(imgs_labelled_pert_orig)
    cp.extend(imgs_labelled_pert_sd)
    grid = torchvision.utils.make_grid(torch.cat(cp))
    torchvision.utils.save_image(grid, f"./{results_dir}/FINAL-{rid}-imgs_labelled_comparison.jpg")

    final_psnr_value = torch.mean(psnr(perturbation_sd_start, perturbation, 1.0))
    with open(f"./{results_dir}/FINAL-{rid}.pickle", "wb") as f:
        h = {
            "imgs": imgs,
            "imgs_pert_orig": imgs_pert_orig,
            "imgs_pert_sd": imgs_pert_sd.detach(),
            "imgs_pert_sd_start": imgs_pert_sd_start, 
            "preds_orig": preds_orig,
            "preds_pert_orig": preds_pert_orig,
            "preds_pert_sd": preds_pert_sd,
            "preds_pert_sd_start": preds_pert_sd_start,
            "original_pert": original_pert,
            "sd_pert": sd_pert,
            "iterations": iterations,
            "initial_psnr_value": initial_psnr_value,
            "final_psnr_value": final_psnr_value,
        }
        pickle.dump(h, f)
    print(f"PREDICTIONS:"\
        f"\nUnpert mean, var:                    \t{np.mean(preds_orig):.4f}, {np.var(preds_orig):.4f}"\
        f"\nOrig pert mean, var:                 \t{np.mean(preds_pert_orig):.4f}, {np.var(preds_pert_orig):.4f}"\
        f"\nOrig pert vs. unpert MSignedE:       \t{np.mean([preds_pert_orig[i]-preds_orig[i] for i in range(len(preds_pert_orig))]):.4f}, {np.var([preds_pert_orig[i]-preds_orig[i] for i in range(len(preds_pert_orig))]):.4f}"\
        f"\nSD pert start mean, var:             \t{np.mean(preds_pert_sd_start):.4f}, {np.var(preds_pert_sd_start):.4f}"\
        f"\nSD pert end mean, var:               \t{np.mean(preds_pert_sd):.4f}, {np.var(preds_pert_sd):.4f}"\
        f"\nSD pert end vs orig pert MAbsE:      \t{np.mean([abs(preds_pert_sd[i]-preds_pert_orig[i]) for i in range(len(preds_pert_orig))]):.4f}, {np.var([abs(preds_pert_sd[i]-preds_pert_orig[i]) for i in range(len(preds_pert_orig))]):.4f}"\
        f"\nSD pert end vs orig pert MSignedE:   \t{np.mean([preds_pert_sd[i]-preds_pert_orig[i] for i in range(len(preds_pert_orig))]):.4f}, {np.var([preds_pert_sd[i]-preds_pert_orig[i] for i in range(len(preds_pert_orig))]):.4f}"\
        f"\nSD pert end vs unpert MAbsE:         \t{np.mean([abs(preds_pert_sd[i]-preds_orig[i]) for i in range(len(preds_pert_orig))]):.4f}, {np.var([abs(preds_pert_sd[i]-preds_orig[i]) for i in range(len(preds_pert_orig))]):.4f}"\
        f"\nSD pert end vs unpert MSignedE:      \t{np.mean([preds_pert_sd[i]-preds_orig[i] for i in range(len(preds_pert_orig))]):.4f}, {np.var([preds_pert_sd[i]-preds_orig[i] for i in range(len(preds_pert_orig))]):.4f}"
            )
    print(f"INITIAL PATCH psnrs (dm vs. sd):  {initial_psnr_value:.2f}")
    print(f"FINAL PATCH psnrs (dm vs. sd):  {final_psnr_value:.2f}")
    print(f"Results saved to {results_dir}")


if __name__ == '__main__':
    args = parse_args()
    main(args)