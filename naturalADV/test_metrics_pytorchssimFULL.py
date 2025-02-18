import sys
sys.path.append("pytorch-ssim")
sys.path.append("../DAVE2-Keras")
import shutil
# https://kornia.readthedocs.io/en/latest/losses.html
# import pytorch_ssim
from pytorch_ssim import SSIM, ssim
import torch
from torch.autograd import Variable
from torch import optim
import cv2
import numpy as np
import os
import pickle
import torchvision
from torchvision.transforms import v2, Compose, ToPILImage, ToTensor, Resize, Lambda, Normalize
import torch.nn as nn
import torch.nn.functional as F
import PIL
from PIL import Image, ImageFilter
from pathlib import Path
import copy

from DAVE2pytorch import *
from natadv_utils import *
import argparse

PRED_MEAN = 0.0
SSIM_MEAN = .963 #0.95
SSIM_TO_PRED_VAR_FACTOR = 1.
USE_DM_PERT = False
topo = "topo1-left"
def parse_args():
    parser = argparse.ArgumentParser(
                    prog='ProgramName',
                    description='What the program does',
                    epilog='Text at the bottom of help')


    parser.add_argument('-s', '--pertsize', type=int, default=50)
    parser.add_argument('-i', '--iters', type=int, default=100)
    parser.add_argument('-sw', '--ssimweight', type=float, default=1.)
    parser.add_argument('-pw', '--predweight', type=float, default=1.)
    parser.add_argument('-o', '--origpertfile', type=str, default="/p/sdbb/natadv/candidate_dm_bbs-clean/pert_billboard-sdbb-5-400-15-9RMDC6-NEW.jpg")      # option that takes a value
    parser.add_argument('-d', '--diffusionimage', type=str, default="CLEAR")
    parser.add_argument('-r', '--resultsparentdir', type=str, default="./results/")
    parser.add_argument('-n', '--addnoise', action='store_true')
    parser.add_argument('-b', '--blur', action='store_true')  # on/off flag
    parser.add_argument('-u', '--useunpertseq', action='store_true')  # on/off flag
    parser.add_argument('-w', '--weighted', action='store_true')  # on/off flag
    parser.add_argument('-sp', '--usedmpert', action='store_true')  # on/off flag

    return parser.parse_args()


@ignore_warnings
def main(args, num_perts):
    global USE_DM_PERT, PRED_MEAN, SSIM_MEAN, SSIM_TO_PRED_VAR_FACTOR
    rid = randstr()
    PERT_SIZE = (args.pertsize, args.pertsize)
    SD_IMG = Path(args.diffusionimage).stem
    USE_DM_PERT = args.usedmpert
    ssimdiv = 2
    results_dir = f"./{args.resultsparentdir}/{SD_IMG}-{topo}-ssimdiv{ssimdiv}-USE_DM_PERT{USE_DM_PERT}-{args.ssimweight}ssim-{args.predweight}pred-noised{args.addnoise}-weighted{args.weighted}-useunpertseq{args.useunpertseq}-SIZE{PERT_SIZE[0]}-BLUR{args.blur}-iters{args.iters}-{rid}/"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir, exist_ok=True)
        shutil.copy(__file__, f"{results_dir}/{Path(__file__).name}")
        shutil.copy("./natadv_utils.py", f"{results_dir}/natadv_utils.py")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dave2 = torch.load("/p/sdbb/natadv/weights/model-DAVE2v3-lr1e4-100epoch-batch64-lossMSE-82Ksamples-INDUSTRIALandHIROCHIandUTAH-135x240-noiseflipblur.pt", map_location=torch.device('cpu'), weights_only=False)
    dave2 = dave2.eval().to(device)
    if topo == "topo1-left":
        picklefile = "/p/sdbb/natadv/input/contextualvalidation-dmpatch-30runs-DCGZBG/results-28-cuton10_31-18_25-38X5MM/results.pickle"
    elif topo == "topo1-right":
        picklefile = "/p/sdbb/natadv/input/topo1-right-run/results.pickle"
    with open(picklefile, "rb") as f:
        inputs = pickle.load(f)
        print(inputs.keys(), flush=True)
        sequences = inputs["sequences"]
        outcomes = inputs["testruns_outcomes"]
        for s,o in zip(sequences, outcomes):
            if "D" in o or "LT" in o:
                sequence = s
                break
        unperturbed_seq = inputs["unperturbed_seq"]
        img_arr_pil = [i["image"] for i in sequence]
        # print(f"{img_arr_pil[0].size=}")
        # print(f"{np.array(img_arr_pil).shape=}")
        corners = [i["bbox"][0] for i in sequence]
        split_idx = None
        if args.useunpertseq:
            t = ToPILImage()
            split_idx = len(img_arr_pil) -1
            unpert_arr_pil = [t(i["image"]) for i in unperturbed_seq]
            corners_unpert = [i["bbox"][0] for i in unperturbed_seq]
            img_arr_pil.extend(unpert_arr_pil)
            corners.extend(corners_unpert)
            # print(f"{img_arr_pil[-1].size=}")
            # print(f"{np.array(img_arr_pil).shape=}")
    if args.weighted:
        step = 1 / (len(corners) - 1)
        weights = [step for i in range(len(img_arr_pil))]
    else:
        weights = [1 for i in range(len(img_arr_pil))]
    weights = torch.Tensor(weights).to(device)
    if args.diffusionimage == "CLEAR":
        if topo == "topo1-left":
            masks_parent_dir = "/p/sdbb/natadv/input/sdbb-0001/masks/"
            images_parent_dir = "/p/sdbb/natadv/input/sdbb-0001/images/"
            paths_masks = [i for i in os.listdir(masks_parent_dir) if "sample-annot-0" in i]
            paths_images = [i for i in os.listdir(images_parent_dir) if "sample-0" in i]
        elif topo == "topo1-right":
            masks_parent_dir = "/p/sdbb/natadv/input/topo1-right/masks/"
            images_parent_dir = "/p/sdbb/natadv/input/topo1-right/images/"
            paths_masks = [i for i in os.listdir(masks_parent_dir) if "colorseg-0" in i]
            paths_images = [i for i in os.listdir(images_parent_dir) if "sample-0" in i]

        paths_masks = sorted(paths_masks, key=lambda x: int(x.split("-")[-1][:-4])) #[:30]
        paths_images = sorted(paths_images, key=lambda x: int(x.split("-")[-1][:-4]))
        if len(paths_masks) > len(paths_images):
            paths_masks = sorted(paths_masks, key=lambda x: int(x.split("-")[-1][:-4]))[:len(paths_images)]
        else:
            paths_images = sorted(paths_images, key=lambda x: int(x.split("-")[-1][:-4]))[:len(paths_masks)]
        
        blank_img_arr_pil = [Image.open(images_parent_dir+i) for i in paths_images]
        blank_mask_arr_pil = [Image.open(masks_parent_dir+i) for i in paths_masks]
        idx = len(blank_mask_arr_pil) // 4 #// 4 * 3 -2 #// 4 quarter --> 10x10 pixel billboard #// 2 # (half --> 15x15 pixel billboard) #// 4 * 3 -2 (threequarters --> 20x20 pixel billboard)
        sd_pert, PERT_SIZE = get_clear_billboard(blank_img_arr_pil[idx], blank_mask_arr_pil[idx])
        sd_pert.save(f"./clear_billboard_{PERT_SIZE[0]}x{PERT_SIZE[0]}.jpg")
        exit(0)

    elif args.diffusionimage == "MONDRIAN":
        sd_pert = Image.open("/p/sdbb/natadv/pert_billboard-0024/output-runwayml-stable-diffusion-v1-5/test-prompt-04-mondrian-museumbrochure.jpg")
    elif args.diffusionimage == "ROUSSEAU":
        sd_pert = Image.open("/p/sdbb/natadv/pert_billboard-0024/output-runwayml-stable-diffusion-v1-5-disabled-xformers/test-prompt-03-henrirousseau-addictive.jpg")
    elif args.diffusionimage == "GTA":
        sd_pert = Image.open("/p/sdbb/natadv/pert_billboard-0024/output-runwayml-stable-diffusion-v1-5-disabled-xformers/test-prompt-01-videogameadvertisement-detailed-grandtheftauto.jpg")
    elif args.diffusionimage == "RACING":
        sd_pert = Image.open("/p/sdbb/natadv/pert_billboard-0024/output-runwayml-stable-diffusion-v1-5/test-prompt-01-videogame-advertisement-detailed-exciting-racing.jpg")
    elif args.diffusionimage == "MUSEUM": 
        sd_pert = Image.open("/p/sdbb/natadv/pert_billboard-0024/output-runwayml-stable-diffusion-v1-5-disabled-xformers/test-prompt-02-museum-detailed-colorful-artistic-modernart.jpg")
    elif args.diffusionimage == "PAIR": 
        p = "/p/sdbb/natadv/highstrength-experiment1/high-strength-billboard-library/test-output/"
        name = Path(args.origpertfile).parts[-1]
        print(f"{args.origpertfile=}\n{name=}")
        print(f"{os.listdir(p + name)=}")
        diffusionoutput = os.listdir(p + name)[num_pert]
        sd_pert = Image.open(f"{p}/{name}/{diffusionoutput}")
    original_pert = Image.open(args.origpertfile)


    if not PERT_SIZE:
        PERT_SIZE = sd_pert.size
    else:
        sd_pert = sd_pert.resize(PERT_SIZE, resample=Image.BICUBIC)
    original_pert = original_pert.resize(PERT_SIZE, resample=Image.BILINEAR)

    if args.blur:
        sd_pert = sd_pert.resize(PERT_SIZE, resample=Image.BICUBIC)
        original_pert = original_pert.resize(PERT_SIZE, resample=Image.NEAREST)
        original_pert = original_pert.filter(ImageFilter.GaussianBlur(radius = 3))
    
    transform = Compose([ToTensor()])
    torch.autograd.set_detect_anomaly(True)

    # print(f"{np.array(img_arr_pil).shape=}")
    img_arr = [np.array(i) for i in img_arr_pil]
    img_arr = np.transpose(np.array(img_arr), (0, 3, 1, 2))
    img_arr = np.concatenate((img_arr, copy.deepcopy(img_arr)))
    img_patches, all_src_coords, all_patch_coords = [], [], []
    print(f"Before mask enumerate, corners shape={np.array(corners).shape=}", flush=True)
    for i, mask in enumerate(corners):
        img_patches.append(mask)
        src_coords, patch_coords, image_pert = add_perturbed_billboard(img_arr[i], np.array(sd_pert), np.array(mask), debug=False, results_dir=results_dir)
        all_src_coords.append(src_coords)
        all_patch_coords.append(patch_coords)

    all_src_coords = np.array([np.array(i) for i in all_src_coords])
    all_src_coords = np.concatenate((all_src_coords, copy.deepcopy(all_src_coords)))

    all_src_coords = torch.tensor(np.squeeze(all_src_coords)).to(device)
    all_patch_coords = np.array([np.array(i) for i in all_patch_coords])
    all_patch_coords = np.concatenate((all_patch_coords, copy.deepcopy(all_patch_coords)))

    all_patch_coords = torch.tensor(np.squeeze(all_patch_coords)).to(device)
    perspective_transforms = kornia.geometry.transform.get_perspective_transform(all_src_coords, all_patch_coords) #.to(device)

    inverse_perspective_transforms = kornia.geometry.transform.get_perspective_transform(all_patch_coords, all_src_coords) #.to(device)
    dataset = DataSequence(img_arr, transform=transform)
    data_loader = data.DataLoader(dataset, batch_size=len(dataset))
    img_shape = img_arr.shape[2:]
    pert_shape = np.transpose(np.array(sd_pert), (2, 0, 1))[None].shape
    mask_patch = torch.ones(len(perspective_transforms), *pert_shape).squeeze().float().to(device)
    if USE_DM_PERT:
        perturbation = (transform(original_pert)[None]).float().to(device)
    else:
        perturbation = (transform(sd_pert)[None]).float().to(device)
    perturbation = torch.clamp(perturbation, min=0., max=1.)
    perturbation_sd_start = (transform(sd_pert)[None]).float().to(device)
    perturbation_orig = (transform(original_pert)[None]).float().to(device) #torch.clone(perturbation)
    perturbation.requires_grad_()

    pert_shape = (perturbation.shape[2], perturbation.shape[3])
    perturbation_warp_orig = kornia.geometry.transform.warp_perspective(
        torch.vstack([perturbation_orig for _ in range(len(perspective_transforms))]),
        perspective_transforms,
        dsize=img_shape,
        mode='nearest',
        align_corners=True
    )

    perturbation_warp_sd_start = kornia.geometry.transform.warp_perspective(
        torch.vstack([perturbation_sd_start for _ in range(len(perspective_transforms))]),
        perspective_transforms,
        dsize=img_shape,
        mode='nearest',
        align_corners=True
    )

    warp_masks = kornia.geometry.transform.warp_perspective(
        mask_patch, 
        perspective_transforms, 
        dsize=img_shape,
        mode="nearest",
        align_corners=True
    )
    imgs = next(iter(data_loader)).to(device)
    imgs = torch.permute(imgs, (0, 2, 3, 1))

    imgs_pert_sd_start = imgs * (1 - warp_masks)
    imgs_pert_sd_start += perturbation_warp_sd_start * warp_masks
    imgs_pert_sd_start = torch.clamp(imgs_pert_sd_start, min=0., max=1.)
    preds_pert_sd_start = dave2(imgs_pert_sd_start).cpu().detach().numpy()

    imgs_pert_orig = imgs * (1 - warp_masks)
    imgs_pert_orig += perturbation_warp_orig * warp_masks
    imgs_pert_orig = torch.clamp(imgs_pert_orig, min=0., max=1.)
    preds_pert_orig = dave2(imgs_pert_orig).cpu().detach().numpy()

    grid = torchvision.utils.make_grid(imgs_pert_orig)
    torchvision.utils.save_image(grid, f"./{results_dir}/imgs_pert_orig.jpg")
    ssim_loss = SSIM(window_size = 3)
    initial_ssim_value = ssim_loss(imgs_pert_orig, imgs_pert_sd_start)
    print(f"INITIAL SSIM: {initial_ssim_value:.3f}")
    print(f"INITIAL PRED LOSS: {np.mean(preds_pert_orig - preds_pert_sd_start):.3f}")
    optimizer = optim.Adam([perturbation], lr=1e-2)
    # optimizer = optim.SGD([perturbation], lr=1e-2, momentum=0.9)
    torchvision.utils.save_image(perturbation, f"./{results_dir}/perturbation_start.jpg")
    torchvision.utils.save_image(perturbation_orig, f"./{results_dir}/perturbation_orig_start.jpg")
    ssim_value = initial_ssim_value
    mse_loss = nn.MSELoss()
    all_perturbations = []

    for it in range(args.iters):
        optimizer.zero_grad()
        ssim_loss.zero_grad()
        dave2.zero_grad()
        perturbation = perturbation.detach()
        perturbation.requires_grad_()

        imgs = next(iter(data_loader)).to(device)
        imgs = torch.permute(imgs, (0, 2, 3, 1))

        if args.addnoise:
            imgs = torch.clamp(imgs + torch.rand(imgs.shape).to(device) / 20, 0, 1)

        perturbation_warp_sd = kornia.geometry.transform.warp_perspective(
            torch.vstack([perturbation for _ in range(len(perspective_transforms))]),
            perspective_transforms,
            dsize=img_shape,
            mode="nearest",
            align_corners=True
        )

        imgs_pert_sd = imgs * (1 - warp_masks)
        imgs_pert_sd += perturbation_warp_sd * warp_masks
        imgs_pert_sd = torch.clamp(imgs_pert_sd, min=0., max=1.)
        sim = ssim_loss(imgs_pert_sd, imgs_pert_sd_start)
        sim = ssim_loss(perturbation, perturbation_sd_start)

        mean_sim = sim / ssimdiv #(SSIM_MEAN - sim) / SSIM_TO_PRED_VAR_FACTOR
        # mean_sim = torch.nn.functional.normalize(sim, p=2.0, dim=0, eps=1e-12)

        # weighted_sim = args.ssimweight * (mean_sim)
        weighted_sim = args.ssimweight * sim / ssimdiv
        torchvision.utils.save_image(imgs_pert_sd, f"./{results_dir}/imgs_pert_sd.jpg")
        split = imgs_pert_orig.shape[0] // 2
        # print(dave2(imgs_pert_sd[split:]).shape, flush=True)
        # print(torch.ones(imgs_pert_orig.shape[0]).shape, flush=True)
        # print(torch.unsqueeze(torch.ones(imgs_pert_orig.shape[0]-split), 1).shape, flush=True)
        target_predictions = torch.cat((dave2(imgs_pert_sd[split:]), -0.3 * torch.unsqueeze(torch.ones(imgs_pert_orig.shape[0]-split), 1).to(device)))
        pred_diff = dave2(imgs_pert_orig) - target_predictions
        pred = torch.mean(weights * torch.square(pred_diff))
        weighted_pred = args.predweight * pred
        ssim_out = -weighted_sim - weighted_pred
        ssim_out.backward()
        optimizer.step()
        pert_adjustment = torch.sign(perturbation.grad)
        perturbation = torch.clamp(perturbation + pert_adjustment / 100, 0., 1.)
        # ssim_value = ssim(perturbation_sd_start, perturbation)
        ssim_value = ssim(imgs_pert_sd, imgs_pert_sd_start)
        print(f"ITER {it:03d} loss={ssim_out.item():.4f} normed {mean_sim=:.3f} plain {ssim_value=:.3f} {pred.item()=:.3f} pred_diff={torch.mean(pred_diff):.3f}", flush=True)
        all_perturbations.append(perturbation)

    t_resize = Resize((100, 100)) #, interpolation=InterpolationMode.BILINEAR, max_size=None, antialias=True)
    grid = torchvision.utils.make_grid(torch.cat([t_resize(perturbation_sd_start), t_resize(perturbation), t_resize(perturbation_orig)]))
    torchvision.utils.save_image(grid, f"./{results_dir}/pert_change_after_{it:03d}iters-1RESIZED100.jpg")
    grid = torchvision.utils.make_grid(torch.cat([perturbation_sd_start, perturbation, perturbation_orig]))
    torchvision.utils.save_image(grid, f"./{results_dir}/pert_change_after_{it:03d}iters-1DIM{pert_shape[0]}.jpg")
    grid = torchvision.utils.make_grid(torch.cat(all_perturbations))
    torchvision.utils.save_image(grid, f"./{results_dir}/all_perturbations.jpg")
    grid = torchvision.utils.make_grid(perturbation)
    torchvision.utils.save_image(grid, f"./{results_dir}/perturbation_new.jpg")

    dataset1 = DataSequence(imgs.detach().cpu())
    dataset2 = DataSequence(imgs_pert_orig.detach().cpu())
    dataset3 = DataSequence(imgs_pert_sd.detach().cpu())
    dl1 = data.DataLoader(dataset1, batch_size=1)
    dl2 = data.DataLoader(dataset2, batch_size=1)
    dl3 = data.DataLoader(dataset3, batch_size=1)
    dl1_iter = iter(dl1)
    dl2_iter = iter(dl2)
    dl3_iter = iter(dl3)
    imgs_labelled_orig, imgs_labelled_pert_orig, imgs_labelled_pert_sd = [], [], []
    preds_orig, preds_pert_orig, preds_pert_sd = [], [], []
    dave2 = dave2.to(torch.device('cpu'))
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

    # final_ssim_value = torch.mean(ssim(perturbation_sd_start, perturbation))
    final_ssim_value = ssim(imgs_pert_sd, imgs_pert_sd_start)
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
            "iterations": args.iters,
            "initial_ssim_value": initial_ssim_value,
            "final_ssim_value": final_ssim_value,
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
    print(f"INITIAL PATCH SSIMs (dm vs. sd):  {initial_ssim_value:.2f}")
    print(f"FINAL PATCH SSIMs (dm vs. sd):  {final_ssim_value:.2f}")
    print(f"Results saved to {results_dir}")


if __name__ == '__main__':
    args = parse_args()
    print(f"{args=}")
    p = "/p/sdbb/natadv/highstrength-experiment1/high-strength-billboard-library/test-output/"
    name = Path(args.origpertfile).parts[-1]
    # num_perts = len(os.listdir(p + name))
    # for num_pert in range(num_perts):
    main(args, 0)