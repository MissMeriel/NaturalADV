import numpy as np
import argparse
import pandas as pd
import cv2
import matplotlib.image as mpimg
import json
from torch.autograd import Variable

import os
from PIL import Image
import PIL
import matplotlib.pyplot as plt

from DAVE2pytorch import DAVE2PytorchModel, DAVE2v1, DAVE2v2, DAVE2v3, Epoch
from UUSTDatasetGenerator import MultiDirectoryDataSequence

import time
import random, string, shutil

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
from torchvision.transforms import Compose, ToPILImage, ToTensor, Resize, Lambda, Normalize

def parse_args():
    parser = argparse.ArgumentParser(prog='ProgramName', description='What the program does',
                                     epilog='Text at the bottom of help')
    # parser.add_argument("-p", '--path_to_trainingdir', type=str, default='/p/sdbb/BeamNG_DAVE2_racetracks', help='path to training data parentdir')
    parser.add_argument('--dataset', help='parent directory of base model dataset', default=None) #default="H:/BeamNG_DeepBillboard_dataset2")
    parser.add_argument('--RRL_dir', help='parent directory of RRL dataset', default=None) #default="H:/RRL-results/RLtrain-max200epi-DDPGhuman-0.05evaleps-bigimg-1_19-20_43-IW6ZTT/RLtrain-max200epi-DDPGhuman-0.05evaleps-bigimg-1_19-20_43-IW6ZTT")
    parser.add_argument('--effect', help='image transformation', default=None)

    parser.add_argument('-n', '--noisevar', type=float, default=20, help='max noisevar to sample')
    parser.add_argument('-o', '--outdir_id', type=str, default="out", help='identifier or slurm job id')
    parser.add_argument("-d", "--img_dims", type=str, default="135 240", help="Image dimensions (H W)")
    parser.add_argument('-r', '--lr', type=float, default=1e-5, help='learning rate')
    parser.add_argument('-e', '--epochs', type=int, default=1000, help='training epochs')
    parser.add_argument('-m', '--pretrained_model', type=str, default=None, help='path to pretrained model')
    parser.add_argument('-s', '--start_epochs', type=int, default=0, help='pretrained model epochs')
    parser.add_argument('-b', '--batch', type=int, default=64, help='batch size')
    args = parser.parse_args()
    print(f"cmd line args:{args}")    
    return args

args = parse_args()

def save_metadata(newdir, iteration, epoch, model_name, dataset, args, optimizer, running_loss, logfreq, device, robustification, noise_level, time_to_train):
    # with open(f'./{newdir}/model-{iteration}-epoch{epoch}-metainfo.txt', "w") as f:
    filename = model_name.replace('.pt', '-metainfo.txt')
    with open(f'./{filename}', "w") as f:
        f.write(f"{model_name=}\n"
                f"total_samples={dataset.get_total_samples()}\n"
                f"{args.epochs=}\n"
                f"{epoch=}\n"
                f"Warm start {args.pretrained_model=}"
                f"{args.lr=}\n"
                f"{args.batch=}\n"
                f"{optimizer=}\n"
                f"final_loss={running_loss / logfreq}\n"
                f"{device=}\n"
                f"{robustification=}\n"
                f"{noise_level=}\n"
                f"dataset_moments={dataset.get_outputs_distribution()}\n"
                f"{time_to_train=}\n"
                f"dirs={dataset.get_directories()}")

def main_pytorch_model():
    start_time = time.time()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_shape = tuple([int(d) for d in args.img_dims.split(" ")]) #(81, 144) # (108, 192) # (135, 240)
    print(f"{input_shape=}")
    print(f"{device=}", flush=True)
    model = DAVE2v3(input_shape=input_shape)
    if args.pretrained_model is not None:
        model = model.load(args.pretrained_model, map_location=device)
    NB_EPOCH = args.epochs - args.start_epochs
    robustification = True
    noise_level = 15
    print(args, flush=True)
    dataset = MultiDirectoryDataSequence(args.dataset, args.RRL_dir, image_size=(model.input_shape[::-1]), transform=Compose([ToTensor()]),\
                                        robustification=robustification, noise_level=noise_level, sample_id="STEERING_INPUT",
                                        effect=args.effect) #, Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]))


    # print("Retrieving output distribution....", flush=True)
    print("Moments of distribution:", dataset.get_outputs_distribution(), flush=True)
    print("Total samples:", dataset.get_total_samples(), flush=True)
    def worker_init_fn(worker_id):
        np.random.seed(np.random.get_state()[1][0] + worker_id)

    trainloader = DataLoader(dataset, batch_size=args.batch, shuffle=True, worker_init_fn=worker_init_fn)
    
    localtime = time.localtime()
    timestr = "{}_{}-{}_{}".format(localtime.tm_mon, localtime.tm_mday, localtime.tm_hour, localtime.tm_min)
    
    randstr = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(6))
    newdir = f"./{model._get_name()}-{args.effect}-{input_shape[0]}x{input_shape[1]}-{int(dataset.get_total_samples()/1000)}samples-{args.epochs}epoch-{args.outdir_id}-{timestr}-{randstr}"
    if not os.path.exists(newdir):
        os.mkdir(newdir,  mode=0o777)
        shutil.copyfile(__file__, f"{newdir}/{__file__.split('/')[-1]}", follow_symlinks=False)
        shutil.copyfile("UUSTDatasetGenerator.py", f"{newdir}/UUSTDatasetGenerator.py", follow_symlinks=False)
    iteration = f'{model._get_name()}-{input_shape[0]}x{input_shape[1]}-{args.epochs}epoch-{args.batch}batch-{int(dataset.get_total_samples()/1000)}Ksamples'
    
    print(f"{iteration=}", flush=True)
    model = model.to(device)
    # if loss doesnt level out after 20 epochs, either inc epochs or inc learning rate
    optimizer = optim.Adam(model.parameters(), lr=args.lr) #, betas=(0.9, 0.999), eps=1e-08)
    lowest_loss = 1e5
    logfreq = 20
    best_model_count = 0
    for epoch in range(NB_EPOCH):
        running_loss = 0.0
        for i, hashmap in enumerate(trainloader, 0):
            x = hashmap['image_base'].float().to(device)
            y = hashmap['steering_input'].float().to(device)
            x = Variable(x, requires_grad=True)
            y = Variable(y, requires_grad=False)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(x)
            loss = F.mse_loss(outputs.flatten(), y)
            # loss = F.mse_loss(outputs, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % logfreq == logfreq-1:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.7f' %
                      (epoch + 1, i + 1, running_loss / logfreq), flush=True)
                if (running_loss / logfreq) < lowest_loss:
                    print(f"New best model! MSE loss: {running_loss / logfreq}", flush=True)
                    model_name = f"./{newdir}/model-{iteration}-epoch{(epoch + args.start_epochs):03d}-best{best_model_count:03d}.pt"
                    print(f"Saving model to {model_name}", flush=True)
                    time_to_train=time.time() - start_time
                    save_metadata(newdir, iteration, epoch, model_name, dataset, args, optimizer, running_loss, logfreq, device, robustification, noise_level, time_to_train)
                    torch.save(model, model_name)
                    best_model_count += 1
                    lowest_loss = running_loss / logfreq
                running_loss = 0.0
        print(f"Finished {epoch=}", flush=True)
        model_name = f"./{newdir}/model-{iteration}-epoch{(epoch + args.start_epochs):03d}.pt"
        print(f"Saving model to {model_name}", flush=True)
        torch.save(model, model_name)
        # if loss < 0.000001:
        #     print(f"Loss at {loss}; quitting training...")
        #     break
    print('Finished Training', flush=True)

    # save model
    torch.save(model.state_dict(), f'./{newdir}/model-{iteration}-weights.pt')
    model_name = f'./{newdir}/model-{iteration}.pt'
    torch.save(model, model_name)

    # delete weights from previous epochs
    # print("Deleting weights from previous epochs...", flush=True)
    # for epoch in range(NB_EPOCH):
    #     os.remove(f"./{newdir}/model-{iteration}-epoch{epoch:03d}.pt")
    print(f"Saving model to {model_name}", flush=True)
    time_to_train=time.time() - start_time
    print("Time to train: {}".format(time_to_train), flush=True)
    # save metainformation about training
    save_metadata(newdir, iteration, model_name, dataset, args, optimizer, running_loss, logfreq, device, robustification, noise_level, time_to_train)
    print(f"{dataset.get_outputs_distribution()=}")


if __name__ == '__main__':
    main_pytorch_model()
