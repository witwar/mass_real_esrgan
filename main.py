#!/usr/bin/env python3

import os
import shutil
import sys
from zipfile import ZipFile

import numpy as np
import torch
from PIL import Image
from RealESRGAN import RealESRGAN

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device:', device)

model_scale = "2"

model = RealESRGAN(device, scale=int(model_scale))
model.load_weights(f'weights/RealESRGAN_x{model_scale}.pth')
print(f"Model Scale {model_scale} Has Been Installed!")

os.makedirs('output', exist_ok=True)

def extract_zip():
    z = ZipFile('input.zip', 'r')
    z.extractall('input')

def create_zip():
    shutil.make_archive('output', 'zip', 'output')

def process_image(ipath, opath):
    image = Image.open(ipath).convert('RGB')
    if not torch.cuda.is_available():
        image.save(opath)
        return False
    sr_image = model.predict(image)
    sr_image.save(opath)

    return True

def main():

    if os.path.exists('input.zip'):
        extract_zip()

    for root, dirs, files in os.walk('input'):
        for fname in files:
            if not fname.lower().endswith(('.jpg', 'jpeg', '.png')):
                continue
            ipath = os.path.join(root, fname)
            opath = os.path.join('output', f'hd_{fname}')
            if not os.path.exists(opath):
                print(ipath, opath)
                process_image(ipath, opath)

    create_zip()

    print('OK')

if __name__ == '__main__':
    main()
