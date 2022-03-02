import argparse
import os
from importlib import import_module

import pandas as pd
import torch
from torch.utils.data import DataLoader

from dataset import TestDataset, MaskBaseDataset
from datetime import datetime

from sklearn.ensemble import VotingClassifier
import numpy as np


def load_model(saved_model, num_classes, device, model):
    model_cls = getattr(import_module("model"), model)
    model = model_cls(
        num_classes=num_classes
    )

    # tarpath = os.path.join(saved_model, 'best.tar.gz')
    # tar = tarfile.open(tarpath, 'r:gz')
    # tar.extractall(path=saved_model)

    model_path = os.path.join(saved_model, 'best.pth')
    model.load_state_dict(torch.load(model_path, map_location=device))

    return model


@torch.no_grad()
def inference(data_dir, model_dir, output_dir, args):
    """
    """
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    if args.task == 'multiclass':
        num_classes = 18
    elif args.task == 'mask' or args.task == 'age':
        num_classes = 3
    else:
        num_classes = 2

    img_root = os.path.join(data_dir, 'images')
    info_path = os.path.join(data_dir, 'info.csv')
    info = pd.read_csv(info_path)

    img_paths = [os.path.join(img_root, img_id) for img_id in info.ImageID]
    dataset = TestDataset(img_paths, args.resize)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=8,
        shuffle=False,
        pin_memory=use_cuda,
        drop_last=False,
    )

    if len(args.ensemble) > 1:
        models = list()
        for i, model_dir in enumerate(args.ensemble):
            if "b0" in model_dir:
                model = load_model(model_dir, num_classes, device, "EfficientNetB0").to(device)
            if "b1" in model_dir:
                model = load_model(model_dir, num_classes, device, "EfficientNetB1").to(device)
            if "b2" in model_dir:
                model = load_model(model_dir, num_classes, device, "EfficientNetB2").to(device)
            if "b3" in model_dir:
                model = load_model(model_dir, num_classes, device, "EfficientNetB3").to(device)
            if "b4" in model_dir:
                model = load_model(model_dir, num_classes, device, "EfficientNetB4").to(device)
            model.eval()
            models.append((f'm{i+1}', model))

        preds = []
        with torch.no_grad():
            for idx, images in enumerate(loader):
                images = images.to(device)
                for i, model in enumerate(models):
                    if i == 0:
                        probs = model[1](images)
                    else:
                        probs = probs + model[1](images)
                pred = probs.argmax(dim=-1)
                preds.extend(pred.cpu().numpy())

    
    else:
        model = load_model(model_dir, num_classes, device, args.model).to(device)
        model.eval()

        print("Calculating inference results..")
        preds = []
        with torch.no_grad():
            for idx, images in enumerate(loader):
                images = images.to(device)
                pred = model(images)
                pred = pred.argmax(dim=-1)
                preds.extend(pred.cpu().numpy())

    info['ans'] = preds
    dt_now = datetime.now().strftime('%y%m%d_%H:%M')
    info.to_csv(os.path.join(output_dir, f'{dt_now}_output.csv'), index=False)
    print(f'Inference Done!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument('--task', type=str, default='multiclass', help='choose 1 task in (multiclass, mask, gender, age)')
    parser.add_argument('--batch_size', type=int, default=32, help='input batch size for validing (default: 1000)')
    parser.add_argument('--resize', type=tuple, default=(380, 380), help='resize size for image when you trained (default: (96, 128))')
    parser.add_argument('--model', type=str, default='BaseModel', help='model type (default: BaseModel)')
    parser.add_argument('--ensemble', nargs="+", type=str, default='', help='give model directory')

    # Container environment
    parser.add_argument('--data_dir', type=str, default=os.environ.get('SM_CHANNEL_EVAL', '/opt/ml/input/data/eval'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_CHANNEL_MODEL', './model'))
    parser.add_argument('--output_dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR', './output'))

    args = parser.parse_args()

    data_dir = args.data_dir
    model_dir = args.model_dir
    output_dir = args.output_dir

    os.makedirs(output_dir, exist_ok=True)

    inference(data_dir, model_dir, output_dir, args)
