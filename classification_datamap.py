import glob
import json
import os
import random
from argparse import ArgumentParser
from PIL import Image
from tqdm.auto import tqdm

import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, random_split, SequentialSampler

def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./cifar10_dataset')
    parser.add_argument('--model_dir', type=str, default='./models')
    parser.add_argument('--datamap_dir', type=str, default='./datamap')
    parser.add_argument('--tfm_type', type=str, default='auto_augment')
    parser.add_argument('--seed', type=str, default=0)
    parser.add_argument('--valid_ratio', type=float, default=0.1)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--n_epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=int, default=1e-5)
    args, unknown = parser.parse_known_args()
    return args

def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return

def get_transform(tfm_type):
    print(tfm_type)

    if tfm_type == 'none':
        return transforms.Compose([
            transforms.ToTensor(),
        ])

    tfm_dict = {
        'horizontal_flip': transforms.RandomHorizontalFlip(p=0.5),
        'color_jitter': transforms.ColorJitter(brightness=(0, 5), contrast=(0, 5), saturation=(0, 5), hue=(-0.1, 0.1)),
        'auto_augment': transforms.AutoAugment(policy=torchvision.transforms.autoaugment.AutoAugmentPolicy.CIFAR10),
    }
    tfm = tfm_dict[tfm_type]

    return transforms.Compose([
        tfm,
        transforms.ToTensor(),
    ])

class Cifar10Dataset(Dataset):
    def __init__(self, fnames, transform=None):
        super(Cifar10Dataset).__init__()
        self.fnames = fnames
        self.transform = transform

    def __len__(self):
        return len(self.fnames)
    
    def __getitem__(self, idx):
        fname = self.fnames[idx]
        img = Image.open(fname)
        img = self.transform(img)

        label = fname.split('/')[-2]
        
        return img, int(label)

def get_dataloader(data_dir, tfm_type, valid_ratio, batch_size):
    data_dir = os.path.join(data_dir, 'train')
    fnames = glob.glob(f'{data_dir}/**/*.png')

    dataset = Cifar10Dataset(fnames=fnames, 
                             transform=get_transform(tfm_type))

    sampler = SequentialSampler(dataset)

    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=4)

    return train_loader, fnames

def get_model(n_classes=10):
    model = models.resnet152(pretrained=True)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, n_classes)
    return model

def train_one_epoch(model, dataloader, optimizer, criterion, device, epoch, n_epochs):
    model.train()

    train_losses = list()
    train_accs = list()

    # For training dynamics
    train_ins_probs = list()
    train_ins_corrects = list()
    
    for batch in tqdm(dataloader):
        imgs, labels = batch
        logits = model(imgs.to(device))

        loss = criterion(logits, labels.to(device))

        optimizer.zero_grad()
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)
        optimizer.step()

        acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()
        
        train_losses.append(loss.item())
        train_accs.append(acc)

        # Training dynamics
        predict_probs = logits.detach().softmax(dim=-1).cpu()
        predict_labels = predict_probs.argmax(dim=-1)
        gold_labels = labels.cpu()
        gold_probs = predict_probs.gather(
            dim=-1,
            index=gold_labels.unsqueeze(-1)
        ).squeeze(-1)
        correct = predict_labels.eq(gold_labels)

        train_ins_probs.extend(gold_probs.tolist())
        train_ins_corrects.extend(correct.tolist())

    train_loss = sum(train_losses) / len(train_losses)
    train_acc = sum(train_accs) / len(train_accs)
    print(f"[ Train | {epoch + 1:03d}/{n_epochs:03d} ] loss = {train_loss:.5f}, acc = {train_acc:.5f}")

    return train_ins_probs, train_ins_corrects, train_acc

def main(args):
    set_seed(args.seed)

    train_loader, fnames = get_dataloader(data_dir=args.data_dir,
                                        tfm_type=args.tfm_type,
                                        valid_ratio=args.valid_ratio,
                                        batch_size=args.batch_size)
    
    model = get_model(n_classes=10)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0
    start_epoch = 0

    train_ins_prob_list = list()
    train_ins_corr_list = list()
    
    for epoch in range(start_epoch, args.n_epochs):
        train_ins_probs, train_ins_corrects, acc = train_one_epoch(model=model, 
                                                                    dataloader=train_loader, 
                                                                    optimizer=optimizer, 
                                                                    criterion=criterion, 
                                                                    device=device,
                                                                    epoch=epoch,
                                                                    n_epochs=args.n_epochs)
        train_ins_prob_list.append(train_ins_probs)
        train_ins_corr_list.append(train_ins_corrects)
        
        if acc > best_acc:
            best_acc = acc
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'acc': best_acc
            }
            if not os.path.exists(args.model_dir):
                os.mkdir(args.model_dir)
            torch.save(checkpoint, os.path.join(args.model_dir, f'{args.tfm_type}_best.ckpt'))
            print(f'Find best acc at epoch {epoch + 1}')
    
    checkpoint = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': args.n_epochs,
        'acc': best_acc
    }
    torch.save(checkpoint, os.path.join(args.model_dir, f'{args.tfm_type}.ckpt'))

    # Training dynamics
    train_ins_gold_prob_means = np.mean(train_ins_prob_list, axis=0)
    train_ins_gold_prob_std = np.std(train_ins_prob_list, axis=0)
    train_ins_correct_means = np.mean(train_ins_corr_list, axis=0)

    print(len(train_ins_gold_prob_means), len(train_ins_gold_prob_std), len(train_ins_correct_means))

    training_dynamics = {
        'fnames': fnames,
        'gold_prob_means': train_ins_gold_prob_means.tolist(),
        'gold_prob_stds': train_ins_gold_prob_std.tolist(),
        'correct_means': train_ins_correct_means.tolist()
    }
    
    if not os.path.exists(args.datamap_dir):
        os.mkdir(args.datamap_dir)
    with open(os.path.join(args.datamap_dir, f'{args.tfm_type}.json'), 'w') as writer:
        writer.write(json.dumps(training_dynamics, indent=4))

    return

if __name__ == '__main__':
    main(parse_arguments())
