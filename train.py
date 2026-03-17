
"""author 
   baiyu
"""

import argparse
import glob
import os

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

#from PIL import Image
import transforms
from torchvision import transforms as tv_transforms
from tensorboardX import SummaryWriter
from conf import settings
from utils import *
from lr_scheduler import WarmUpLR
from criterion import LSR

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, required=True, help='net type')
    parser.add_argument('-dataset', type=str, default='cub200', choices=['cub200', 'cifar100'], help='dataset to use')
    parser.add_argument('-w', type=int, default=2, help='number of workers for dataloader')
    parser.add_argument('-b', type=int, default=256, help='batch size for dataloader')
    parser.add_argument('-lr', type=float, default=0.04, help='initial learning rate')
    parser.add_argument('-e', type=int, default=450, help='training epoches')
    parser.add_argument('-warm', type=int, default=5, help='warm up phase')
    parser.add_argument('-gpus', nargs='+', type=int, default=0, help='gpu device')
    args = parser.parse_args()

    #checkpoint directory
    checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.net, settings.TIME_NOW)
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    checkpoint_path = os.path.join(checkpoint_path, '{net}-{epoch}-{type}.pth')

    #tensorboard log directory
    log_path = os.path.join(settings.LOG_DIR, args.net, settings.TIME_NOW)
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    writer = SummaryWriter(log_dir=log_path)

    #output log file
    output_path = 'output'
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    log_file = open(os.path.join(output_path, '{}-{}-{}.log'.format(args.net, args.dataset, settings.TIME_NOW)), 'w')

    def log(msg):
        print(msg)
        log_file.write(msg + '\n')
        log_file.flush()

    #get dataloader
    if args.dataset == 'cifar100':
        train_mean = settings.CIFAR100_TRAIN_MEAN
        train_std = settings.CIFAR100_TRAIN_STD
        test_mean = settings.CIFAR100_TEST_MEAN
        test_std = settings.CIFAR100_TEST_STD
        num_classes = 100
    else:
        train_mean = settings.TRAIN_MEAN
        train_std = settings.TRAIN_STD
        test_mean = settings.TEST_MEAN
        test_std = settings.TEST_STD
        num_classes = 200

    if args.dataset == 'cifar100':
        train_transforms = tv_transforms.Compose([
            tv_transforms.RandomCrop(32, padding=4),
            tv_transforms.RandomHorizontalFlip(),
            tv_transforms.AutoAugment(tv_transforms.AutoAugmentPolicy.CIFAR10),
            tv_transforms.ToTensor(),
            tv_transforms.Normalize(train_mean, train_std),
            tv_transforms.RandomErasing(p=0.25),
        ])
        test_transforms = tv_transforms.Compose([
            tv_transforms.ToTensor(),
            tv_transforms.Normalize(test_mean, test_std),
        ])
    else:
        train_transforms = transforms.Compose([
            transforms.ToCVImage(),
            transforms.RandomResizedCrop(settings.IMAGE_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.4, saturation=0.4, hue=0.4),
            transforms.ToTensor(),
            transforms.Normalize(train_mean, train_std)
        ])
        test_transforms = transforms.Compose([
            transforms.ToCVImage(),
            transforms.CenterCrop(settings.IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(test_mean, test_std)
        ])

    if args.dataset == 'cifar100':
        train_dataloader = get_cifar100_train_dataloader(
            settings.CIFAR100_DATA_PATH,
            train_transforms,
            args.b,
            args.w
        )
        test_dataloader = get_cifar100_test_dataloader(
            settings.CIFAR100_DATA_PATH,
            test_transforms,
            args.b,
            args.w
        )
    else:
        train_dataloader = get_train_dataloader(
            settings.DATA_PATH,
            train_transforms,
            args.b,
            args.w
        )
        test_dataloader = get_test_dataloader(
            settings.DATA_PATH,
            test_transforms,
            args.b,
            args.w
        )

    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    net = get_network(args, num_classes=num_classes)
    net = init_weights(net)
    net = net.to(device)

    #visualize the network
    visualize_network(writer, net)

    #cross_entropy = nn.CrossEntropyLoss() 
    lsr_loss = LSR()

    #apply no weight decay on bias
    params = split_weights(net)
    optimizer = optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=1e-4, nesterov=True)

    #set up warmup phase learning rate scheduler
    iter_per_epoch = len(train_dataloader)
    warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * args.warm)

    #set up training phase learning rate scheduler
    train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=settings.MILESTONES)
    #train_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.e - args.warm)

    best_acc = 0.0
    for epoch in range(1, args.e + 1):
        if epoch > args.warm:
            train_scheduler.step(epoch)

        #training procedure
        net.train()
        train_loss = 0.0
        for batch_index, (images, labels) in enumerate(train_dataloader):
            if epoch <= args.warm:
                warmup_scheduler.step()

            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            predicts = net(images)
            loss = lsr_loss(predicts, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            n_iter = (epoch - 1) * len(train_dataloader) + batch_index + 1

            #visualization
            visualize_lastlayer(writer, net, n_iter)
            visualize_train_loss(writer, loss.item(), n_iter)

        visualize_learning_rate(writer, optimizer.param_groups[0]['lr'], epoch)
        visualize_param_hist(writer, net, epoch)

        avg_train_loss = train_loss / len(train_dataloader)
        msg = 'Epoch [{}/{}]  Train Loss: {:.4f}  LR: {:.6f}'.format(
            epoch, args.e, avg_train_loss, optimizer.param_groups[0]['lr'])
        print(msg)
        log_file.write(msg + '\n')
        log_file.flush()

        net.eval()

        total_loss = 0
        correct = 0
        for images, labels in test_dataloader:

            images = images.to(device)
            labels = labels.to(device)

            predicts = net(images)
            _, preds = predicts.max(1)
            correct += preds.eq(labels).sum().float()

            loss = lsr_loss(predicts, labels)
            total_loss += loss.item()

        test_loss = total_loss / len(test_dataloader)
        acc = correct / len(test_dataloader.dataset)
        msg = 'Test set: loss: {:.4f}, Accuracy: {:.4f}'.format(test_loss, acc)
        print(msg)
        print()
        log_file.write(msg + '\n\n')
        log_file.flush()

        visualize_test_loss(writer, test_loss, epoch)
        visualize_test_acc(writer, acc, epoch)

        #save weights file
        if epoch > settings.MILESTONES[1] and best_acc < acc:
            torch.save(net.state_dict(), checkpoint_path.format(net=args.net, epoch=epoch, type='best'))
            best_acc = acc
            continue
        
        if not epoch % settings.SAVE_EPOCH:
            torch.save(net.state_dict(), checkpoint_path.format(net=args.net, epoch=epoch, type='regular'))
    
    writer.close()
    log_file.close()










    


    

