import argparse
import os
import numpy as np
import random

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import resnet
import utils
from pathlib import Path
from sklearn.model_selection import StratifiedShuffleSplit


def get_args_parser():
    parser = argparse.ArgumentParser('Propert ResNets for CIFAR10 in pytorch', add_help=False)
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=200, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch-size', default=128, type=int,
                        metavar='N', help='mini-batch size (default: 128)')
    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('--print-freq', '-p', default=50, type=int,
                        metavar='N', help='print frequency (default: 50)')
    parser.add_argument('--resume', default=None, type=str,
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--validation', action='store_true', default=True)
    parser.add_argument('--save-every', dest='save_every',
                        help='Saves checkpoints at every specified number of epochs',
                        type=int, default=10)
    parser.add_argument('--output_dir', default='logs', help='path where to save logs')
    parser.add_argument('--seed', default=111, type=int)
    parser.add_argument('--device', default='cuda', type=str,
                        help='device to use for training / testing')
    parser.add_argument('--GPU_ids', type=str, default='0', help='Ids of GPUs')

    parser.add_argument('--cutout', action='store_true', default=False,
                        help='apply cutout')
    parser.add_argument('--n_holes', type=int, default=1,
                        help='number of holes to cut out from image')
    parser.add_argument('--length', type=int, default=16,
                        help='length of the holes')
    parser.add_argument('--mixup', action='store_true', default=False,
                        help='apply mixup')
    parser.add_argument('--alpha', type=float, default=1.0,
                        help='alpha for mixup and cutmix')
    parser.add_argument('--cutmix_prob', type=float, default=.0,
                        help='prob for cutmix')
    return parser


def prepare_training(args):
    model = torch.nn.DataParallel(resnet.resnet32())
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                        milestones=[100, 150], last_epoch=args.start_epoch - 1)

    log('resnet32: #params={}'.format(utils.compute_num_params(model, text=True)))
    log(model)
    log(criterion)
    log('Building dataset...')
    normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                     std=[x / 255.0 for x in [63.0, 62.1, 66.7]])

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    if args.cutout:
        transform_train.transforms.append(utils.Cutout(n_holes=args.n_holes, length=args.length))
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 120, 160],
                                                            last_epoch=args.start_epoch - 1, gamma=0.2)

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
    trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    validset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_test)
    testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

    if args.validation:
        labels = [trainset[i][1] for i in range(len(trainset))]
        ss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=0)
        train_indices, valid_indices = list(ss.split(np.array(labels)[:, np.newaxis], labels))[0]
        trainset = torch.utils.data.Subset(trainset, train_indices)
        validset = torch.utils.data.Subset(validset, valid_indices)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True,
                                              num_workers=args.workers, pin_memory=True)
    validloader = torch.utils.data.DataLoader(validset, batch_size=100, shuffle=False,
                                              pin_memory=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False,
                                             pin_memory=True)

    if args.resume:
        checkpoint = torch.load(args.resume)
        args.start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])

    return model, criterion, optimizer, lr_scheduler, trainloader, validloader, testloader


def train(train_loader, model, criterion, optimizer, epoch, device):
    """
        Run one train epoch
    """
    t = {'losses': utils.AverageMeter(),
         'top1': utils.AverageMeter()}
    for i in range(10):
        t[i] = utils.AverageMeter()
    log('Epoch: [{}]'.format(epoch))

    # switch to train mode
    model.train()
    criterion.train()

    for i, (input, target) in enumerate(train_loader):
        cutmix_flag = False
        target = target.to(device)
        input = input.to(device)

        if args.mixup:
            input, targets_a, targets_b, lam = utils.mixup_data(input, target, args.alpha, device)
        if args.cutmix_prob > 0:
            r = np.random.rand(1)
            if r < args.cutmix_prob:
                cutmix_flag = True
                input, targets_a, targets_b, lam = utils.cutmix_data(input, target, args.alpha, device)
        output = model(input)
        if args.mixup:
            loss = utils.mixup_criterion(criterion, output, targets_a, targets_b, lam)
        elif cutmix_flag:
            loss = utils.cutmix_criterion(criterion, output, targets_a, targets_b, lam)
        else:
            loss = criterion(output, target)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        output = output.float()
        loss = loss.float()
        # measure accuracy and record loss
        t['losses'].update(loss.item(), input.size(0))
        if args.mixup:
            _, predicted = torch.max(output.data, 1)
            correct = (lam * predicted.eq(targets_a.data).cpu().sum().float()
                       + (1 - lam) * predicted.eq(targets_b.data).cpu().sum().float())
            t['top1'].update(sum=correct, n=input.size(0))
        elif cutmix_flag:
            _, predicted = torch.max(output.data, 1)
            correct = (lam * predicted.eq(targets_a.data).cpu().sum().float()
                       + (1 - lam) * predicted.eq(targets_b.data).cpu().sum().float())
            t['top1'].update(sum=correct, n=input.size(0))
        else:
            prec1, num_map, correct_map = utils.accuracy(output.data, target)
            for j in range(10):
                t[j].update(sum=correct_map[j], n=num_map[j])
            t['top1'].update(sum=prec1[0].item(), n=input.size(0))
        if i % 50 == 0:
            print('step {}/{}:'.format(i, len(train_loader)))
            for k, v in t.items():
                print('{}:{:.4f}'.format(k, v.avg))
    return t


@torch.no_grad()
def validate(val_loader, model, criterion, device):
    """
    Run evaluation
    """
    model.eval()
    criterion.eval()
    t = {'losses': utils.AverageMeter(),
         'top1': utils.AverageMeter()}
    for i in range(10):
        t[i] = utils.AverageMeter()
    total_steps = len(val_loader)
    iterats = iter(val_loader)
    for step in range(total_steps):
        input, target = next(iterats)
        target = target.to(device)
        input_var = input.to(device)
        target_var = target
        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        output = output.float()
        loss = loss.float()

        # measure accuracy and record loss
        prec1, num_map, correct_map = utils.accuracy(output.data, target)
        t['losses'].update(loss.item(), input.size(0))
        t['top1'].update(sum=prec1[0].item(), n=input.size(0))
        for j in range(10):
            t[j].update(sum=correct_map[j], n=num_map[j])
    return t


def main(args):
    global log, writer
    log, writer = utils.set_save_path(args.output_dir)
    log(args)
    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model, criterion, optimizer, lr_scheduler, train_loader, val_loader, test_loader = prepare_training(args)
    model.to(device)
    criterion.to(device)
    cudnn.benchmark = True

    # training
    output_dir = Path(args.output_dir)
    log("Start training")
    timer = utils.Timer()
    for epoch in range(args.start_epoch, args.epochs + 1):
        t_epoch_start = timer.t()
        log_info = ['epoch {}/{}'.format(epoch, args.epochs)]
        writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)
        train_stats = train(train_loader, model, criterion, optimizer, epoch, device)
        lr_scheduler.step()

        log_info.append('train:')
        log_info = log_info + ['{}={:.4f}'.format(k, v.avg) for k, v in train_stats.items()]

        if args.output_dir:
            checkpoint_path = output_dir / 'checkpoint.pth'
            if epoch > 0 and epoch % args.save_every == 0:
                sv_file = {
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'args': args
                }
                torch.save(sv_file, checkpoint_path)

        val_stats = validate(val_loader, model, criterion, device)
        log_info.append('val:')
        log_info = log_info + ['{}={:.4f}'.format(k, v.avg) for k, v in val_stats.items()]

        test_stats = validate(test_loader, model, criterion, device)
        log_info.append('test:')
        log_info = log_info + ['{}={:.4f}'.format(k, v.avg) for k, v in test_stats.items()]

        writer.add_scalars('loss', {'train': train_stats['losses'].avg,
                                    'val': val_stats['losses'].avg,
                                    'test': test_stats['losses'].avg}, epoch)
        writer.add_scalars('Acc', {'train': train_stats['top1'].avg,
                                   'val': val_stats['top1'].avg,
                                   'test': test_stats['top1'].avg}, epoch)
        writer.add_figure('val', utils.generate_fig(val_stats), epoch)
        writer.add_figure('test', utils.generate_fig(test_stats), epoch)

        t = timer.t()
        prog = (epoch - args.start_epoch + 1) / (args.epochs - args.start_epoch + 1)
        t_epoch = utils.time_text(t - t_epoch_start)
        t_elapsed, t_all = utils.time_text(t), utils.time_text(t / prog)
        log_info.append('{} {}/{}'.format(t_epoch, t_elapsed, t_all))

        log(', '.join(log_info))
        writer.flush()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Cifar-10 training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(args.GPU_ids)
    main(args)
