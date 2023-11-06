import copy
import os
import pdb

import numpy as np
from math import ceil
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import Subset
from train import define_model, train
from data import TensorDataset, ImageFolder, MultiEpochsDataLoader
from data import save_img, transform_imagenet, transform_cifar, transform_svhn, transform_mnist, transform_fashion
import models.resnet as RN
from misc.reproduce import set_arguments
import models.densenet_cifar as DN
from coreset import randomselect, herding
from efficientnet_pytorch import EfficientNet
import math
from networks import VGG11, AlexNet



def return_data_path(args):
    if args.factor > 1:
        init = 'mix'
    else:
        init = 'random'
    if args.dataset == 'imagenet' and args.nclass == 100:
        args.slct_type = 'idc_cat'
        args.nclass_sub = 20

    if 'idc' in args.slct_type:
        name = args.name
        if name == '':
            if args.dataset == 'cifar10':
                name = f'cifar10/conv3in_grad_mse_nd2000_cut_niter2000_factor{args.factor}_lr0.005_{init}'

            elif args.dataset == 'cifar100':
                name = f'cifar100/conv3in_grad_mse_nd2000_cut_niter2000_factor{args.factor}_lr0.005_{init}'

            elif args.dataset == 'imagenet':
                if args.nclass == 10:
                    name = f'imagenet10/resnet10apin_grad_l1_ely10_nd500_cut_factor{args.factor}_{init}'
                elif args.nclass == 100:
                    name = f'imagenet100/resnet10apin_grad_l1_pt5_nd500_cut_nlr0.1_wd0.0001_factor{args.factor}_lr0.001_b_real128_{init}'

            elif args.dataset == 'svhn':
                name = f'svhn/conv3in_grad_mse_nd500_cut_niter2000_factor{args.factor}_lr0.005_{init}'
                if args.factor == 1 and args.ipc == 1:
                    args.mixup = 'vanilla'
                    args.dsa_strategy = 'color_crop_cutout_scale_rotate'

            elif args.dataset == 'mnist':
                if args.factor == 1:
                    name = f'mnist/conv3in_grad_l1_nd500_cut_niter2000_factor{args.factor}_lr0.0001_{init}'
                else:
                    name = f'mnist/conv3in_grad_l1_nd500_niter2000_factor{args.factor}_color_crop_lr0.0001_{init}'
                    args.mixup = 'vanilla'
                    args.dsa_strategy = 'color_crop_scale_rotate'

            elif args.dataset == 'fashion':
                name = f'fashion/conv3in_grad_l1_nd500_cut_niter2000_factor{args.factor}_lr0.0001_{init}'

        path_list = [f'{name}_ipc{args.ipc}']

    elif args.slct_type == 'dsa':
        path_list = [f'cifar10/dsa/res_DSA_CIFAR10_ConvNet_{args.ipc}ipc']
    elif args.slct_type == 'kip':
        path_list = [f'cifar10/kip/kip_ipc{args.ipc}']
    else:
        path_list = ['']

    return path_list


def resnet10_in(args, nclass, logger=None):
    model = RN.ResNet(args.dataset, 10, nclass, 'instance', args.size, nch=args.nch)
    if logger is not None:
        logger(f"=> creating model resnet-10, norm: instance")
    return model


def resnet10_bn(args, nclass, logger=None):
    model = RN.ResNet(args.dataset, 10, nclass, 'batch', args.size, nch=args.nch)
    if logger is not None:
        logger(f"=> creating model resnet-10, norm: batch")
    return model


def resnet18_bn(args, nclass, logger=None):
    model = RN.ResNet(args.dataset, 18, nclass, 'batch', args.size, nch=args.nch)
    if logger is not None:
        logger(f"=> creating model resnet-18, norm: batch")
    return model


def densenet(args, nclass, logger=None):
    if 'cifar' == args.dataset[:5]:
        model = DN.densenet_cifar(nclass)
    else:
        raise AssertionError("Not implemented!")

    if logger is not None:
        logger(f"=> creating DenseNet")
    return model


def efficientnet(args, nclass, logger=None):
    if args.dataset == 'imagenet':
        model = EfficientNet.from_name('efficientnet-b0', num_classes=nclass)
    else:
        raise AssertionError("Not implemented!")

    if logger is not None:
        logger(f"=> creating EfficientNet")
    return model


def load_ckpt(model, file_dir, verbose=True):
    checkpoint = torch.load(file_dir)
    if 'state_dict' in checkpoint:
        checkpoint = checkpoint['state_dict']
    checkpoint = remove_prefix_checkpoint(checkpoint, 'module')
    model.load_state_dict(checkpoint)

    if verbose:
        print(f"\n=> loaded checkpoint '{file_dir}'")


def remove_prefix_checkpoint(dictionary, prefix):
    keys = sorted(dictionary.keys())
    for key in keys:
        if key.startswith(prefix):
            newkey = key[len(prefix) + 1:]
            dictionary[newkey] = dictionary.pop(key)
    return dictionary


def decode_zoom(img, target, factor, size=-1):
    if size == -1:
        size = img.shape[-1]
    resize = nn.Upsample(size=size, mode='bilinear')

    h = img.shape[-1]
    remained = h % factor
    if remained > 0:
        img = F.pad(img, pad=(0, factor - remained, 0, factor - remained), value=0.5)
    s_crop = ceil(h / factor)
    n_crop = factor**2

    cropped = []
    for i in range(factor):
        for j in range(factor):
            h_loc = i * s_crop
            w_loc = j * s_crop
            cropped.append(img[:, :, h_loc:h_loc + s_crop, w_loc:w_loc + s_crop])
    cropped = torch.cat(cropped)
    data_dec = resize(cropped)
    target_dec = torch.cat([target for _ in range(n_crop)])

    return data_dec, target_dec


def decode_zoom_multi(img, target, factor_max):
    data_multi = []
    target_multi = []
    for factor in range(1, factor_max + 1):
        decoded = decode_zoom(img, target, factor)
        data_multi.append(decoded[0])
        target_multi.append(decoded[1])

    return torch.cat(data_multi), torch.cat(target_multi)


def decode_fn(data, target, factor, decode_type, bound=128):
    if factor > 1:
        if decode_type == 'multi':
            data, target = decode_zoom_multi(data, target, factor)
        else:
            data, target = decode_zoom(data, target, factor)

    return data, target


def decode(args, data, target):
    data_dec = []
    target_dec = []
    ipc = len(data) // args.nclass
    for c in range(args.nclass):
        idx_from = ipc * c
        idx_to = ipc * (c + 1)
        data_ = data[idx_from:idx_to].detach()
        target_ = target[idx_from:idx_to].detach()
        data_, target_ = decode_fn(data_,
                                   target_,
                                   args.factor,
                                   args.decode_type,
                                   bound=args.batch_syn_max)
        data_dec.append(data_)
        target_dec.append(target_)

    data_dec = torch.cat(data_dec)
    target_dec = torch.cat(target_dec)

    print("Dataset is decoded! ", data_dec.shape)
    save_img('./results/test_dec.png', data_dec, unnormalize=False, dataname=args.dataset)
    return data_dec, target_dec


def load_data_path(args):
    """Load condensed data from the given path
    """
    if args.pretrained:
        args.augment = False

    print()
    if args.dataset == 'imagenet':
        traindir = os.path.join(args.imagenet_dir, 'train')
        valdir = os.path.join(args.imagenet_dir, 'val')

        train_transform, test_transform = transform_imagenet(augment=args.augment,
                                                             from_tensor=False,
                                                             size=args.size,
                                                             rrc=args.rrc)
        # Load condensed dataset
        if 'idc' in args.slct_type:
            if args.slct_type == 'idc':
                data, target = torch.load(os.path.join(f'{args.save_dir}', 'data_com_best.pt'))         # modi:

            elif args.slct_type == 'idc_cat':
                data_all = []
                target_all = []
                for idx in range(args.nclass // args.nclass_sub):
                    path = f'{args.save_dir}_{args.nclass_sub}_phase{idx}'
                    data, target = torch.load(os.path.join(path, 'data_com_best.pt'))         # modi:
                    data_all.append(data)
                    target_all.append(target)
                    print(f"Load data from {path}")

                data = torch.cat(data_all)
                target = torch.cat(target_all)

            print("Load condensed data ", data.shape, args.save_dir)

            if args.factor > 1:
                data, target = decode(args, data, target)
            train_transform, _ = transform_imagenet(augment=args.augment,
                                                    from_tensor=True,
                                                    size=args.size,
                                                    rrc=args.rrc)
            train_dataset = TensorDataset(data, target, train_transform)
        else:
            train_dataset = ImageFolder(traindir,
                                        train_transform,
                                        nclass=args.nclass,
                                        seed=args.dseed,
                                        slct_type=args.slct_type,
                                        ipc=args.ipc_k,
                                        load_memory=args.load_memory)
            print(f"Test {args.dataset} random selection {args.ipc_k} (total {len(train_dataset)})")
        val_dataset = ImageFolder(valdir,
                                  test_transform,
                                  nclass=args.nclass,
                                  seed=args.dseed,
                                  load_memory=args.load_memory)

    else:
        if args.dataset[:5] == 'cifar':
            transform_fn = transform_cifar
        elif args.dataset == 'svhn':
            transform_fn = transform_svhn
        elif args.dataset == 'mnist':
            transform_fn = transform_mnist
        elif args.dataset == 'fashion':
            transform_fn = transform_fashion
        train_transform, test_transform = transform_fn(augment=args.augment, from_tensor=False)

        # Load condensed dataset
        if 'idc' in args.slct_type:
            data, target = torch.load(os.path.join(f'{args.save_dir}', 'data_com_best.pt'))         # modi:
            print("Load condensed data ", args.save_dir, data.shape)
            # This does not make difference to the performance
            # data = torch.clamp(data, min=0., max=1.)
            if args.factor > 1:
                data, target = decode(args, data, target)

            train_transform, _ = transform_fn(augment=args.augment, from_tensor=True)
            train_dataset = TensorDataset(data, target, train_transform)

        elif args.slct_type in ['dsa', 'kip']:
            condensed = torch.load(f'{args.save_dir}.pt')
            try:
                condensed = condensed['data']
                data = condensed[-1][0]
                target = condensed[-1][1]
            except:
                data = condensed[0].permute(0, 3, 1, 2)
                target = torch.arange(args.nclass).repeat_interleave(len(data) // args.nclass)

            if args.factor > 1:
                data, target = decode(args, data, target)
            # These data are saved as the normalized values!
            train_transform, _ = transform_fn(augment=args.augment,
                                              from_tensor=True,
                                              normalize=False)
            train_dataset = TensorDataset(data, target, train_transform)
            print("Load condensed data ", args.save_dir, data.shape)

        else:
            if args.dataset == 'cifar10':
                train_dataset = torchvision.datasets.CIFAR10(args.data_dir,
                                                             train=True,
                                                             transform=train_transform)
            elif args.dataset == 'cifar100':
                train_dataset = torchvision.datasets.CIFAR100(args.data_dir,
                                                              train=True,
                                                              transform=train_transform)
            elif args.dataset == 'svhn':
                train_dataset = torchvision.datasets.SVHN(os.path.join(args.data_dir, 'svhn'),
                                                          split='train',
                                                          transform=train_transform)
                train_dataset.targets = train_dataset.labels
            elif args.dataset == 'mnist':
                train_dataset = torchvision.datasets.MNIST(args.data_dir,
                                                           train=True,
                                                           transform=train_transform)
            elif args.dataset == 'fashion':
                train_dataset = torchvision.datasets.FashionMNIST(args.data_dir,
                                                                  train=True,
                                                                  transform=train_transform)
            if args.record:     # *only when recording the losses in every iteration
                print(f"Load all  data (total {len(train_dataset)})")
            else:
                indices = randomselect(train_dataset, args.ipc_k, nclass=args.nclass)
                train_dataset = Subset(train_dataset, indices)
                print(f"Random select {args.ipc_k} data (total {len(indices)})")


        # Test dataset
        if args.dataset == 'cifar10':
            val_dataset = torchvision.datasets.CIFAR10(args.data_dir,
                                                       train=False,
                                                       transform=test_transform)
        elif args.dataset == 'cifar100':
            val_dataset = torchvision.datasets.CIFAR100(args.data_dir,
                                                        train=False,
                                                        transform=test_transform)
        elif args.dataset == 'svhn':
            val_dataset = torchvision.datasets.SVHN(os.path.join(args.data_dir, 'svhn'),
                                                    split='test',
                                                    transform=test_transform)
        elif args.dataset == 'mnist':
            val_dataset = torchvision.datasets.MNIST(args.data_dir,
                                                     train=False,
                                                     transform=test_transform)
        elif args.dataset == 'fashion':
            val_dataset = torchvision.datasets.FashionMNIST(args.data_dir,
                                                            train=False,
                                                            transform=test_transform)

    # For sanity check
    print("Training data shape: ", train_dataset[0][0].shape)
    # os.makedirs('./results', exist_ok=True)
    # save_img('./results/test.png',
    #          torch.stack([d[0] for d in train_dataset]),
    #          dataname=args.dataset)
    print()

    return train_dataset, val_dataset


def test_data(args,
              train_loader,
              val_loader,
              test_resnet=False,
              model=None,
              model_fn=None,
              logger=print,
              num_val=4,
              repeat=1,
              record=False):
    """Train neural networks on condensed data
    """

    args.epoch_print_freq = math.ceil(args.epochs / num_val)

    if model is None:
        if model_fn is None and test_resnet:
            model_fn = resnet10_bn
        elif model_fn is None:
            model_fn = define_model
        else:
            model_fn = model_fn

    best_acc_l = []
    acc_l = []

    if not record:
        for i in range(repeat):
            if model is None:
                m = model_fn(args, args.nclass, logger=logger)
            else:
                m = copy.deepcopy(model)
            best_acc, acc = train(args, m, train_loader, val_loader, logger=str)
            best_acc_l.append(best_acc)
            acc_l.append(acc)
        logger(
            f'Best, last acc: {np.mean(best_acc_l):.1f} {np.mean(acc_l):.1f}\n')

        # modi:
        return m, np.mean(best_acc_l), np.mean(acc_l)
    else:
        if model is None:
            m = model_fn(args, args.nclass, logger=logger)
        else:
            m = copy.deepcopy(model)
        best_acc, acc, losses_record = train(args, m, train_loader, val_loader, logger=str, record=num_val)
        return m, best_acc, acc, losses_record



if __name__ == '__main__':
    from argument import args, inner_setting
    import torch.backends.cudnn as cudnn
    import numpy as np

    if args.seed > 0:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)

    cudnn.benchmark = True

    if args.same_compute and args.factor > 1:
        args.epochs = int(args.epochs / args.factor**2)

    best_acc_avg = np.zeros((args.repeat, 3))
    acc_avg = np.zeros((args.repeat, 3))
    for i in range(args.repeat):
        model1, model2, model3 = None, None, None
        for k, (ipc, p) in enumerate(zip(eval(args.ipcs), eval(args.test_paths))):
            args.k = k
            args.ipc_k = ipc
            if args.reproduce:
                args = set_arguments(args)

            args = inner_setting(args)
            args.save_dir = os.path.join(args.data_path, p)

            if args.slct_type == 'herding':
                train_dataset, val_dataset = herding(args)
            else:
                syn_dataset, val_dataset = load_data_path(args)
                train_dataset, val_dataset = load_data_path(args)

            train_loader = MultiEpochsDataLoader(train_dataset,
                                                 batch_size=args.batch_size,
                                                 shuffle=True,
                                                 num_workers=args.workers if args.augment else 0,
                                                 persistent_workers=args.augment > 0)
            val_loader = MultiEpochsDataLoader(val_dataset,
                                               batch_size=args.batch_size // 2,
                                               shuffle=False,
                                               persistent_workers=True,
                                               num_workers=4)
            print(f"\nTest in cumulate S{k}")
            model1, best_acc1, acc1 = test_data(args, train_loader, val_loader, test_resnet=False, model=model1)           # modi:
            if args.dataset[:5] == 'cifar':
                model2, best_acc2, acc2 = test_data(args, train_loader, val_loader, model_fn=resnet18_bn, model=model2)    # modi:
                if (not args.same_compute) and (args.ipc_k >= 50 and args.factor > 1):
                    args.epochs = 400
                model3, best_acc3, acc3 = test_data(args, train_loader, val_loader, model_fn=densenet, model=model3)       # modi:
            elif args.dataset == 'imagenet':
                model2, best_acc2, acc2 = test_data(args, train_loader, val_loader, model_fn=resnet18_bn, model=model2)    # modi:
                model3, best_acc3, acc3 = test_data(args, train_loader, val_loader, model_fn=efficientnet, model=model3)   # modi:
            else:
                best_acc2, best_acc3 = 0, 0
                acc2, acc3 = 0, 0

        best_acc_avg[i] = np.array([best_acc1, best_acc2, best_acc3])
        acc_avg[i]= np.array([acc1, acc2, acc3])

    print(f"Average Best ACC = {np.mean(best_acc_avg, 0)}, Average ACC = {np.mean(acc_avg, 0)}, "
          f"Average Best STD = {np.std(best_acc_avg, 0)}, Average STD = {np.std(acc_avg, 0)}")


