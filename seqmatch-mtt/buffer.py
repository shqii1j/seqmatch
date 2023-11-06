import os
import argparse
import torch
import torch.nn as nn
from tqdm import tqdm
from utils import get_dataset, get_network, get_daparam, get_eval_pool, TensorDataset, epoch, ParamDiffAug, DiffAugment, evaluate_synset
import copy
from random import choice
from reparam_module import ReparamModule
import warnings
import pdb
warnings.filterwarnings("ignore", category=DeprecationWarning)


def main(args):

    args.dsa = True if args.dsa == 'True' else False
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.dsa_param = ParamDiffAug()
    print('Hyper-parameters: \n', args.__dict__)

    save_dir = os.path.join(args.buffer_path, args.dataset)


    ''' Organize the real dataset '''
    channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader, loader_train_dict, class_map, class_map_inv = get_dataset(args.dataset, args.data_path, args.batch_real, args.subset, args=args)

    images_all = []
    labels_all = []
    indices_class = [[] for c in range(num_classes)]

    print("BUILDING DATASET")
    for i in tqdm(range(len(dst_train))):
        sample = dst_train[i]
        images_all.append(torch.unsqueeze(sample[0], dim=0))
        labels_all.append(class_map[torch.tensor(sample[1]).item()])
    images_all = torch.cat(images_all, dim=0).to("cpu")
    labels_all = torch.tensor(labels_all, dtype=torch.long, device="cpu")

    for i, lab in tqdm(enumerate(labels_all)):
        indices_class[lab].append(i)
    for c in range(num_classes):
        print('class c = %d: %d real images'%(c, len(indices_class[c])))
    for ch in range(channel):
        print('real images channel %d, mean = %.4f, std = %.4f'%(ch, torch.mean(images_all[:, ch]), torch.std(images_all[:, ch])))

    dst_train = TensorDataset(copy.deepcopy(images_all.detach()), copy.deepcopy(labels_all.detach()))
    trainloader = torch.utils.data.DataLoader(dst_train, batch_size=args.batch_train, shuffle=True, num_workers=0)

    ''' Set augmentation for whole-dataset training '''
    args.dc_aug_param = get_daparam(args.dataset, args.model, args.model, None)
    args.dc_aug_param['strategy'] = 'crop_scale_rotate'  # for whole-dataset training
    print('DC augmentation parameters: \n', args.dc_aug_param)


    ''' *Load previous subsets '''
    if args.reparam_syn:
        image_path = os.path.join(args.image_path, args.dataset, args.run_name)
        images_best = []
        labels_best = []
        args.lrs_net = []       # *There are two ways to load args.lrs_net: (1) Manually add parameter lrs_net (2) Load the best_lr.pt files in Line 78
        if args.lrs_net:
            args.lrs_net = [torch.tensor(eval(lr)).to(args.device).item() for lr in args.lrs_net.split(',')]

        for f in args.pre_names.split(','):
            if images_best:
                image_syn = torch.cat([images_best[-1], torch.load(os.path.join(image_path, f, 'images_best.pt'))], dim=0)
                label_syn = torch.cat([labels_best[-1], torch.load(os.path.join(image_path, f, 'labels_best.pt'))], dim=0)
            else:
                image_syn = torch.load(os.path.join(image_path, f, 'images_best.pt'))
                label_syn = torch.load(os.path.join(image_path, f, 'labels_best.pt'))
            if args.dsa and (not args.no_aug):
                DiffAugment(image_syn, args.dsa_strategy, param=args.dsa_param)
            images_best.append(image_syn)
            labels_best.append(label_syn)

            if not args.lrs_net:
                args.lrs_net.append(torch.load(os.path.join(image_path, f, 'best_lr.pt')))

        save_dir = os.path.join(image_path, f, 'buffer')

    if args.dataset == "ImageNet":
        save_dir = os.path.join(save_dir, args.subset)
    if args.dataset in ["CIFAR10", "CIFAR100", "SVHN"] and not args.zca:
        save_dir += "_NO_ZCA"
    save_dir = os.path.join(save_dir, args.model)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)


    ''' *Buffer '''
    criterion = nn.CrossEntropyLoss().to(args.device)
    trajectories = []
    for it in range(0, args.num_experts):
        ''' *initialize teacher network'''
        teacher_net = get_network(args.model, channel, num_classes, im_size).to(args.device)
        teacher_net.train()
        if args.reparam_syn:
            net_eval = get_network(args.model, channel, num_classes, im_size).to(
                args.device)
            for image_syn, label_syn, lr in zip(images_best, labels_best, args.lrs_net):
                eval_labs = label_syn
                args.lr_net = lr
                with torch.no_grad():
                    image_save = image_syn
                image_syn_eval, label_syn_eval = copy.deepcopy(image_save.detach()), copy.deepcopy(eval_labs.detach())  # avoid any unaware modification
                net_eval, acc_train, acc_test = evaluate_synset(it, net_eval, image_syn_eval, label_syn_eval,
                                                                testloader, args,
                                                                texture=args.texture, printer=True)
            teacher_net.load_state_dict(net_eval.state_dict())

        lr = args.lr_teacher
        teacher_optim = torch.optim.SGD(teacher_net.parameters(), lr=lr, momentum=args.mom, weight_decay=args.l2)  # optimizer_img for synthetic data
        teacher_optim.zero_grad()
        timestamps = []
        timestamps.append([p.detach().cpu() for p in teacher_net.parameters()])

        lr_schedule = [args.train_epochs // 2 + 1]

        for e in range(args.train_epochs):
            ''' *update teacher network '''
            train_loss, train_acc = epoch("train", dataloader=trainloader, net=teacher_net, optimizer=teacher_optim,
                                        criterion=criterion, args=args, aug=True)
            test_loss, test_acc = epoch("test", dataloader=testloader, net=teacher_net, optimizer=None,
                                        criterion=criterion, args=args, aug=False)
            print("Itr: {}\tEpoch: {}\tTrain Acc: {}\tTest Acc: {}".format(it, e, train_acc, test_acc))

            timestamps.append([p.detach().cpu() for p in teacher_net.parameters()])

            if e in lr_schedule and args.decay:
                lr *= 0.1
                teacher_optim = torch.optim.SGD(teacher_net.parameters(), lr=lr, momentum=args.mom, weight_decay=args.l2)
                teacher_optim.zero_grad()

        trajectories.append(timestamps)

        if len(trajectories) == args.save_interval:
            n = 0
            while os.path.exists(os.path.join(save_dir, "replay_buffer_{}.pt".format(n))):
                n += 1
            print("Saving {}".format(os.path.join(save_dir, "replay_buffer_{}.pt".format(n))))
            torch.save(trajectories, os.path.join(save_dir, "replay_buffer_{}.pt".format(n)))
            trajectories = []


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parameter Processing')
    parser.add_argument('--dataset', type=str, default='CIFAR10', help='dataset')
    parser.add_argument('--subset', type=str, default='imagenette', help='ImageNet subset. This only does anything when --dataset=ImageNet')
    parser.add_argument('--model', type=str, default='ConvNet', help='model')
    parser.add_argument('--num_experts', type=int, default=100, help='training iterations')
    parser.add_argument('--lr_teacher', type=float, default=0.01, help='learning rate for updating network parameters')
    parser.add_argument('--batch_train', type=int, default=256, help='batch size for training networks')
    parser.add_argument('--batch_real', type=int, default=256, help='batch size for real loader')
    parser.add_argument('--dsa', type=str, default='True', choices=['True', 'False'],
                        help='whether to use differentiable Siamese augmentation.')
    parser.add_argument('--dsa_strategy', type=str, default='color_crop_cutout_flip_scale_rotate',
                        help='differentiable Siamese augmentation strategy')
    parser.add_argument('--data_path', type=str, default='data', help='dataset path')
    parser.add_argument('--buffer_path', type=str, default='./buffers', help='buffer path')
    parser.add_argument('--train_epochs', type=int, default=50, help='*the length of teacher trajectories')
    parser.add_argument('--zca', action='store_true')
    parser.add_argument('--decay', action='store_true')
    parser.add_argument('--mom', type=float, default=0, help='momentum')
    parser.add_argument('--l2', type=float, default=0, help='l2 regularization')
    parser.add_argument('--save_interval', type=int, default=10, help='how often to save the teacher trajectories')

    parser.add_argument('--reparam_syn', action='store_true', help='*use it when need to load the previous subsets')
    parser.add_argument('--image_path', type=str, default='logged_files', help="*the path for saving")
    parser.add_argument('--run_name', type=str, default=None, help="*the name of the experiment")
    parser.add_argument('--pre_names', type=str, default=None, help="the names of the previous subsets")
    parser.add_argument('--lrs_net', type=str, default=None, help="The lrs of the previous subsets")
    parser.add_argument('--no_aug', type=bool, default=False, help='this turns off diff aug during distillation')
    parser.add_argument('--texture', action='store_true', help="will distill textures instead")

    args = parser.parse_args()
    main(args)


