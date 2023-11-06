import os
import argparse
import pdb

import numpy as np
import torch
from utils import get_dataset, get_network, get_eval_pool, evaluate_synset, get_time, DiffAugment, ParamDiffAug
import copy

def main(args):
    if args.zca and args.texture:
        raise AssertionError("Cannot use zca and texture together")
    if args.texture and args.pix_init == "real":
        print("WARNING: Using texture with real initialization will take a very long time to smooth out the boundaries between images.")

    print("CUDNN STATUS: {}".format(torch.backends.cudnn.enabled))

    args.dsa = True if args.dsa == 'True' else False
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader, loader_train_dict, class_map, class_map_inv = get_dataset(args.dataset, args.data_path, args.batch_real, args.subset, args=args)
    model_eval_pool = get_eval_pool(args.eval_mode, args.model, args.model)

    args.im_size = im_size

    if args.dsa:
        # args.epoch_eval_train = 1000
        args.dc_aug_param = None

    args.dsa_param = ParamDiffAug()
    dsa_params = args.dsa_param
    if args.zca:
        zca_trans = args.zca_trans
    else:
        zca_trans = None
    args.dsa_param = dsa_params
    args.zca_trans = zca_trans

    args.distributed = torch.cuda.device_count() > 1

    ''' *Load the subsets '''
    image_path = os.path.join(args.image_path, args.dataset, args.run_name)
    image_folder = args.file_names.split(',')

    img_l = []
    lab_l = []
    args.lrs_net = []
    if args.lrs_net:
        args.lrs_net = [torch.tensor(eval(lr)).to(args.device).item() for lr in args.lrs_net.split(',')]

    for f in image_folder:
        print(f'\nLoad the subset {f}')
        image_name = 'images_best.pt'
        labels_name = 'labels_best.pt'
        img = torch.load(os.path.join(image_path, f, image_name)).to(args.device).requires_grad_(True)
        lab = torch.load(os.path.join(image_path, f, labels_name))

        if not args.lrs_net:
            args.lrs_net.append(torch.load(os.path.join(image_path, f, 'best_lr.pt')))

        if img_l:
            img_l.append(torch.cat([img_l[-1], img], dim=0))
            lab_l.append(torch.cat([lab_l[-1], lab], dim=0))
        else:
            img_l.append(img.to(args.device).requires_grad_(True))
            lab_l.append(lab)


    model_eval_pool = get_eval_pool(args.eval_mode, args.model, args.model)

    if args.dsa:
        print('DSA augmentation strategy: \n', args.dsa_strategy)
        print('DSA augmentation parameters: \n', args.dsa_param.__dict__)
    else:
        print('DC augmentation parameters: \n', args.dc_aug_param)

    acc_test_mean = dict()
    acc_test_std = dict()


    ''' Evaluate '''
    for i, model_eval in enumerate(model_eval_pool):
        if isinstance(model_eval, str):
            print(f'EVAL: {model_eval} ---------------')
        else:
            print(f'EVAL: model_{i} ---------------')
        accs_test = []
        accs_train = []

        for it_eval in range(args.num_eval):
            net_eval = get_network(model_eval, channel, num_classes, im_size).to(args.device)  # get a random model
            for img, lab, lr in zip(img_l, lab_l, args.lrs_net):
                eval_labs = lab
                with torch.no_grad():
                    image_save = img
                image_syn_eval, label_syn_eval = copy.deepcopy(image_save.detach()), copy.deepcopy(
                    eval_labs.detach())  # avoid any unaware modification
                args.lr_net = torch.tensor(eval(lr)).to(args.device).requires_grad_(True).item()

                net_eval, acc_train, acc_test = evaluate_synset(it_eval, net_eval, image_syn_eval, label_syn_eval, testloader, args,
                                                     texture=args.texture, printer=True)

            accs_test.append(acc_test)
            accs_train.append(acc_train)

        accs_test = np.array(accs_test)
        accs_train = np.array(accs_train)
        if isinstance(model_eval,str):
            acc_test_mean[model_eval] = np.mean(accs_test)
            acc_test_std[model_eval] = np.std(accs_test)
        else:
            acc_test_mean['model_'+str(i)] = np.mean(accs_test)
            acc_test_std['model_'+str(i)] = np.std(accs_test)
        print(np.mean(accs_test), np.std(accs_test))
    print(acc_test_mean)
    print(acc_test_std)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parameter Processing')

    parser.add_argument('--dataset', type=str, default='CIFAR10', help='dataset')
    parser.add_argument('--subset', type=str, default='imagenette', help='ImageNet subset. This only does anything when --dataset=ImageNet')
    parser.add_argument('--model', type=str, default='ConvNet', help='model')
    parser.add_argument('--eval_mode', type=str, default='S',
                        help='eval_mode, check utils.py for more info')
    parser.add_argument('--num_eval', type=int, default=5, help='how many networks to evaluate on')
    parser.add_argument('--epoch_eval_train', type=int, default=1000,
                        help='epochs to train a model with synthetic data')

    parser.add_argument('--batch_real', type=int, default=256, help='batch size for real data')
    parser.add_argument('--batch_train', type=int, default=256, help='batch size for training networks')

    parser.add_argument('--dsa', type=str, default='True', choices=['True', 'False'],
                        help='whether to use differentiable Siamese augmentation.')
    parser.add_argument('--dsa_strategy', type=str, default='color_crop_cutout_flip_scale_rotate',
                        help='differentiable Siamese augmentation strategy')

    parser.add_argument('--data_path', type=str, default='data', help='dataset path')
    parser.add_argument('--buffer_path', type=str, default='./buffers', help='buffer path')
    parser.add_argument('--image_path', type=str, default='logged_files', help='the image path of the subsets')
    parser.add_argument('--run_name', type=str, default=None, help='the name of the experiment')
    parser.add_argument('--file_names', type=str, default=None, help='the name of the subsets')
    parser.add_argument('--lrs_net', type=str, default='0.01', help='the learning rate in the eval model')

    parser.add_argument('--zca', action='store_true', help="do ZCA whitening")
    parser.add_argument('--no_aug', type=bool, default=False, help='this turns off diff aug during distillation')
    parser.add_argument('--texture', action='store_true', help="will distill textures instead")
    parser.add_argument('--canvas_size', type=int, default=2, help='size of synthetic canvas')
    parser.add_argument('--canvas_samples', type=int, default=1, help='number of canvas samples per iteration')

    args = parser.parse_args()

    main(args)

