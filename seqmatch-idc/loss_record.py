import torch
from test import *

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

    args.record = True

    syn_slct_type = copy.deepcopy(args.slct_type)
    args.slct_type = 'origin'
    whole_dataset, val_dataset = load_data_path(args)


    model1, model2, model3 = None, None, None
    losses_records = []
    for k, (ipc, p, num_record) in enumerate(zip(eval(args.ipcs), eval(args.test_paths), eval(args.snapshots))):
        args.k = k
        args.ipc_k = ipc
        if args.reproduce:
            args = set_arguments(args)

        args = inner_setting(args)
        args.save_dir = os.path.join(args.data_path, p)

        args.slct_type = syn_slct_type
        if args.slct_type == 'herding':
            train_dataset, val_dataset = herding(args)
        else:
            train_dataset, val_dataset = load_data_path(args)


        train_loader = MultiEpochsDataLoader(train_dataset,
                                             batch_size=args.batch_size,
                                             shuffle=True,
                                             num_workers=args.workers if args.augment else 0,
                                             persistent_workers=args.augment > 0)
        whole_loader = MultiEpochsDataLoader(whole_dataset,
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
        model1, best_acc1, acc1, losses_record = test_data(args, train_loader, val_loader, test_resnet=False, model=model1, record=True, num_val=num_record)           # modi:
        losses_records.append(losses_record)
        print(f"Average Best ACC = {best_acc1}, Average ACC = {acc1}")

    losses_records = torch.cat(losses_records, dim=1)
    torch.save(losses_records, os.path.join(args.data_path, f'{args.number}training_loss_snapshot{args.snapshots}_.pt'))
