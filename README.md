# Sequential Subset Matching for Dataset Distillation
[Paper](https://arxiv.org/abs/2311.01570)

The code is for training expert trajectories and distilling synthetic data from our Sequential Subset Matching for Dataset Distillation paper (NIPS 2023).
 We provide the Sequential Subset Matching methods in [MTT](https://arxiv.org/abs/2203.11932) model and [IDC](https://arxiv.org/abs/2205.14959) model

### Getting Started
#### Download
```bash
git clone https://github.com/shqii1j/seqmatch.git
cd seqmatch
```
#### The requirement for SeqMatch in MTT model
If you have an RTX 30XX GPU (or newer), run

```bash
conda env create -f requirements_11_3.yaml
```

If you have an RTX 20XX GPU (or older), run

```bash
conda env create -f requirements_10_2.yaml
```

You can then activate your conda environment with
```bash
conda activate distillation
```
#### The requirement for SeqMatch in IDC model
If you have created distillation environment, run it

```bash
conda activate distillation
```

If not, install the packages ```pytorch``` and ```efficientnet_pytorch```

### Sequential Subset Matching in MTT
There is an example .sh file to use our code. This command will generate 3 subsets to distill CIFAR-10 50 image per class:
```bash
cd seqmatch-mtt
python run.sh
```

Using ```buffer.py```, you can generate some expert trajectories for the first subset. The following command will train 100 ConvNet models on CIFAR-10 with ZCA whitening for 50 epochs each:
```bash
python buffer.py --dataset=CIFAR10 --model=ConvNet --train_epochs=50 --num_experts=100 --zca --buffer_path={path_to_buffer_storage} --data_path={path_to_dataset}
```

Using ```distill_eval_new.py```, you can generate the first subset via the buffers. The following command will generate the first distilled subset for CIFAR-10 down to just 1 image per class:
```bash
python distill.py --dataset=CIFAR10 --ipc=1 --syn_steps=30 --expert_epochs=2 --max_start_epoch=20 --zca --lr_img=100 --lr_lr=1e-05 --lr_teacher=0.01 --buffer_path={path_to_buffer_storage} --data_path={path_to_dataset} --run_name={path to the task} --name={path to the subset}
```

For the following subsets, you need to use ```--pre_names``` and ```--reparam_syn``` flags. The following command will generate the expert trajectories based on the previous subsets and the new distilled subset for CIFAR-10:
```bash
python buffer.py --dataset=CIFAR10 --model=ConvNet --train_epochs=20 --num_experts=100 --zca --image_path=logged_files --data_path={path_to_dataset} --run_name={path to the task} --pre_names={paths to the previous subsets} --reparam_syn

python distill_eval_new.py --dataset=CIFAR10 --ipc=1 --syn_steps=30 --expert_epochs=2 --zca --image_path=logged_files --data_path={path_to_dataset} --buffer_path=./logged_files/CIFAR10/{path to the task}/{paths to the last subset}/buffer --intervals=0-20 --lr_img=100 --lr_lr=1e-05 --lr_teacher=0.01 --run_name={path to the task} --pre_names={paths to the previous subsets} --name={path to the new subset} --reparam_syn
```

Please find a full list of hyper-parameters in our paper (https://arxiv.org/abs/2311.01570).

#### ImageNet
When generating expert trajectories with ```buffer.py``` or distilling the dataset with ```distill.py``` for ImageNet, you must designate a named subset of ImageNet with the ```--subset``` flag.

### Sequential Subset Matching in IDC
There is an example .sh file to use our code. This command will generate 2 subsets to distill CIFAR-10 50 image per class (25 image per class in one subset):
```bash
cd seqmatch-idc
python condense_new.py --reproduce -d cifar10 -f 2 --ipcs=[25,25] --inner_loop=[50,100] --niters=[2000,4000] --lrs_img_ori=[5e-3,5e-3] --it_log=100 --it_eval=100 --seed=2023 --fix_iter=50
```
You can get more robust result (repeat test 5 times) via th following code:
```bash
python test.py -d cifar10 -n convnet -f 2 --reproduce --ipcs=[5,5] --repeat=5 --seed=2023 --data_path={path to the results} --test_paths={paths to the subsets} 
```


# Reference
If you find our code useful for your research, please cite our paper.
```
@inproceedings{
du2023sequential,
title={Sequential Subset Matching for Dataset Distillation},
author={Jiawei Du and Qin Shi and Joey Tianyi Zhou},
booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
year={2023}
}
```
