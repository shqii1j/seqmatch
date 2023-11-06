python condense_new.py --reproduce -d cifar10 -f 2 --ipcs=[25,25] \
--inner_loop=[50,100] --niters=[2000,4000] --lrs_img_ori=[5e-3,5e-3] \
--it_log=100 --it_eval=100 --seed=2023 --fix_iter=50

# You can get more robust result (repeat test 5 times) via th following code
#python test.py -d cifar10 -n convnet -f 2 --reproduce --ipcs=[5,5] --repeat=5 --seed=2023 \
#--data_path='./results/cifar10/conv3in/<your time>/_grad_l1_fix50_nd500_cut_factor2_mix' \
#--test_paths="['S0_ipc25_inloop50_niter2000_lr0.0125','S1_ipc25_inloop100_niter4000_lr0.0125']"
