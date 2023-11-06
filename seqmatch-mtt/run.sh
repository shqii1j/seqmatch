DATASET="CIFAR10"
RUN_NAME="ipc50_3S_1"
INTERVAL1="0-20"
INTERVAL2="0-20"
INTERVAL3="0-10"
IPC1=17
IPC2=17
IPC3=16

'''init buffer'''
python buffer.py --dataset=$DATASET --model=ConvNet --train_epochs=50 --num_experts=100 --zca \
--buffer_path=buffer --data_path=dataset

'''first distill'''
python distill_eval_new.py --dataset=$DATASET --ipc=$IPC1 --syn_steps=30 --expert_epochs=2 --zca \
--buffer_path=buffer/$DATASET --data_path=dataset \
--intervals=$INTERVAL1 --lr_img=100 --lr_lr=1e-05 --lr_teacher=0.001 \
--run_name=$RUN_NAME --name=S1

'''buffer on S1'''
python buffer.py --dataset=$DATASET --model=ConvNet --train_epochs=20 --num_experts=100 --zca \
--data_path=dataset --image_path=logged_files \
--run_name=$RUN_NAME --pre_names=S1 --reparam_syn

'''sencond distill'''
python distill_eval_new.py --dataset=$DATASET --ipc=$IPC2 --syn_steps=30 --expert_epochs=2 --data_path=dataset --zca \
--buffer_path=./logged_files/$DATASET/$RUN_NAME/S1/buffer \
--intervals=$INTERVAL2 --lr_img=100 --lr_lr=1e-05 --lr_teacher=0.01 \
--run_name=$RUN_NAME --pre_names=S1 --name=S2 --reparam_syn

'''buffer on S2'''
python buffer.py --dataset=$DATASET --model=ConvNet --train_epochs=20 --num_experts=100 --zca \
--data_path=dataset --image_path=logged_files \
--run_name=$RUN_NAME --pre_names=S1,S2 --reparam_syn

'''third distill'''
python distill_eval_new.py --dataset=$DATASET --ipc=$IPC3 --syn_steps=30 --expert_epochs=2 --data_path=dataset --zca \
--buffer_path=./logged_files/$DATASET/$RUN_NAME/S2/buffer \
--intervals=$INTERVAL3 --lr_img=100 --lr_lr=1e-05 --lr_teacher=0.01 \
--run_name=$RUN_NAME --pre_names=S1,S2 --name=S3 --reparam_syn
