DATASET="CIFAR10"
RUN_NAME="ipc50_3S_1"

python evaluation.py --data_path=dataset --buffer_path=buffer --dataset=$DATASET --zca \
--run_name=$RUN_NAME --file_names=S1,S2,S3

