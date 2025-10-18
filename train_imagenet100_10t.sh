#!/bin/sh
t=0

# Backbone train
CUDA_VISIBLE_DEVICES=0 \
python train.py \
--dataset Imagenet100_10t \
--model resnet18 \
--mode train_IOE_DAC \
--batch_size 32 \
--epoch 100 \
--DAC_epoch 0 \
--t $t \
--amp \
--lamb0 1.0 \
--lamb1 0.75 \
--data_path /root/autodl-tmp

# OOD classifier train
CUDA_VISIBLE_DEVICES=0 \
python train.py \
--mode train_OOD_classifier \
--dataset Imagenet100_10t \
--model resnet18 \
--batch_size 64 \
--epoch 50 \
--t $t \
--data_path /root/autodl-tmp

# Inference
CUDA_VISIBLE_DEVICES=0 \
python eval.py \
--mode cil \
--dataset Imagenet100_10t \
--model resnet18 \
--batch_size 128 \
--t $t \
--all_dataset \
--printfn "cil results.txt" \
--data_path /root/autodl-tmp

for t in 1 2 3 4 5 6 7 8 9
do
	# Backbone train
	CUDA_VISIBLE_DEVICES=0 \
	python train.py \
	--dataset Imagenet100_10t \
	--model resnet18 \
	--mode train_IOE_DAC \
	--batch_size 32 \
	--epoch 100 \
    --DAC_epoch 0 \
	--t $t \
	--amp \
	--lamb0 1.0 \
	--lamb1 0.75 \
    --data_path /root/autodl-tmp

	# OOD classifier train
	CUDA_VISIBLE_DEVICES=0 \
	python train.py \
	--mode train_OOD_classifier \
	--dataset Imagenet100_10t \
	--model resnet18 \
	--batch_size 64 \
	--epoch 50 \
	--t $t \
    --data_path /root/autodl-tmp

	# Inference
	CUDA_VISIBLE_DEVICES=0 \
	python eval.py \
	--mode cil \
	--dataset Imagenet100_10t \
	--model resnet18 \
	--batch_size 128 \
	--t $t \
	--all_dataset \
	--printfn "cil results.txt" \
    --data_path /root/autodl-tmp

done