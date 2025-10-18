#!/bin/sh
t=9

# Inference
CUDA_VISIBLE_DEVICES=0 \
python eval.py \
--mode cil \
--dataset Imagenet100_10t \
--model resnet18 \
--batch_size 128 \
--t $t \
--all_dataset \
--printfn "cil test results.txt" \
--data_path ./data