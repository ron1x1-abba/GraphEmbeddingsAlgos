#!/usr/bin/bash

DATA_DIR="../../../datasets"
SAVE_DIR="../../../models"
LOG_DIR="../../../mylogs"

TRAIN_DATASET="--train_dataset=${DATA_DIR}/test_graph_50k.pkl"

SAVE_PATH="--save_path=${SAVE_DIR}/TransE/"

ETA="--eta=5"
ETA_VAL="--eta_val=20"
TRAIN_CORRUPT="--train_corrupt=s+o"
VAL_CORRUPT="--val_corrupt=s+o"
COMPARISSON_TYPE="--comparisson_type=best"

VAL_RATIO="--val_ratio=0.05"
TRAIN_BS="--train_bs=512"
VAL_BS="--val_bs=128"
EPOCHS="--epochs=50"
LR="--lr=1e-3"
PCT_START="--pct_start=0.05"
STRATEGY="--strategy=None"

LOGDIR="--logdir=${LOG_DIR}/TransE/"
GPUS="--gpus=1"

python train_transe.py ${TRAIN_BS} ${VAL_BS} ${SAVE_PATH} ${EPOCHS} ${LOGDIR} ${GPUS} ${LR} \
    ${PCT_START} ${ETA} ${ETA_VAL} ${TRAIN_CORRUPT} ${VAL_CORRUPT} ${TRAIN_DATASET} ${COMPARISSON_TYPE} ${VAL_RATIO} \
    ${STRATEGY}
