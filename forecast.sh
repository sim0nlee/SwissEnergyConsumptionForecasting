#!/bin/bash
method="anio"
anchoring="week" # only has effect if anio
loss_type="single"
model="resnet"
#model="resnet"
#dataset="spain"
dataset="en"
device="cuda"

feature_scaler="max"
load_scaler=""
epochs=2000
lr=0.001
batch_size=32
scheduler_start=500
scheduler_step=500
lr_decay=0.5

use_tensorboard=1
plot_predictions=0

options=\
"--method=$method \
--anchoring=$anchoring
--loss_type=$loss_type
--model=$model
--dataset=$dataset \
--device=$device \
--load_scaler=$load_scaler \
--feature_scaler=$feature_scaler \
--epochs=$epochs \
--lr=$lr \
--batch_size=$batch_size \
--scheduler_start=$scheduler_start \
--scheduler_step=$scheduler_step \
--lr_decay=$lr_decay"

if [[ $use_tensorboard = 1 ]]; then
  options="$options --use_tensorboard"
fi
if [[ $plot_predictions = 1 ]]; then
  options="$options --plot_predictions"
fi

#if [[ "$CONDA_DEFAULT_ENV" != "semproject" ]]; then
#  echo "Not in semproject environment"
#fi

python forecast.py $options
