import os
import random
from time import time

import numpy as np
import torch
from matplotlib import pyplot as plt

from sklearn.metrics import mean_absolute_percentage_error
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import data
from models.mlp import MLP
from models.resnet import ResNet, ExtendedResNet
from models.lstm import LSTM, ExtendedLSTM
from models.initialization import kaiming_initialization
from utils.log import print_run_info
from utils.parser import parser
from utils.plot import plot_gt_pred_graph

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)

args = parser.parse_args()

tensorboard_file = (f'{args.dataset}_'
                    f'{args.method}_'
                    f'{args.loss_type}-loss_'
                    f'{args.model}_'
                    f'lr{args.lr}_'
                    f'scheduler_start{args.scheduler_start}_'
                    f'scheduler_step{args.scheduler_step}_'
                    f'gamma{args.lr_decay}_'
                    f'feature-scaler={args.feature_scaler}_'
                    f'load-scaler={args.load_scaler}')

if args.loss_type == 'double' and args.model not in ['resnet_ext', 'lstm_ext']:
    raise RuntimeError('Double loss only supported with extended models')
if args.model == 'resnet_ext' and args.loss_type == 'single':
    raise RuntimeError('Extended ResNet only supported with double loss')

extra = "_RELU"
tensorboard_file += extra

if args.use_tensorboard:
    writer = SummaryWriter("tensorboard_plots/" + tensorboard_file)
# if args.use_tensorboard:
#     writer = SummaryWriter("tensorboard_plots_best/" + tensorboard_file)

print_run_info(args)


########################################################################################################################

def get_model_from_type(model_type, n_features, n_out):
    match model_type:
        case 'mlp':
            return MLP(n_features, n_out).to(args.device)
        case 'resnet':
            return ResNet(n_features, n_out).to(args.device)
        case 'resnet_ext':
            return ExtendedResNet(n_features, n_out).to(args.device)
        case 'lstm':
            return LSTM(input_size=n_features, output_size=n_out).to(args.device)
        case 'lstm_ext':
            return ExtendedLSTM(input_size=n_features, output_size=n_out).to(args.device)
        case _:
            return None


(train_dataset,
 val_dataset,
 test_dataset) = data.get_train_val_test_split(args.dataset, args.method, args.anchoring, args.feature_scaler,
                                               args.load_scaler)

train_dl = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
val_dl = DataLoader(val_dataset, batch_size=256, drop_last=False)
test_dl = DataLoader(test_dataset, batch_size=256, drop_last=False)

n_features = train_dataset.features.shape[1]
n_out = train_dataset.load.shape[1]

model = get_model_from_type(args.model, n_features, n_out)
model.apply(kaiming_initialization)

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
criterion = torch.nn.SmoothL1Loss()
milestones = np.arange(args.scheduler_start, args.epochs, step=args.scheduler_step)
# milestones = [500, 1000, 1100] + list(np.arange(1300, args.epochs, step=args.scheduler_step))
sch = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=args.lr_decay)


# TRAIN / VAL / TEST LOOP #

def train(dataloader, model, criterion, optimizer, lr_scheduler, cur_epoch):
    model.train()

    training_running_loss = 0.
    training_batches = 0

    for idx, (training_features, training_targets) in enumerate(dataloader):
        training_features = training_features.to(args.device)
        training_targets = training_targets.to(args.device)

        optimizer.zero_grad()

        if args.loss_type == 'double':
            training_output_1, training_output_2 = model(training_features)
            training_loss_1 = criterion(training_output_1, training_targets)
            training_loss_2 = criterion(training_output_2, training_targets)
            training_loss = training_loss_1 + training_loss_2
        else:
            training_output = model(training_features)
            training_loss = criterion(training_output, training_targets)

        training_loss.backward()
        optimizer.step()

        training_running_loss += training_loss.clone().cpu().item()
        training_batches += 1

    lr_scheduler.step()

    train_loss = training_running_loss / training_batches
    if args.use_tensorboard:
        writer.add_scalar('Training Loss', scalar_value=train_loss, global_step=cur_epoch)

    if cur_epoch % 100 == 0 or cur_epoch == args.epochs - 1:
        print(f'Epoch {cur_epoch + 1} | Training Loss {train_loss:.1e}', end=' | ')


def validate(dataloader, model, criterion, cur_epoch):
    model.eval()

    validation_running_loss = 0.
    validation_batches = 0

    val_gt = []
    val_pr = []

    with torch.no_grad():
        for val_idx, (validation_features, validation_targets) in enumerate(dataloader):
            validation_features = validation_features.to(args.device)
            validation_targets = validation_targets.to(args.device)

            if args.loss_type == 'double':
                _, validation_output = model(validation_features)
            else:
                validation_output = model(validation_features)

            validation_loss = criterion(validation_output, validation_targets)

            validation_targets_np = validation_targets.cpu().numpy()
            validation_output_np = validation_output.cpu().numpy()

            val_gt.extend(validation_targets_np)
            val_pr.extend(validation_output_np)

            validation_running_loss += validation_loss.clone().cpu().item()
            validation_batches += 1

        val_loss = validation_running_loss / validation_batches

    val_targets_np = np.asarray(val_gt)
    val_output_np = np.asarray(val_pr)
    if args.method != 'baseline':
        val_inv_targets = val_dataset.inv_gt_function(val_targets_np)
        val_inv_preds = val_dataset.inv_gt_function(val_output_np)
        val_mape = mean_absolute_percentage_error(val_inv_targets, val_inv_preds)
    else:
        val_mape = mean_absolute_percentage_error(val_targets_np, val_output_np)

    if cur_epoch % 100 == 0 or cur_epoch == args.epochs - 1:
        print(f'Validation Loss {val_loss:.1e} | Validation MAPE {100 * val_mape:.2f}%')


def test(dataloader, model, train_start_time, cur_epoch):
    test_gt = []
    test_pr = []

    with torch.no_grad():
        for test_idx, (test_features, test_targets) in enumerate(dataloader):
            test_features = test_features.to(args.device)
            test_targets = test_targets.to(args.device)

            if args.loss_type == 'double':
                _, test_output = model(test_features)
            else:
                test_output = model(test_features)

            test_gt.extend(test_targets.cpu().numpy())
            test_pr.extend(test_output.cpu().numpy())

    test_targets_np = np.asarray(test_gt)
    test_output_np = np.asarray(test_pr)

    if args.method != 'baseline':
        test_inv_targets = test_dataset.inv_gt_function(test_targets_np)
        test_inv_preds = test_dataset.inv_gt_function(test_output_np)
        test_mape = mean_absolute_percentage_error(test_inv_targets, test_inv_preds)
    else:
        test_mape = mean_absolute_percentage_error(test_targets_np, test_output_np)

    # Plotting a ground-truth/prediction graph for 5 selected test samples (always the same)
    if (epoch % 500 == 0 or epoch == args.epochs - 1) and args.plot_predictions:
        if args.method != 'baseline':
            plot_gt_pred_graph(args.dataset, epoch, test_inv_preds, test_inv_targets, 5, method=args.method,
                               loss_type=args.loss_type)
        else:
            plot_gt_pred_graph(args.dataset, epoch, test_targets_np, test_output_np, 5, method=args.method,
                               loss_type=args.loss_type)

    if cur_epoch % 100 == 0 or epoch == args.epochs - 1:
        print(f'Test MAPE: {100 * test_mape:.2f}% - Time elapsed: {time() - train_start_time:.0f}s')

    if args.use_tensorboard:
        writer.add_scalar('Test MAPE', scalar_value=test_mape, global_step=cur_epoch)


start = time()
for epoch in range(args.epochs):
    train(train_dl, model, criterion, optimizer, sch, epoch)
    validate(val_dl, model, criterion, epoch)
    test(test_dl, model, start, epoch)
