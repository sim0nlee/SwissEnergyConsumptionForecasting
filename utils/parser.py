import argparse

import data

parser = argparse.ArgumentParser(prog='python forecast.py',
                                 description='Run forecasting on the given dataset using the chosen method and model hyperparameters')
parser.add_argument('--method', choices=['baseline', 'anio'], default='baseline')
parser.add_argument('--loss_type', choices=['single', 'double'], default='single')
parser.add_argument('--anchoring', choices=['week', 'month'], default='week')
parser.add_argument('--model', choices=['mlp', 'resnet', 'resnet_ext', 'lstm', 'lstm_ext'], default='mlp')
parser.add_argument('--dataset', choices=data.dataset_names, default='swiss-consumption')
parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
parser.add_argument('--use_tensorboard', action='store_true')
parser.add_argument('--plot_predictions', action='store_true')
parser.add_argument('--feature_scaler', choices=['max', 'standard', 'minmax', ''], default='')
parser.add_argument('--load_scaler', choices=['max', 'standard', 'minmax', ''], default='')
parser.add_argument('--epochs', type=int, default=2000)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--scheduler_start', type=int, default=500)
parser.add_argument('--scheduler_step', type=int, default=500)
parser.add_argument('--lr_decay', type=float, default=0.5)
