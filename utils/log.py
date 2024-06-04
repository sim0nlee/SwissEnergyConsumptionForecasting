def print_run_info(args):
    print(f'\nRunning {args.method} method with {args.model} on {args.dataset} dataset with {args.loss_type} loss')
    print(f'Device: {args.device}')

    if args.use_tensorboard:
        print(f'Plotting Test MAPEs on Tensorboard')  #: saving plots in {"tensorboard_plots/" + tensorboard_file}')

    if args.plot_predictions:
        print(f'Plotting prediction comparisons with ground truth')

    print('\n--- HYPERPARAMETERS ---\n')

    print(f'Feature scaler: {args.feature_scaler}')
    print(f'Load scaler: {args.load_scaler}')
    print(f'Batch size: {args.batch_size}')
    print(f'Epochs: {args.epochs}')
    print(f'Learning rate: {args.lr}')
    print(f'First scheduler milestone: {args.scheduler_start}')
    print(f'Scheduler step: {args.scheduler_step}')
    print(f'Learning rate decay: {args.lr_decay}')