import os

from matplotlib import pyplot as plt


def plot_gt_pred_graph(dataset_name, epoch, predictions, ground_truth, how_many, method, loss_type):
    for i in range(how_many):
        xs = [f'0{h}:00' if h < 10 else f'{h}:00' for h in range(24)]
        plt.plot(xs, predictions[i], color='red', label='Prediction')
        plt.plot(xs, ground_truth[i], color='black', label='Ground Truth')
        plt.legend()
        plt.xlabel('Hour')
        plt.ylabel('Consumption')
        plt.xticks(rotation=90)
        plt.subplots_adjust(bottom=0.2)
        plots_dir = f'./comp_graphs/{method}/{loss_type}/{dataset_name}/epoch{epoch}/'
        if not os.path.exists(plots_dir):
            os.makedirs(plots_dir)
        plt.savefig(plots_dir + str(i))
        plt.clf()
