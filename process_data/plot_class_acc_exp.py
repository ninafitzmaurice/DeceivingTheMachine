import pandas as pd
import matplotlib.pyplot as plt
import ast

######### class acc for EXPERIMENT DATA, just to test how cycles influence class predictions
######### (in my data cycles did not influence class predictions too much, thats good)
######### for uni and nonuni images

## you can check this for each epoch if you saved that data (not much reason to do so)

# string lists into actual lists becase data saving is messsyyyyy :c
def process_list_column(column):
    processed_column = []
    for item in column:
        try:
            processed_column.append(ast.literal_eval(item)) 
        except (ValueError, SyntaxError):
            processed_column.append([item]) 
    return processed_column

def calculate_accuracy(targets, predictions):
    correct = sum([t == p for t, p in zip(targets, predictions)])
    return correct / len(targets) if len(targets) > 0 else 0

# load experiment data for adversarial (nonuniform) and clean (uniform) images
def load_experiment_data():
    exp_adv_data = pd.read_csv('1cyc_15epoch_exp_all_epoch_res0.01/exp_data/exp_adv_data.csv')
    exp_clean_data = pd.read_csv('1cyc_15epoch_exp_all_epoch_res0.01/exp_data/exp_clean_data.csv')

    # list-like columns to actual lists
    exp_adv_data['class targets'] = process_list_column(exp_adv_data['class targets'])
    exp_adv_data['class predictions'] = process_list_column(exp_adv_data['class predictions'])
    exp_adv_data['nonuni predictions'] = process_list_column(exp_adv_data['nonuni predictions'])

    exp_clean_data['class targets'] = process_list_column(exp_clean_data['class targets'])
    exp_clean_data['class predictions'] = process_list_column(exp_clean_data['class predictions'])
    exp_clean_data['uni predictions'] = process_list_column(exp_clean_data['uni predictions'])

    return exp_adv_data, exp_clean_data

# plot class accuracy over feedback cycles for selected epochs for nonuniform stimuli (aka adversarial examples)
def plot_class_accuracy_over_cycles_adv(exp_adv_data, selected_epochs=None):
    epochs = sorted(exp_adv_data['epoch'].unique())
    if selected_epochs is not None:
        epochs = [epoch for epoch in epochs if epoch in selected_epochs]
    cycles = sorted(exp_adv_data['cycle'].unique())

    plt.figure()

    for epoch in epochs:
        exp_adv_epoch = exp_adv_data[exp_adv_data['epoch'] == epoch]
        class_accuracies = []

        for cycle in cycles:
            cycle_data = exp_adv_epoch[exp_adv_epoch['cycle'] == cycle]
            targets = [item for sublist in cycle_data['class targets'] for item in sublist]
            predictions = [item for sublist in cycle_data['class predictions'] for item in sublist]
            accuracy = calculate_accuracy(targets, predictions)
            class_accuracies.append(accuracy)

        plt.plot(cycles, class_accuracies, label=f'Epoch {epoch}')

    plt.xlabel('Feedback Cycles')
    plt.ylabel('Class Accuracy')
    plt.title(f'Class Accuracy Over Feedback Cycles (Adversarial Data)')
    plt.legend()
    plt.xticks(ticks=range(min(cycles), max(cycles) + 1))
    plt.savefig(f'class_accuracy_adv_selected_epochs.png')
    plt.show()

# class accuracy plots over feedback cycles for selected epochs for uniform (clean) stimuli
def plot_class_accuracy_over_cycles_clean(exp_clean_data, selected_epochs=None):
    epochs = sorted(exp_clean_data['epoch'].unique())
    if selected_epochs is not None:
        epochs = [epoch for epoch in epochs if epoch in selected_epochs]
    cycles = sorted(exp_clean_data['cycle'].unique())

    plt.figure()

    for epoch in epochs:
        exp_clean_epoch = exp_clean_data[exp_clean_data['epoch'] == epoch]
        class_accuracies = []

        for cycle in cycles:
            cycle_data = exp_clean_epoch[exp_clean_epoch['cycle'] == cycle]
            targets = [item for sublist in cycle_data['class targets'] for item in sublist]
            predictions = [item for sublist in cycle_data['class predictions'] for item in sublist]
            accuracy = calculate_accuracy(targets, predictions)
            class_accuracies.append(accuracy)

        plt.plot(cycles, class_accuracies, label=f'Epoch {epoch}')

    plt.xlabel('Feedback Cycles')
    plt.ylabel('Class Accuracy')
    plt.title(f'Class Accuracy Over Feedback Cycles (Clean Data)')
    plt.legend()
    plt.xticks(ticks=range(min(cycles), max(cycles) + 1))
    plt.savefig(f'class_accuracy_clean_selected_epochs.png')
    plt.show()

# same for uniformity accuracy plots.....
def plot_uniformity_accuracy_over_cycles_adv(exp_adv_data, selected_epochs=None):
    epochs = sorted(exp_adv_data['epoch'].unique())
    if selected_epochs is not None:
        epochs = [epoch for epoch in epochs if epoch in selected_epochs]
    cycles = sorted(exp_adv_data['cycle'].unique())

    plt.figure()

    for epoch in epochs:
        exp_adv_epoch = exp_adv_data[exp_adv_data['epoch'] == epoch]
        uniformity_accuracies = []

        for cycle in cycles:
            cycle_data = exp_adv_epoch[exp_adv_epoch['cycle'] == cycle]
            nonuni_targets = [item for sublist in cycle_data['nonuni targets'] for item in sublist]
            nonuni_predictions = [item for sublist in cycle_data['nonuni predictions'] for item in sublist]
            accuracy = calculate_accuracy(nonuni_targets, nonuni_predictions)
            uniformity_accuracies.append(accuracy)

        plt.plot(cycles, uniformity_accuracies, label=f'Epoch {epoch}')

    plt.xlabel('Feedback Cycles')
    plt.ylabel('Uniformity Accuracy')
    plt.title(f'Uniformity Accuracy Over Feedback Cycles (Adversarial Data)')
    plt.legend()
    plt.xticks(ticks=range(min(cycles), max(cycles) + 1))
    plt.savefig(f'uniformity_accuracy_adv_selected_epochs.png')
    plt.show()

if __name__ == "__main__":
    exp_adv_data, exp_clean_data = load_experiment_data()

    plot_class_accuracy_over_cycles_adv(exp_adv_data, selected_epochs=[15])
    plot_class_accuracy_over_cycles_clean(exp_clean_data, selected_epochs=[15])

