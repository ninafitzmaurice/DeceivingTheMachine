import pandas as pd
import matplotlib.pyplot as plt
import ast


###### THIS IS THE MAIN EXPERIMENT! THIS PLOTS THE LOGITS FOR THE UI TASK OVER CYCLES!!

### if you want to do this for every epoch durining training, then you need to go to the training code
### and make sure it runs the exp for all epochs

def process_list_column(column):
    processed_column = []
    for item in column:
        try:
            processed_column.append(ast.literal_eval(item)) 
        except (ValueError, SyntaxError):
            processed_column.append([item]) 
    return processed_column

# experimental data for adversarial (nonuniform) and clean (uniform) images
def load_experiment_data():
    exp_adv_data = pd.read_csv('1cyc_15epoch_exp_all_epoch_res0.01/exp_data/exp_adv_data.csv')
    exp_clean_data = pd.read_csv('1cyc_15epoch_exp_all_epoch_res0.01/exp_data/exp_clean_data.csv')

    exp_adv_data['nonuni logits'] = process_list_column(exp_adv_data['nonuni logits'])
    exp_adv_data['nonuni cycle logits'] = process_list_column(exp_adv_data['nonuni cycle logits'])

    exp_clean_data['uni logits'] = process_list_column(exp_clean_data['uni logits'])
    exp_clean_data['uni cycle logits'] = process_list_column(exp_clean_data['uni cycle logits'])

    return exp_adv_data, exp_clean_data

# plot logits over feedback cycles for both uniform (clean) and nonuniform (adversarial) data for selected epochs
def plot_logits_over_cycles(exp_adv_data, exp_clean_data, selected_epochs):
    cycles = sorted(exp_adv_data['cycle'].unique()) 

    plt.figure()

    for epoch in selected_epochs:
        # adversarial (non-uniform)
        exp_adv_epoch = exp_adv_data[exp_adv_data['epoch'] == epoch]
        logits_adv_over_cycles = []

        for cycle in cycles:
            cycle_data = exp_adv_epoch[exp_adv_epoch['cycle'] == cycle]
            
            if cycle == 0:
                logits = [item for sublist in cycle_data['nonuni logits'] for item in sublist]
            else:
                logits = [item for sublist in cycle_data['nonuni cycle logits'] for item in sublist]
                
            avg_logits = sum(logits) / len(logits) if len(logits) > 0 else 0
            logits_adv_over_cycles.append(avg_logits)

        # clean (uniform) data
        exp_clean_epoch = exp_clean_data[exp_clean_data['epoch'] == epoch]
        logits_clean_over_cycles = []

        for cycle in cycles:
            cycle_data = exp_clean_epoch[exp_clean_epoch['cycle'] == cycle]
            
            if cycle == 0:
                logits = [item for sublist in cycle_data['uni logits'] for item in sublist]
            else:
                logits = [item for sublist in cycle_data['uni cycle logits'] for item in sublist]
            
            avg_logits = sum(logits) / len(logits) if len(logits) > 0 else 0
            logits_clean_over_cycles.append(avg_logits)

        # plot clean and adv logits for the current epoch
        plt.plot(cycles, logits_adv_over_cycles, label=f'Epoch {epoch} - Non-Uniform')
        plt.plot(cycles, logits_clean_over_cycles, label=f'Epoch {epoch} - Uniform')

    plt.xlabel('Feedback Cycles')
    plt.ylabel('Average Logits')
    plt.title('Logits Over Feedback Cycles (Uniform and Non-Uniform)')
    plt.xticks(ticks=range(min(cycles), max(cycles) + 1)) 
    plt.legend()
    plt.savefig(f'logits_uniform_nonuniform_selected_epochs.png')
    plt.show()

if __name__ == "__main__":
    exp_adv_data, exp_clean_data = load_experiment_data()

    # selected epochs to plt (shows how learning to reconstruct over epochs)
    epochs = [15] 

    plot_logits_over_cycles(exp_adv_data, exp_clean_data, epochs)
