import pandas as pd
import matplotlib.pyplot as plt
import ast
import numpy as np


####### this is for the experiment ONLY FOR THE train_UI_adv experimental protocol! 

#### it uses pixel noise in periphery as adversarial examples
#### not training to reconstruct non uniform to uniform, trained to reconstruct noise!!

data = pd.read_csv('/home/nfitzmaurice/cnnf_UI/ADV_protocol_3/exp_data/exp_data.csv', converters={
    'class targets': ast.literal_eval,
    'clean class predictions': ast.literal_eval,
    'adv class predictions': ast.literal_eval,
    'uni targets': ast.literal_eval,
    'clean uni predictions': ast.literal_eval,
    'adv uni predictions': ast.literal_eval,
    'clean uni logits': ast.literal_eval,
    'adv uni logits': ast.literal_eval
})

def extract_first_value(x):
    if isinstance(x, list) and len(x) > 0:
        return x[0]
    else:
        return pd.NA

columns_to_extract = [
    'clean class predictions',
    'adv class predictions',
    'uni targets',
    'clean uni predictions',
    'adv uni predictions',
    'clean uni logits',
    'adv uni logits'
]

for col in columns_to_extract:
    data[col] = data[col].apply(extract_first_value)

data['epoch'] = data['epoch'].astype('Int64')
data['batch index'] = data['batch index'].astype('Int64')
data['cycle'] = data['cycle'].astype('Int64')
data['uni targets'] = data['uni targets'].astype(float)
data['class targets'] = data['class targets'].astype('Int64')
data['clean class predictions'] = data['clean class predictions'].astype('Int64')
data['adv class predictions'] = data['adv class predictions'].astype('Int64')
data['clean uni predictions'] = data['clean uni predictions'].astype(float)
data['adv uni predictions'] = data['adv uni predictions'].astype(float)
data['clean uni logits'] = data['clean uni logits'].astype(float)
data['adv uni logits'] = data['adv uni logits'].astype(float)

#  data for 'big_diff' and 'small_diff'
big_diff_data = data[data['data folder'] == 'big_diff']
small_diff_data = data[data['data folder'] == 'small_diff']

# return cycles and logits
def process_data(data_subset, is_uniform=True):
    target_value = 1.0 if is_uniform else 0.0
    data_subset = data_subset[data_subset['uni targets'] == target_value]

    cycles = sorted(data_subset['cycle'].dropna().unique())
    cycles = [int(cycle) for cycle in cycles]

    clean_logits_per_cycle = []
    adv_logits_per_cycle = []

    for cycle in cycles:
        cycle_data = data_subset[data_subset['cycle'] == cycle]

        # Use 'clean uni logits' and 'adv uni logits' for all cycles
        clean_logits = cycle_data['clean uni logits'].dropna().mean()
        adv_logits = cycle_data['adv uni logits'].dropna().mean()

        clean_logits_per_cycle.append(clean_logits)
        adv_logits_per_cycle.append(adv_logits)

    # cycles with NaN logits
    filtered_cycles = []
    filtered_clean_logits = []
    filtered_adv_logits = []

    for i, cycle in enumerate(cycles):
        if pd.notna(clean_logits_per_cycle[i]) and pd.notna(adv_logits_per_cycle[i]):
            filtered_cycles.append(cycle)
            filtered_clean_logits.append(clean_logits_per_cycle[i])
            filtered_adv_logits.append(adv_logits_per_cycle[i])

    return filtered_cycles, filtered_clean_logits, filtered_adv_logits

# for uniform images
big_cycles_uniform, big_clean_logits_uniform, big_adv_logits_uniform = process_data(big_diff_data, is_uniform=True)
small_cycles_uniform, small_clean_logits_uniform, small_adv_logits_uniform = process_data(small_diff_data, is_uniform=True)

# for non-uniform images
big_cycles_nonuniform, big_clean_logits_nonuniform, big_adv_logits_nonuniform = process_data(big_diff_data, is_uniform=False)
small_cycles_nonuniform, small_clean_logits_nonuniform, small_adv_logits_nonuniform = process_data(small_diff_data, is_uniform=False)

plt.figure(figsize=(10, 6))

##### UNIFORM IMAGES CLEAN AND ADV
# plot clean logits for 'big_diff' and 'small_diff' (uniform images)
plt.plot(big_cycles_uniform, big_clean_logits_uniform, label='Big Diff Clean Logits (Uniform Images)', marker='o')
plt.plot(small_cycles_uniform, small_clean_logits_uniform, label='Small Diff Clean Logits (Uniform Images)', marker='o')

# adversarial logits for 'big_diff' and 'small_diff' (uniform images)
plt.plot(big_cycles_uniform, big_adv_logits_uniform, label='Big Diff Adv Logits (Uniform Images)', marker='x')
plt.plot(small_cycles_uniform, small_adv_logits_uniform, label='Small Diff Adv Logits (Uniform Images)', marker='x')

plt.title('Uniformity Logits Across Cycles for Uniform Images\nClean and Adversarial Examples')
plt.xlabel('Cycle')
plt.ylabel('Logits')
plt.legend()
plt.grid(True)

plt.ylim(0, 1)
plt.yticks(np.arange(0, 1.1, 0.1))

all_cycles_uniform = sorted(set(big_cycles_uniform + small_cycles_uniform))
plt.xticks(all_cycles_uniform)

plt.savefig('uniformity_logits_uniform_images.png')
plt.close()


##### NON UNIFORM IMAGES CLEAN AND ADV
plt.figure(figsize=(10, 6))

# clean logits for 'big_diff' and 'small_diff' (non-uniform images)
plt.plot(big_cycles_nonuniform, big_clean_logits_nonuniform, label='Big Diff Clean Logits (Non-Uniform Images)', marker='o')
plt.plot(small_cycles_nonuniform, small_clean_logits_nonuniform, label='Small Diff Clean Logits (Non-Uniform Images)', marker='o')

# adversarial logits for 'big_diff' and 'small_diff' (non-uniform images)
plt.plot(big_cycles_nonuniform, big_adv_logits_nonuniform, label='Big Diff Adv Logits (Non-Uniform Images)', marker='x')
plt.plot(small_cycles_nonuniform, small_adv_logits_nonuniform, label='Small Diff Adv Logits (Non-Uniform Images)', marker='x')

plt.title('Uniformity Logits Across Cycles for Non-Uniform Images\nClean and Adversarial Examples')
plt.xlabel('Cycle')
plt.ylabel('Logits')
plt.legend()
plt.grid(True)

plt.ylim(0, 1)
plt.yticks(np.arange(0, 1.1, 0.1))

all_cycles_nonuniform = sorted(set(big_cycles_nonuniform + small_cycles_nonuniform))
plt.xticks(all_cycles_nonuniform)

plt.savefig('uniformity_logits_nonuniform_images.png')
plt.close()
