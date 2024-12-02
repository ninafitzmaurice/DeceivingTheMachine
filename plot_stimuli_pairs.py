import os
import numpy as np
import matplotlib.pyplot as plt
import random
import torch

### this plots random selections of the generated data as a sanity check
### plot the matching uniform and non uniform stimuli in the data directory
### after running create_data.py with the stimuli dicts for generating
### check the stimuli are generated as desired

def load_and_plot_pairs(data_dir):
    class_dirs = [os.path.join(data_dir, d) for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    
    for class_dir in class_dirs:
        uniform_dir = os.path.join(class_dir, 'uniform')
        nonuniform_dir = os.path.join(class_dir, 'nonuniform')
        
        uniform_files = sorted(os.listdir(uniform_dir))
        print('uniform_files ', len(uniform_files))
        nonuniform_files = sorted(os.listdir(nonuniform_dir))
        print('nonuniform_files ', len(nonuniform_files))
        
        if len(uniform_files) > 0 and len(nonuniform_files) > 0:
            random_n = random.randint(0, len(uniform_files) - 1)
            random_n2 = random.randint(0, len(uniform_files) - 1)
            
            u_data = np.loadtxt(os.path.join(uniform_dir, uniform_files[random_n]))
            n_data = np.loadtxt(os.path.join(nonuniform_dir, nonuniform_files[random_n]))
            u_data2 = np.loadtxt(os.path.join(uniform_dir, uniform_files[random_n2]))
            n_data2 = np.loadtxt(os.path.join(nonuniform_dir, nonuniform_files[random_n2]))
            
            u_tensor = torch.from_numpy(u_data).unsqueeze(0)
            n_tensor = torch.from_numpy(n_data).unsqueeze(0)
            u_tensor2 = torch.from_numpy(u_data2).unsqueeze(0)
            n_tensor2 = torch.from_numpy(n_data2).unsqueeze(0)
            
            fig, axs = plt.subplots(2, 2, figsize=(10, 10)) 
            axs[0, 0].imshow(u_tensor[0], cmap='gray')
            axs[0, 0].set_title(f'Uniform: {uniform_files[random_n]}')
            axs[0, 0].axis('off')
            
            axs[0, 1].imshow(n_tensor[0], cmap='gray')
            axs[0, 1].set_title(f'Nonuniform: {nonuniform_files[random_n]}')
            axs[0, 1].axis('off')

            axs[1, 0].imshow(u_tensor2[0], cmap='gray')
            axs[1, 0].set_title(f'Uniform: {uniform_files[random_n2]}')
            axs[1, 0].axis('off')
            
            axs[1, 1].imshow(n_tensor2[0], cmap='gray')
            axs[1, 1].set_title(f'Nonuniform: {nonuniform_files[random_n2]}')
            axs[1, 1].axis('off')
            
            plt.tight_layout()
            output_filename = os.path.join(f"{os.path.basename(class_dir)}_comparison.png")
            plt.savefig(output_filename)
            plt.close(fig) 

        else:
            print(f"No files in {class_dir} will skip.")


data_directory = '/home/nfitzmaurice/stim_gen/TTTEST/aa'
load_and_plot_pairs(data_directory)
