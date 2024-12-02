import os
import numpy as np
import matplotlib.pyplot as plt

def get_recons(base_dir, save_path):
    cycle_images = []
    cycle_titles = []

    for cycle_folder in sorted(os.listdir(base_dir)):
        cycle_path = os.path.join(base_dir, cycle_folder)
        
        if os.path.isdir(cycle_path):
            print(f"processingggg {cycle_folder}")
            
            first_batch_folder = sorted(os.listdir(cycle_path))[0]
            batch_path = os.path.join(cycle_path, first_batch_folder)
            
            if os.path.isdir(batch_path):
                first_img_folder = sorted(os.listdir(batch_path))[0]
                img_path = os.path.join(batch_path, first_img_folder)

                if os.path.isdir(img_path):
                    adv_input_path = os.path.join(img_path, 'adv_input.npy')
                    ff_current_path = os.path.join(img_path, 'recon.npy')

                    if os.path.exists(adv_input_path) and os.path.exists(ff_current_path):
                        adv_input = np.load(adv_input_path)
                        ff_current = np.load(ff_current_path)

                        if adv_input.shape[0] == 1:
                            adv_input = np.squeeze(adv_input)
                        
                        # !!!!!!!average across channels for ff_current if it's 3D
                        if len(ff_current.shape) == 3:
                            ff_current = np.mean(ff_current, axis=-1)

                        cycle_images.append((adv_input, ff_current))
                        cycle_titles.append(cycle_folder)

    num_cycles = len(cycle_images)
    plt.figure(figsize=(10, 5 * num_cycles))

    for i, (adv_input, ff_current) in enumerate(cycle_images):
        plt.subplot(num_cycles, 2, 2 * i + 1)
        plt.imshow(adv_input, cmap='gray')
        plt.title(f'{cycle_titles[i]}: adv_input')
        plt.axis('off')

        plt.subplot(num_cycles, 2, 2 * i + 2)
        plt.imshow(ff_current, cmap='gray')
        plt.title(f'{cycle_titles[i]}: ff_current')
        plt.axis('off')

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"saved to {save_path}")
    plt.show()

# directory for the cycle folders
base_dir = "1cyc_15epoch_train_exp_res0.01/exp_recon_arrays_big_diff"
# output path
save_path = "/home/nfitzmaurice/cnnf_UI/cycle_images_plot.png"

get_recons(base_dir, save_path)