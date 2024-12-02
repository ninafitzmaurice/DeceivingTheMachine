from __future__ import print_function
import torch
import os
import numpy as np
# from tensorboardX import SummaryWriter
import skimage as sk
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import Sampler
import os
import numpy as np

def save_tensors_npy(uniform_image, nonuniform_image, ff_current, ff_prev, 
                                  recon_clean, recon_adv, recon_block1_clean, recon_block1_adv, 
                                  recon_block2_clean, recon_block2_adv, recon_block3_clean, recon_block3_adv, 
                                  epoch, i_cycle, args):
    
    epoch_dir = f"{args.model_dir}/recon_plots/epoch_{epoch+1}"
    cycle_dir = os.path.join(epoch_dir, f"cycle_{i_cycle}")
    clean_dir = os.path.join(cycle_dir, "clean")
    adv_dir = os.path.join(cycle_dir, "adv")
    os.makedirs(clean_dir, exist_ok=True)
    os.makedirs(adv_dir, exist_ok=True)

    np.save(os.path.join(cycle_dir, "uni_img.npy"), uniform_image[0].mean(dim=0).cpu().numpy())
    np.save(os.path.join(cycle_dir, "nonuni_img.npy"), nonuniform_image[0].mean(dim=0).cpu().numpy())

    np.save(os.path.join(cycle_dir, "ff_current.npy"), ff_current[0].mean(dim=0).cpu().numpy())
    np.save(os.path.join(cycle_dir, "ff_prev.npy"), ff_prev[0].mean(dim=0).cpu().numpy())

    np.save(os.path.join(clean_dir, "final_clean.npy"), recon_clean[0].mean(dim=0).cpu().numpy())
    np.save(os.path.join(adv_dir, "final_adv.npy"), recon_adv[0].mean(dim=0).cpu().numpy())

    np.save(os.path.join(clean_dir, "b1_clean.npy"), recon_block1_clean[0].mean(dim=0).cpu().numpy())
    np.save(os.path.join(adv_dir, "b1_adv.npy"), recon_block1_adv[0].mean(dim=0).cpu().numpy())

    np.save(os.path.join(clean_dir, "b2_clean.npy"), recon_block2_clean[0].mean(dim=0).cpu().numpy())
    np.save(os.path.join(adv_dir, "b2_adv.npy"), recon_block2_adv[0].mean(dim=0).cpu().numpy())

    np.save(os.path.join(clean_dir, "b3_clean.npy"), recon_block3_clean[0].mean(dim=0).cpu().numpy())
    np.save(os.path.join(adv_dir, "b3_adv.npy"), recon_block3_adv[0].mean(dim=0).cpu().numpy())


def pixel_correlation(uniform_image_np, nonuniform_image_np, input_image_np):
    """
    Compute combined pixel correlation between the input image and the uniform and nonuniform images.
    Positive correlation = input is more similar to uniform image
    Negative correlation = nput is more similar to nonuniform image

    Args:
        uniform_image_np (np.ndarray): Numpy array of the uniform image.
        nonuniform_image_np (np.ndarray): Numpy array of the nonuniform image.
        input_image_np (np.ndarray): Numpy array of the input image.
    """
    # flatten
    uniform_image_flat = uniform_image_np.flatten()
    nonuniform_image_flat = nonuniform_image_np.flatten()
    input_image_flat = input_image_np.flatten()

    input_uniform_corr = np.corrcoef(input_image_flat, uniform_image_flat)[0, 1]
    input_nonuniform_corr = np.corrcoef(input_image_flat, nonuniform_image_flat)[0, 1]

    combined_corr = input_uniform_corr - input_nonuniform_corr

    return combined_corr


def tensor_to_list(value):
    if isinstance(value, torch.Tensor):
        return value.cpu().numpy().tolist()
    return value


def calculate_accuracy(targets, predictions):
    correct = sum(t == p for t, p in zip(targets, predictions))
    accuracy = correct / len(targets)
    return accuracy


def flatten_list(nested_list):
    return [item for sublist in nested_list for item in sublist]


def norm_t(tensor):
    tensor = tensor.cpu() 
    tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min())
    return tensor


def get_lr(step, total_steps, lr_max, lr_min):
    """Compute learning rate according to cosine annealing schedule."""
    return lr_min + (lr_max - lr_min) * 0.5 * (1 +
                                             np.cos(step / total_steps * np.pi))


def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter) ** (power))


def seed_torch(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.


class RandomSampler(Sampler):
    '''
    Custom sampler to randomly sample a specified number of images from the dataset,
    ensuring a balance between uniform and non-uniform images.

    Attributes:
        data_source (Dataset): The dataset to sample from.
        num_samples (int): The number of samples to draw in each iteration.

    Only for the adversarial training code. 
    '''

    def __init__(self, data_source, num_samples=40):
        self.data_source = data_source
        self.num_samples = num_samples

        self.uniform_indices = [i for i, (_, _, is_uniform) in enumerate(data_source.samples) if is_uniform == 1]
        self.nonuniform_indices = [i for i, (_, _, is_uniform) in enumerate(data_source.samples) if is_uniform == 0]

    def __iter__(self):
        num_uniform_samples = min(len(self.uniform_indices), max(1, self.num_samples // 2))
        num_nonuniform_samples = min(len(self.nonuniform_indices), max(1, self.num_samples - num_uniform_samples))

        sampled_uniform_indices = random.sample(self.uniform_indices, num_uniform_samples)
        sampled_nonuniform_indices = random.sample(self.nonuniform_indices, num_nonuniform_samples)

        sampled_indices = sampled_uniform_indices + sampled_nonuniform_indices
        random.shuffle(sampled_indices)

        return iter(sampled_indices)

    def __len__(self):
        return self.num_samples


# def save_train_plots(train_performance, save_dir):
#     # TRAIN
#     fig, ax = plt.subplots(figsize=(10, 6)) 
#     # Plot 1: Training Accuracies (Class Acc and Total Uniformity Acc)
#     ax.plot(train_performance['Epoch'], train_performance['Class Acc'], label='Class Accuracy', color='C0', alpha=0.8)
#     ax.plot(train_performance['Epoch'], train_performance['Total Uniformity Acc'], label='Total Uniformity Accuracy', color='C1', alpha=0.8)
#     ax.plot(train_performance['Epoch'], train_performance['Uniform Acc'], label='Uniform Accuracy', color='C2', linestyle='--', alpha=0.8)
#     ax.plot(train_performance['Epoch'], train_performance['Nonuniform Acc'], label='Nonuniform Accuracy', color='C3', linestyle='--', alpha=0.8)
#     ax.set_title('Training Accuracies')
#     ax.set_xlabel('Epoch')
#     ax.set_ylabel('Accuracy')
#     ax.legend()

#     plt.tight_layout()
#     plt.savefig(f'{save_dir}/training_accuracies_plot.png')
#     plt.close()

#     # Plot 2: Training Losses (Class Loss and Total Uniformity Loss)
#     fig, axs = plt.subplots(2, 1, figsize=(10, 12))
#     #  Class Loss
#     axs[0].plot(train_performance['Epoch'], train_performance['Class Loss'], label='Class Loss', color='C0')
#     axs[0].set_title('Training Class Loss')
#     axs[0].set_xlabel('Epoch')
#     axs[0].set_ylabel('Loss')
#     axs[0].legend()

#     # Total Uniformity Loss
#     axs[1].plot(train_performance['Epoch'], train_performance['Total Uniformity Loss'], label='Total Uniformity Loss', color='C1')
#     axs[1].set_title('Training Total Uniformity Loss')
#     axs[1].set_xlabel('Epoch')
#     axs[1].set_ylabel('Loss')
#     axs[1].legend()

#     plt.tight_layout()
#     plt.savefig(f'{save_dir}/training_losses_plot.png')
#     plt.close()


# def save_test_plots(test_performance, save_dir):
#     # TEST
#     # Plot 1: Test Accuracies
#     fig, ax = plt.subplots(figsize=(10, 6))
#     ax.plot(test_performance['Epoch'], test_performance['Class Acc: Uniform Images'], label='Class Acc: Uniform Images', color='C0', alpha=0.8)
#     ax.plot(test_performance['Epoch'], test_performance['Class Acc: Nonuniform Images'], label='Class Acc: Nonuniform Images', color='C1', alpha=0.8)
#     ax.plot(test_performance['Epoch'], test_performance['Uniform Acc'], label='Uniform Acc', color='C2', linestyle='--', alpha=0.8)
#     ax.plot(test_performance['Epoch'], test_performance['Nonuniform Acc'], label='Nonuniform Acc', color='C3', linestyle='--', alpha=0.8)
#     ax.set_title('Test Accuracies')
#     ax.set_xlabel('Epoch')
#     ax.set_ylabel('Accuracy')
#     ax.legend()
#     plt.tight_layout()
#     plt.savefig(f'{save_dir}/test_accuracies_plot.png')
#     plt.close()

#     # Plot 2: Test Losses (Class Loss and Uniformity Loss)
#     fig, axs = plt.subplots(2, 1, figsize=(10, 12))
#     # Class Loss
#     axs[0].plot(test_performance['Epoch'], test_performance['Class Loss: Uniform Images'], label='Class Loss: Uniform Images', color='C0', alpha=0.8)
#     axs[0].plot(test_performance['Epoch'], test_performance['Class Loss: Nonuniform Images'], label='Class Loss: Nonuniform Images', color='C1', alpha=0.8)
#     axs[0].set_title('Test Class Loss for Uniform and Nonuniform Images')
#     axs[0].set_xlabel('Epoch')
#     axs[0].set_ylabel('Loss')
#     axs[0].legend()

#     # Uniform and Nonuniform Loss
#     axs[1].plot(test_performance['Epoch'], test_performance['Uniform Loss'], label='Uniform Loss', color='C2', alpha=0.8)
#     axs[1].plot(test_performance['Epoch'], test_performance['Nonniform Loss'], label='Nonuniform Loss', color='C3', alpha=0.8)  # Correct typo if necessary
#     axs[1].set_title('Test Uniform and Nonuniform Loss')
#     axs[1].set_xlabel('Epoch')
#     axs[1].set_ylabel('Loss')
#     axs[1].legend()

#     plt.tight_layout()
#     plt.savefig(f'{save_dir}/test_losses_plot.png')
#     plt.close()