import sys
import os
import numpy as np
import random as std_random
import argparse
from tqdm import tqdm
from stimuligenerator_UI import UIStimuliGenerator

### generate the data for each class in stimuli dictionary 
### automatically makes pairs of uniform and nonuniform images
### where the centre of the nonuniform image makes up the whole uniform image 
### sorry its kinda a mess.... 

default_stimuli_dictionary = None
default_output_path = None

def process_data(i, data_dict, data_dir, generator_instance, file_count_dict):
    stimulus_params = data_dict[i]
    data = generator_instance.create_UI_stim(stimulus_params, 128)
    stimulus, label = data

    class_name, uniformity = label.split('_')

    # class and uniformity subfolders
    class_dir = os.path.join(data_dir, class_name)
    os.makedirs(class_dir, exist_ok=True)

    uniformity_dir = os.path.join(class_dir, uniformity)
    os.makedirs(uniformity_dir, exist_ok=True)

    # unique key for each class and uniformity combination
    key = f"{class_name}_{uniformity}"

    if key not in file_count_dict:
        file_count_dict[key] = 0
    else:
        # reset the index if a new stimulus type is encountered
        if i > 0:
            # previous data entry exists and has a label
            previous_label = data_dict[i-1].get('label')
            if previous_label and previous_label.split('_')[0] != class_name:
                file_count, file_count_dict[key] = 0

    file_index = file_count_dict[key]
    file_count_dict[key] += 1

    file_path = os.path.join(uniformity_dir, f"{label}{file_index+1}.npy")

    # unique file name by adding zeros if file already exists
    base_name, ext = os.path.splitext(file_path)
    counter = 0
    while os.path.exists(file_path):
        counter += 1
        file_path = f"{base_name}{str(counter).zfill(3)}{ext}"

    np.savetxt(file_path, stimulus)  # use np.save to write the numpy array to a .npy file

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--wd_path', type=str, default='stim_gen', help='Path to working dir')
    parser.add_argument('--stim_dict_file', type=str, default=default_stimuli_dictionary, help='Name of dict to generate data')
    parser.add_argument('--output_path', type=str, default=default_output_path, help='Path to output folder')
    parser.add_argument('--output_folder', type=str, required=True, help='Name of the folder to store output data')

    file_args = parser.parse_args()

    # UIStimuliGenerator instance
    generator_instance = UIStimuliGenerator()

    namespace = {}
    input_file_path = os.path.join(file_args.wd_path, file_args.stim_dict_file)
    output_file_path = os.path.join(file_args.output_path, file_args.output_folder)
    
    with open(input_file_path, 'r') as file:
        exec(file.read(), {}, namespace)
    shared_params_dict = namespace['shared_params_dict']
    features_dict = namespace['features_dict']

    # dict for each data entry
    data_dict = generator_instance.generate_combinations(features_dict, shared_params_dict)
    print('total stimuli: ', len(data_dict))

    # out folder
    os.makedirs(output_file_path, exist_ok=True)

    file_count_dict = {}

    for i in tqdm(range(len(data_dict))):
        process_data(i, data_dict, output_file_path, generator_instance, file_count_dict)

if __name__ == '__main__':
    main()


##########
### ??????
#### no idea what i did here....
# '''
# folders for each class, saves data of that class to folder
# '''

# import sys
# import os
# import numpy as np
# # from tqdm import tqdm
# import random as std_random
# import multiprocessing
# import argparse
# from tqdm import tqdm

# from stimuligenerator_UI import UIStimuliGenerator


# def iter(args):
#     i, data_dict, data_dir, generator_instance = args

#     stimulus_params = data_dict[i]
#     data = generator_instance.create_UI_stim(stimulus_params, 128)
#     stimulus, label = data

#     # get class and uniformity from the label
#     class_name, uniformity = label.split('_')

#     # class and uniformity subfolders
#     class_dir = os.path.join(data_dir, class_name)
#     os.makedirs(class_dir, exist_ok=True)

#     uniformity_dir = os.path.join(class_dir, uniformity)
#     os.makedirs(uniformity_dir, exist_ok=True)

#     # save the stimulus data in the uniformity subfolder
#     file_path = os.path.join(uniformity_dir, f"{label}{i}.npy")
    
#     if not os.path.exists(file_path):
#         np.savetxt(file_path, stimulus)


# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--wd_path', type=str, default='stim_gen', 
#                         help='Path to working dir')
    
#     parser.add_argument('--stim_dict_file', type=str, default = '/home/nfitzmaurice/stim_gen/stimuli_params_dict.txt',
#                         help='Name of dict to generate data')
    
#     parser.add_argument('--output_path', type=str, default='/home/nfitzmaurice/cnnf_UI/data', 
#                         help='Path to output folder')
    
#     parser.add_argument('--output_folder', type=str, 
#                         required=True, help='Name of the folder to store output data')

#     file_args = parser.parse_args()

#     # UIStimuliGenerator instance
#     generator_instance = UIStimuliGenerator()

#     namespace = {}
#     input_file_path = os.path.join(file_args.wd_path, file_args.stim_dict_file)
#     output_file_path = os.path.join(file_args.output_path, file_args.output_folder)
    
#     with open(input_file_path, 'r') as file:
#         exec(file.read(), {}, namespace)
#     shared_params_dict = namespace['shared_params_dict']
#     features_dict = namespace['features_dict']

#     # dict for each data entry
#     std_random.seed(666)
#     data_dict = generator_instance.generate_combinations(features_dict, shared_params_dict)
#     print('total stimuli: ', len(data_dict))

#     # output folder
#     os.makedirs(output_file_path, exist_ok=True)

#     # pass the generator_instance to the iter function
#     with multiprocessing.Pool(100) as pool:
#         args = [(i, data_dict, output_file_path, generator_instance) for i in range(len(data_dict))]
#         results = list(tqdm(pool.imap(iter, args), total=len(data_dict)))
#         # results = pool.map(iter, args)

# if __name__ == '__main__':
#     main()

# import sys
# import os
# import numpy as np
# import random as std_random
# import multiprocessing
# import argparse
# from tqdm import tqdm
# from stimuligenerator_UI import UIStimuliGenerator

# def iter(args, file_count_dict, lock):
#     i, data_dict, data_dir, generator_instance = args

#     stimulus_params = data_dict[i]
#     print(stimulus_params)
#     data = generator_instance.create_UI_stim(stimulus_params, 128)
#     stimulus, label = data

#     # Get class and uniformity from the label
#     class_name, uniformity = label.split('_')

#     # Class and uniformity subfolders
#     class_dir = os.path.join(data_dir, class_name)
#     os.makedirs(class_dir, exist_ok=True)

#     uniformity_dir = os.path.join(class_dir, uniformity)
#     os.makedirs(uniformity_dir, exist_ok=True)

#     # Create a unique key for each class and uniformity combination
#     key = f"{class_name}_{uniformity}"

#     # Synchronize access to the file count dictionary
#     with lock:
#         if key not in file_count_dict:
#             file_count_dict[key] = 0
#         file_index = file_count_dict[key]
#         file_count_dict[key] += 1

#     # Save the stimulus data in the uniformity subfolder
#     file_path = os.path.join(uniformity_dir, f"{label}{file_index}.npy")
#     np.savetxt(file_path, stimulus)

# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--wd_path', type=str, default='stim_gen', help='Path to working dir')
#     parser.add_argument('--stim_dict_file', type=str, default='/home/nfitzmaurice/stim_gen/stimuli_params_dict.txt', help='Name of dict to generate data')
#     parser.add_argument('--output_path', type=str, default='/home/nfitzmaurice/stim_gen/data', help='Path to output folder')
#     parser.add_argument('--output_folder', type=str, required=True, help='Name of the folder to store output data')

#     file_args = parser.parse_args()

#     # UIStimuliGenerator instance
#     generator_instance = UIStimuliGenerator()

#     namespace = {}
#     input_file_path = os.path.join(file_args.wd_path, file_args.stim_dict_file)
#     output_file_path = os.path.join(file_args.output_path, file_args.output_folder)
    
#     with open(input_file_path, 'r') as file:
#         exec(file.read(), {}, namespace)
#     shared_params_dict = namespace['shared_params_dict']
#     features_dict = namespace['features_dict']

#     # Dict for each data entry
#     std_random.seed(666)
#     data_dict = generator_instance.generate_combinations(features_dict, shared_params_dict)
#     print('total stimuli: ', len(data_dict))

#     # Output folder
#     os.makedirs(output_file_path, exist_ok=True)

#     # Setup multiprocessing manager for file count dictionary and lock
#     manager = multiprocessing.Manager()
#     file_count_dict = manager.dict()
#     lock = manager.Lock()

#     # Pass the generator_instance to the iter function
#     with multiprocessing.Pool(100) as pool:
#         args = [(i, data_dict, output_file_path, generator_instance) for i in range(len(data_dict))]
#         results = list(tqdm(pool.starmap(iter, [(arg, file_count_dict, lock) for arg in args]), total=len(data_dict)))

# if __name__ == '__main__':
#     main()
