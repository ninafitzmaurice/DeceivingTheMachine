import os
import shutil
import random

def split_data(src_dir, train_dir, test_dir, test_size):
    for class_dir in os.listdir(src_dir):
        class_path = os.path.join(src_dir, class_dir)
        if os.path.isdir(class_path):
            uniform_dir = os.path.join(class_path, 'uniform')
            nonuniform_dir = os.path.join(class_path, 'nonuniform')

            uniform_files = sorted(os.listdir(uniform_dir))
            nonuniform_files = sorted(os.listdir(nonuniform_dir))

            num_test_files = int(len(uniform_files) * test_size)
            test_indices = random.sample(range(len(uniform_files)), num_test_files)

            for i in range(len(uniform_files)):
                uniform_file = uniform_files[i]
                nonuniform_file = nonuniform_files[i]

                if i in test_indices:
                    dst_uniform_dir = os.path.join(test_dir, class_dir, 'uniform')
                    dst_nonuniform_dir = os.path.join(test_dir, class_dir, 'nonuniform')
                else:
                    dst_uniform_dir = os.path.join(train_dir, class_dir, 'uniform')
                    dst_nonuniform_dir = os.path.join(train_dir, class_dir, 'nonuniform')

                os.makedirs(dst_uniform_dir, exist_ok=True)
                os.makedirs(dst_nonuniform_dir, exist_ok=True)

                src_uniform_path = os.path.join(uniform_dir, uniform_file)
                dst_uniform_path = os.path.join(dst_uniform_dir, uniform_file)
                shutil.copy(src_uniform_path, dst_uniform_path)

                src_nonuniform_path = os.path.join(nonuniform_dir, nonuniform_file)
                dst_nonuniform_path = os.path.join(dst_nonuniform_dir, nonuniform_file)
                shutil.copy(src_nonuniform_path, dst_nonuniform_path)

src_dir = 'data/UI128'
train_dir = 'data/train'
test_dir = 'data/test'
test_size = 0.2

split_data(src_dir, train_dir, test_dir, test_size)
