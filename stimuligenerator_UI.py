import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import rotate, gaussian_filter
from itertools import product, chain 
import random
from PIL import Image

class UIStimuliGenerator:
    def __init__(self, test_mode=False):
        self.test_mode = test_mode

    def create_shape_matrix(self, size, shape_type, shade, param1=0, param2=0, angle=0, sigma=0):
        # as matrix
        ## size will be determined from the coordinate grid! it will be the 
        ## relative size that the matrix can fit snug into the image
        
        # shade must be > 0 or mask will not work
        if shade <= 0:
            raise ValueError("Shape pixels must have a shade greater than 0. Use RGB values.")
        
        shade = np.array(shade, dtype=np.uint8)

        # always set background shade to 0 because of mask in fill grid function
        # can be set to different value for whatever reason... 
        bg_shade = 0
        bg_shade = np.array(bg_shade, dtype=np.uint8)
        matrix = np.full((size, size), fill_value=bg_shade, dtype=shade.dtype)
        
        if shape_type == 'cross':
            thickness = param1
            matrix[size//2 - thickness//2:size//2 + thickness//2, :] = shade
            matrix[:, size//2 - thickness//2:size//2 + thickness//2] = shade
        elif shape_type == 'curve':
            thickness = param1
            x_values = np.linspace(0, 2 * np.pi, size)
            y_values = np.sin(x_values) * (size / 4) + (size / 2)
            y_values = y_values.astype(int)
            half_thickness = thickness // 2
            for x, y in enumerate(y_values):
                matrix[max(0, y - half_thickness):min(size, y + half_thickness + 1), x] = shade
        elif shape_type == 'line':
            thickness = param1
            matrix[:, size//2 - thickness//2:size//2 + thickness//2] = shade
        elif shape_type == 'square':
            side_length = param1
            start = (size - side_length) // 2
            end = start + side_length
            matrix[start:end, start:end] = shade
        elif shape_type == 'diamond':
            side_length = param1
            for y in range(size):
                for x in range(size):
                    if abs(x - size//2) + abs(y - size//2) < side_length:
                        matrix[y, x] = shade
        elif shape_type == 'circle':
            radius = param1
            for y in range(size):
                for x in range(size):
                    if (x - size//2) ** 2 + (y - size//2) ** 2 < radius ** 2:
                        matrix[y, x] = shade
        elif shape_type == 'oval':
            width, height = param1, param2
            for y in range(size):
                for x in range(size):
                    if (((x - size//2) ** 2) / (width ** 2) + ((y - size//2) ** 2) / (height ** 2)) < 1:
                        matrix[y, x] = shade

        
        # change orientation
        if angle != 0:
            matrix = rotate(matrix, angle, reshape=False, order=0, mode='constant', cval=bg_shade)
        
        # for gaussian blur(sigma > 0)
        if sigma > 0:
            matrix = gaussian_filter(matrix, sigma=sigma)

        return matrix

    def make_grid(self, x_points, y_points, image_height, image_width, shift):
        # step size based on the number of points and image dimensions
        x_step = image_width / (x_points + 1)
        y_step = image_height / (y_points + 1)
        
        # start and end points 1 step in from the edge
        x_start, x_end = x_step, image_width - x_step
        y_start, y_end = y_step, image_height - y_step
        
        # base coordinates for the grid
        x_coords = np.linspace(x_start, x_end, x_points)
        y_coords = np.linspace(y_start, y_end, y_points)
        
        # meshgrid to create the base coordinate grid
        coords = np.array(np.meshgrid(x_coords, y_coords)).T.reshape(-1, 2)
        
        if shift == True: 
            x_shift = np.zeros((y_points, x_points))
            y_shift = np.zeros((y_points, x_points))
            
            # applying shift for every second row
            x_shift[:, 1::2] = 0.2 * x_step  # modify shift factor! I find it works better when not a full step
            y_shift[1::2, :] = 0.2 * x_step  # modify shift factor! I find it works better when not a full step
            

            # flatten the shift arrays to match the coords shape
            shift = np.stack((x_shift.flatten(), y_shift.flatten()), axis=-1)
            
            # add the shift
            coords += shift
        return coords

    # RETURNS CENTRAL AND PERIPHERAL COORDINATES!!
    def get_cent_periph(self, coords, percentage, selection_type):
        # get cent of each axis
        cent_x = np.mean(coords[:, 0])
        cent_y = np.mean(coords[:, 1])
        
        # grid dimensions and aspect ratio
        x_min, x_max = np.min(coords[:, 0]), np.max(coords[:, 0])
        y_min, y_max = np.min(coords[:, 1]), np.max(coords[:, 1])
        x_range = x_max - x_min
        y_range = y_max - y_min
        aspect_ratio = x_range / y_range
        
        # for rectangular central region
        if selection_type == 'rect':

            # total area and the target area
            total_area = x_range * y_range
            target_area = total_area * (percentage / 100)
            
            # dimensions of the target rectangle preserving aspect ratio
            target_x = np.sqrt(target_area * aspect_ratio)
            target_y = target_x / aspect_ratio
            
            # bounds of the central rectangle
            rect_x_min = cent_x - target_x / 2
            rect_x_max = cent_x + target_x / 2
            rect_y_min = cent_y - target_y / 2
            rect_y_max = cent_y + target_y / 2
            
            # filter the points
            central_mask = (coords[:, 0] >= rect_x_min) & (coords[:, 0] <= rect_x_max) & \
                        (coords[:, 1] >= rect_y_min) & (coords[:, 1] <= rect_y_max)
            
            central_coordinates = coords[central_mask]

        # radial central selection
        elif selection_type == 'radial':
            
            # scale the x and y coordinates by the aspect ratio to preserve the grid shape
            scaled_x = (coords[:, 0] - cent_x) * (1 / np.sqrt(aspect_ratio))
            scaled_y = (coords[:, 1] - cent_y) * np.sqrt(aspect_ratio)
            
            # calculate scaled distances
            distances = np.sqrt(scaled_x**2 + scaled_y**2)
            
            # threshold distance for the desired percentage of points
            points_to_select = int(np.ceil(len(coords) * (percentage / 100)))
            sorted_distances = np.sort(distances)
            distance_threshold = sorted_distances[points_to_select - 1]
            
            # filter points within the scaled distance threshold
            central_mask = distances <= distance_threshold
            central_coordinates = coords[central_mask]

        else:
            raise ValueError("Invalid selection type. Choose 'rect' or 'radial'.")
        
        # compute peripheral coordinates by inverting the central mask
        peripheral_coordinates = coords[~central_mask]

        return central_coordinates, peripheral_coordinates

    def test_grid(self, original_coords, cent_points, periph_points):
        plt.figure(figsize=(12, 6))

        # periph
        plt.subplot(1, 2, 1)
        plt.scatter(original_coords[:, 0], original_coords[:, 1], s=10, c='lightgray', label='Original Grid')
        plt.scatter(periph_points[:, 0], periph_points[:, 1], s=30, c='red', label='Peripheral Points')
        plt.xlim(np.min(original_coords[:, 0]), np.max(original_coords[:, 0]))
        plt.ylim(np.min(original_coords[:, 1]), np.max(original_coords[:, 1]))
        plt.axis('equal')
        plt.grid(True)
        plt.legend()
        plt.title('Peripheral Points')

        # cent
        plt.subplot(1, 2, 2)
        plt.scatter(original_coords[:, 0], original_coords[:, 1], s=10, c='lightgray', label='Original Grid')
        plt.scatter(cent_points[:, 0], cent_points[:, 1], s=30, c='blue', label='Central Points')
        plt.xlim(np.min(original_coords[:, 0]), np.max(original_coords[:, 0]))
        plt.ylim(np.min(original_coords[:, 1]), np.max(original_coords[:, 1]))
        plt.axis('equal')
        plt.grid(True)
        plt.legend()
        plt.title('Central Points')

        plt.show()


    @staticmethod
    def generate_combinations(feature_params, shared_params):
        '''
        This takes the shared feature parameters and the combined parameters and creates a dict
        containing all possible combinations of stimuli for each feature type

        it excludes all conicdentally uniform stimuli where the centre and periphery are matched 
        from the combinatorial process

        uniform stimuli dictionaries are generated to matched the nonuniform instance before them,
        so, the uniform stimuli will follow the same order 

        '''
        num_features = len(feature_params) * 2

        def expand_feature_params(feature, feature_params):
            # all params iterable and have defaults
            param1 = feature_params.get('param1', [0])
            orientation = feature_params.get('orientation', [0]) if 'orientation' in feature_params else [0]
            param2 = feature_params.get('param2', [0]) if 'param2' in feature_params else [0]

            bg_shade = [shared_params['bg_shade']] if isinstance(shared_params['bg_shade'], int) else shared_params['bg_shade']
            x_points = [shared_params['x_points']] if isinstance(shared_params['x_points'], int) else shared_params['x_points']
            # I set x points to y points because the stimuli always look better this way 
            y_points = [shared_params['y_points']] if isinstance(shared_params['y_points'], int) else shared_params['x_points']
            cent_size = [shared_params['cent_size']] if isinstance(shared_params['cent_size'], int) else shared_params['cent_size']

            cent_type = shared_params.get('cent_type', 'rect')
            feature_shade = shared_params.get('feature_shade', 255)
            shift = shared_params.get('shift')

            combinations = product(
                param1, param1,  # for central and peripheral
                orientation, orientation,  # for central and peripheral
                param2, param2,  # for central and peripheral
                cent_size, cent_type, 
                feature_shade,
                x_points, shift, bg_shade
            )

            # generate stimulus params and labels for each combination
            all_combinations = []
            for p1c, p1p, oc, op, p2c, p2p, cs, ct, fs, xp, shift, bg in combinations:
                stimulus_params = {
                    **shared_params,
                    'feature_type': feature,
                    'param1_cent': p1c, 'param1_periph': p1p,
                    'orientation_cent': oc, 'orientation_periph': op,
                    'param2_cent': p2c, 'param2_periph': p2p,
                    'cent_size': cs, 'cent_type': ct,
                    'feature_shade': fs, 'x_points': xp, 'y_points': xp,
                    'shift': shift, 'bg_shade': bg
                }

                # logic for uniformity label in STIM PARAMS
                is_uniform = stimulus_params['cent_size'] in [0, 100] or (
                    p1c == p1p and p2c == p2p and oc == op
                )

                if is_uniform:
                    continue

                uniformity_label = 'uniform' if is_uniform else 'nonuniform'
                # add new key for unifomrity
                stimulus_params['uniformity'] = uniformity_label
                # print(feature, uniformity_label, stimulus_params['onehot_label'])

                # add the stimulus_params to the all_combinations list
                all_combinations.append(stimulus_params)

            return all_combinations

        # for each feature all combinations 
        all_combinations = list(chain.from_iterable(
            expand_feature_params(feature, params) 
            for feature, params in feature_params.items()
        ))

        def create_uniform_stimuli(nonuniform_stimuli):
            uniform_stimuli = []
            for stimulus in nonuniform_stimuli:
                uniform_stimulus = stimulus.copy()
                uniform_stimulus['param1_periph'] = uniform_stimulus['param1_cent']
                uniform_stimulus['orientation_periph'] = uniform_stimulus['orientation_cent']
                uniform_stimulus['param2_periph'] = uniform_stimulus['param2_cent']
                uniform_stimulus['uniformity'] = 'uniform'
                uniform_stimuli.append(uniform_stimulus)
            return uniform_stimuli
        
        uniform_combinations = create_uniform_stimuli(all_combinations)
        all_combinations.extend(uniform_combinations)
        
        print('miaow')
        return all_combinations


    def place_feature_on_grid(self, grid, shape, coordinate):
        # mask shape
        mask = shape > 0

        size = shape.shape[0]
        half_size = size // 2
        x_center, y_center = coordinate

        # start and end of subgrid
        x_start = int(max(x_center - half_size, 0))
        y_start = int(max(y_center - half_size, 0))
        x_end = int(min(x_center + half_size, grid.shape[1]))
        y_end = int(min(y_center + half_size, grid.shape[0]))

        # adjust for subgrid alignment
        x_end = int(min(x_end, x_start + shape.shape[1]))
        y_end = int(min(y_end, y_start + shape.shape[0]))

        # calculate area of shape to use
        shape_x_start = int(max(0, half_size - (x_center - x_start)))
        shape_y_start = int(max(0, half_size - (y_center - y_start)))
        shape_x_end = int(min(shape.shape[1], shape_x_start + (x_end - x_start)))
        shape_y_end = int(min(shape.shape[0], shape_y_start + (y_end - y_start)))

        # use shape and mask dimensions to align
        subgrid = grid[y_start:y_end, x_start:x_end]
        submask = mask[shape_y_start:shape_y_end, shape_x_start:shape_x_end]

        # check mask aligns with subgrid dimension
        subgrid[submask] = shape[shape_y_start:shape_y_end, shape_x_start:shape_x_end][submask]

    def compress_img(self, input, out_size):
        #  numpy array to a PIL Image
        img = Image.fromarray(input)
        img_resized = img.resize((out_size, out_size))
        output_array = np.array(img_resized)
        
        return output_array
    

    def create_UI_stim(self, stimulus_params, compress_size=None):
        label_class = str(stimulus_params['feature_type']+'_'+stimulus_params['uniformity'])

        bg_shade = stimulus_params['bg_shade']
        image_height = stimulus_params['image_height']
        image_width = stimulus_params['image_width']
        shift = stimulus_params['shift']
        x_points = stimulus_params['x_points']
        y_points = stimulus_params['y_points']
        cent_size = stimulus_params['cent_size']
        cent_type = stimulus_params['cent_type']
        feature_type = stimulus_params['feature_type']
        feature_shade = stimulus_params['feature_shade']

        feature_orientation_cent = float(stimulus_params.get('orientation_cent', stimulus_params.get('orientation', 0)))
        feature_orientation_periph = float(stimulus_params.get('orientation_periph', stimulus_params.get('orientation', 0)))

        param1_cent = stimulus_params.get('param1_cent', stimulus_params.get('param1')) 
        param1_periph = stimulus_params.get('param1_periph', stimulus_params.get('param1')) 

        param2_cent = stimulus_params.get('param2_cent', stimulus_params.get('param2'))
        param2_periph = stimulus_params.get('param2_periph', stimulus_params.get('param2'))

        # 1. generate coords
        coords = self.make_grid(x_points, y_points, image_height, image_width, shift)
        cent_coords, periph_coords = self.get_cent_periph(coords, cent_size, cent_type)
        # print(len(cent_coords),len(periph_coords),len(coords))

        # 2. generate features for central and peripheral areas
        # feature size based on grid steps
        x_step = image_width / (x_points + 1)
        y_step = image_height / (y_points + 1)
        feature_size = int(min(x_step, y_step) - (min(x_step, y_step) * 0.3))  # prevent overlap

        cent_feature = self.create_shape_matrix(feature_size, feature_type, feature_shade, 
                                        param1=param1_cent, param2=param2_cent, angle=feature_orientation_cent)
        
        periph_feature = self.create_shape_matrix(feature_size, feature_type, feature_shade, 
                                            param1=param1_periph, param2=param2_periph, angle=feature_orientation_periph)
        
        # 3. blank canvas
        canvas = np.full((image_height, image_width), fill_value=bg_shade, dtype=np.uint8)

        # 4. place the features on the coordinate grids
        for coord in cent_coords:
            self.place_feature_on_grid(canvas, cent_feature, coord)
        for coord in periph_coords:
            self.place_feature_on_grid(canvas, periph_feature, coord)
        
        # flatten canvas to create a vector
        # canvas_vector = canvas.flatten()

        if compress_size is not None:
            canvas = self.compress_img(canvas, compress_size)

        return canvas, label_class
    
    
    def plot_stimuli(self, stimuli_params_list, compress_size=None):
        num_stimuli = len(stimuli_params_list)
        cols = int(np.ceil(np.sqrt(num_stimuli))) 
        rows = int(np.ceil(num_stimuli / cols)) 
        
        fig, axs = plt.subplots(rows, cols, figsize=(5*cols, 5*rows))
        if num_stimuli == 1:
            axs = [axs] 
        
        for ax, stimulus_params in zip(np.ravel(axs), stimuli_params_list):
            if not compress_size:
                canvas_vector, label_class = self.create_UI_stim(stimulus_params)
                image = canvas_vector.reshape(stimulus_params['image_height'], stimulus_params['image_width'])

            elif compress_size is not None:
                canvas_vector, label_class = self.create_UI_stim(stimulus_params, compress_size)
                image = canvas_vector.reshape(compress_size, compress_size)
            
            feature_type = stimulus_params['feature_type']
            uniformity_rating = stimulus_params['uniformity']
            
            ax.imshow(image, cmap='gray',vmin = 0, vmax = 255)
            ax.set_title(f"{label_class} ({feature_type}, {uniformity_rating})")
            ax.axis('off')
        
        for ax in np.ravel(axs)[num_stimuli:]:
            ax.axis('off')
        
        plt.tight_layout()
        plt.savefig('plots.jpg')

    def get_stim_counts(self, stimuli_params_list):
        class_counts = {}

        # iterate through the stimuli parameters list
        for stimulus_params in stimuli_params_list:
            # the class and uniformity label
            class_name = stimulus_params['feature_type']
            uniformity_label = stimulus_params['uniformity']

            if class_name not in class_counts:
                class_counts[class_name] = {'uniform': 0, 'nonuniform': 0}
            class_counts[class_name][uniformity_label.lower()] += 1

        # the counts for each class after processing all items
        return class_counts
        
    # TESTS RANDOM SELECTION OF STIMULI
    def run_test_mode(self, features_dict, shared_params_dict, compress_size=None):
        combinations = self.generate_combinations(features_dict, shared_params_dict)
        print('TOTAL STIMULI:  ', len(combinations))
        print(self.get_stim_counts(combinations))
        
        # randomly sample combinations
        selected_combinations = random.sample(combinations, 6)
        # selected_combinations = combinations[100:106]
        if not compress_size:
            self.plot_stimuli(selected_combinations)
        elif compress_size is not None:
            print('awa')
            print(compress_size)
            self.plot_stimuli(selected_combinations, compress_size)
        print(selected_combinations)


if __name__ == "__main__":
    generator = UIStimuliGenerator()
    
    namespace = {}
    # path to dict stimuli params txt file 
    file_path = '/home/nfitzmaurice/stim_gen/demos/big_diff.txt'
    with open(file_path, 'r') as file:
        exec(file.read(), {}, namespace)
    # extract dicts for shared and feature specific params
    shared_params_dict = namespace['shared_params_dict']
    features_dict = namespace['features_dict']

    generator.run_test_mode(features_dict, shared_params_dict, 128)
