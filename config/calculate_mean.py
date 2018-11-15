import numpy as np
import os.path as osp
from tqdm import tqdm
import glob


def get_all_nps(data_path):
    # Set paths
    label_path = list(glob.iglob(osp.join(data_path, '*_label.npy'), recursive=False))
    image_path = []
    for i, single_image_path in enumerate(label_path):
        single_image_path = single_image_path.split('.')[0][:-6] + '.npy'
        image_path.append(single_image_path)

    return image_path


def calculate_mean(image_paths):
    pixel_count = 0
    sums = np.zeros((3))
    for i, image_path in tqdm(enumerate(image_paths)):
        image = np.load(image_path)
        sums += np.sum(image, (0, 1))

        w, h, c = image.shape
        pixel_count += w * h

    means = sums / pixel_count
    return means


if __name__ == '__main__':
    images_path = get_all_nps('/mnt/nas/Dataset/dstl/numpy')
    mean = calculate_mean(images_path)
    print(mean)
