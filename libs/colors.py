import numpy as np


COLORS = np.array([
    [0, 0, 0],            # Background
    [128, 0, 0],          # Building
    [0, 0, 255],          # Structures
    [0, 128, 128],        # Road
    [128, 128, 128],      # Track
    [0, 128, 0],          # Trees
    [128, 128, 0],        # Crops
    [128, 0, 128],        # Waterway
    [64, 0, 0],           # S. Water
    [192, 128, 128],      # Track(Vehicle Large)
    [64, 192, 0],         # Car(Vehicle Small)
])


def color_mapping_on_batch(labels):
    n, h, w = labels.shape

    color_map = np.zeros((n, 3, h, w))
    for k in range(n):
        for i in range(h):
            for j in range(w):
                color_map[k, :, i, j] = COLORS[labels[k, i, j]]/255.

    return color_map



"""
def color_mapping_on_batch(labels):
    labels = cm.jet_r(labels / 182.)[..., :3] * 255
    labels = np.transpose(labels, (0, 3, 1, 2))

    return labels
"""