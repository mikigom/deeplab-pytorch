import math
import random
import numpy as np
import cv2


def _transform(image, label, scale=(0.5, 0.75, 1., 1.25, 1.5), rotation=15, crop_size=256, flip=False):
    scale_factor = 1
    if scale is not None:
        scale_factor = random.choice(scale)

    large_border_scale = 1
    small_border_scale = 1
    if rotation is not None:
        thetas = [k * rotation for k in range(int(90 / rotation))]
        theta = math.radians(random.choice(thetas))
        large_border_scale = math.cos(theta) + math.sin(theta)
        small_border_scale = math.fabs(math.sin(theta) - math.cos(theta))

    if flip:
        # Random flipping
        if random.random() < 0.5:
            image = np.fliplr(image).copy()  # HWC
            label = np.fliplr(label).copy()  # HW

    if rotation is not None:
        k = random.choice([0, 1, 2, 3])
        image = np.rot90(image, k=k)
        label = np.rot90(label, k=k)

    base_h, base_w = label.shape
    large_patch_border_from_center = int(large_border_scale * scale_factor * crop_size / 2)
    small_patch_border_from_center = int(small_border_scale * scale_factor * crop_size / 2)

    center_h = random.randint(large_patch_border_from_center, base_h - large_patch_border_from_center)
    center_w = random.randint(large_patch_border_from_center, base_w - large_patch_border_from_center)

    src = np.float32([[center_w + small_patch_border_from_center, center_h - large_patch_border_from_center],
                      [center_w + large_patch_border_from_center, center_h + small_patch_border_from_center],
                      [center_w - large_patch_border_from_center, center_h - small_patch_border_from_center]]
                     )

    lineThickness = 2
    cv2.line(image, (center_w + large_patch_border_from_center, center_h + small_patch_border_from_center),
                    (center_w - large_patch_border_from_center, center_h - small_patch_border_from_center), (0, 255, 0), lineThickness)
    cv2.line(image, (center_w + small_patch_border_from_center, center_h - large_patch_border_from_center),
                    (center_w, center_h), (0, 255, 0), lineThickness)
    cv2.imshow('image', image)
    cv2.waitKey(0)

    dst = np.float32([[crop_size, 0],
                      [crop_size, crop_size],
                      [0, 0]]
                     )

    M = cv2.getAffineTransform(src, dst)
    print(M)
    image = cv2.warpAffine(image, M, (crop_size, crop_size))
    label = cv2.warpAffine(label, M, (crop_size, crop_size))
    
    print(image.shape)
    print(label.shape)

    # HWC -> CHW
    # image = image.transpose(2, 0, 1)
    return image, label


if __name__ == '__main__':
    image = cv2.imread('test.png')
    label = image[:, :, 0]

    image, label = _transform(image, label, scale=None, rotation=15, crop_size=256)

    cv2.imshow('image', image)
    cv2.imshow('label', label)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

