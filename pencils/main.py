import numpy as np 
import matplotlib.pyplot as plt
from skimage.measure import regionprops, label
from skimage.filters import sobel, threshold_otsu
from scipy.ndimage import binary_fill_holes

def detect_pencil(shape, img_size):
    height, width = shape.image.shape
    top, left, bottom, right = shape.bbox
    center_y, center_x = shape.centroid
    local_y = center_y - top
    local_x = center_x - left
    norm_x = local_x / width
    norm_y = local_y / height

    is_centered = (0.4 < norm_x < 0.6) and (0.4 < norm_y < 0.6)

    diagonal = np.sqrt(height**2 + width**2)
    size_valid = (diagonal > img_size / 2) and (diagonal < img_size)

    perimeter_ratio = shape.perimeter / diagonal
    form_valid = 2.48 < perimeter_ratio < 5.52

    elongated = (shape.perimeter ** 2) / shape.area > 33.33

    return is_centered and size_valid and form_valid and elongated

total_pencils = 0
for idx in range(1, 13):
    img = plt.imread(f"./pencils/images/img ({idx}).jpg").mean(axis=2)
    edges = sobel(img)

    threshold = threshold_otsu(edges) / 2
    edges[edges < threshold] = 0
    edges[edges >= threshold] = 1

    filled = binary_fill_holes(edges, np.ones((3, 3)))
    labeled_img = label(filled)
    shapes = regionprops(labeled_img)
    shapes = sorted(shapes, key=lambda x: x.perimeter)

    pencil_count = 0
    min_dimension = min(labeled_img.shape)

    for shape in shapes:
        if detect_pencil(shape, min_dimension):
            pencil_count += 1

    print(f'Изображение {idx}: {pencil_count} карандашей')
    total_pencils += pencil_count

print(f'Всего карандашей: {total_pencils}')