import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import label, regionprops
from skimage.color import rgb2hsv

img = plt.imread('figures_and_colors/balls_and_rects.png')

grayscale = img.mean(axis=2)
bin_img = grayscale > 0

labeled_img = label(bin_img)
shapes = regionprops(labeled_img)

num_circles = 0
circle_hues = []
num_rectangles = 0
rectangle_hues = []

for shape in shapes:
    height, width = shape.image.shape
    total_area = height * width
    is_rect_shape = (shape.area == total_area)
  
    proportion = shape.minor_axis_length / shape.major_axis_length
    is_circle_shape = (proportion > 0.9)

    y_coord, x_coord = shape.centroid
    color_hue = rgb2hsv(img[int(y_coord), int(x_coord)])[0]
 
    if is_circle_shape and not is_rect_shape:
        num_circles += 1
        circle_hues.append(color_hue)
    else:
        num_rectangles += 1
        rectangle_hues.append(color_hue)

all_hues = [rgb2hsv(img[int(s.centroid[0]), int(s.centroid[1])])[0] for s in shapes]

print(f'Количество кругов: {num_circles}')
print(f'Количество прямоугольников: {num_rectangles}')
print(f'Количество всех фигур: {len(all_hues)}')

def evaluate_hues(hue_list, threshold_mult=1.0, category=''):
    hue_diffs = np.diff(sorted(hue_list))
    split_points = np.where(hue_diffs > np.std(hue_diffs) * threshold_mult)[0]
    shade_count = len(split_points) + 1
    print(f"\n{category} Оттенков: {shade_count}")
    start_idx = 0
    for idx, point in enumerate(split_points, start=1):
        group_size = point - start_idx + 1
        print(f"Оттенок {idx}: {group_size} объектов")
        start_idx = point + 1
    final_group = len(hue_list) - start_idx
    print(f"Оттенок {shade_count}: {final_group} объектов")

print("\nОбщие оттенки")
evaluate_hues(all_hues, threshold_mult=1.0, category='Общие')
print("\nОттенки кругов")
evaluate_hues(circle_hues, threshold_mult=2.0, category='Круги')
print("\nОттенки прямоугольников")
evaluate_hues(rectangle_hues, threshold_mult=2.0, category='Прямоугольники')