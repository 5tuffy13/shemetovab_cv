import matplotlib.pyplot as plt
import numpy as np
from skimage.measure import regionprops, label
from pathlib import Path

def extractor(region):
    normalized_area = region.area / region.image.size
    cy_rel, cx_rel = region.centroid_local
    cy_rel /= region.image.shape[0]
    cx_rel /= region.image.shape[1]
    normalized_perimeter = region.perimeter / region.image.size
    eccentricity = region.eccentricity
    holes = 1 - region.euler_number
    solidity = region.solidity

    aspect_ratio = region.image.shape[1] / region.image.shape[0] if region.image.shape[0] != 0 else 0


    vert_lines = np.sum(np.all(region.image, axis=0))
    horiz_lines = np.sum(np.all(region.image, axis=1))

    cy_idx = int(region.centroid_local[0])
    cx_idx = int(region.centroid_local[1])
    cy_idx = max(0, min(cy_idx, region.image.shape[0] - 1))
    cx_idx = max(0, min(cx_idx, region.image.shape[1] - 1))

    row_crossings = np.sum(region.image[cy_idx, :-1] != region.image[cy_idx, 1:]) if region.image.shape[1] > 1 else 0
    col_crossings = np.sum(region.image[:-1, cx_idx] != region.image[1:, cx_idx]) if region.image.shape[0] > 1 else 0

    h, w = region.image.shape[0] // 2, region.image.shape[1] // 2
    if h > 0 and region.image.size > 0:
        sym_h = np.sum(region.image[:h, :] == np.flipud(region.image[-h:, :])) / region.image.size
    else:
        sym_h = 1.0

    if w > 0 and region.image.size > 0:
        sym_v = np.sum(region.image[:, :w] == np.fliplr(region.image[:, -w:])) / region.image.size
    else:
        sym_v = 1.0

    return np.array([
        normalized_area, cy_rel, cx_rel, normalized_perimeter, eccentricity,
        holes, solidity, aspect_ratio, vert_lines, horiz_lines,
        row_crossings, col_crossings, sym_v, sym_h
    ])

def euclidean_distance(v1, v2):
    return np.sqrt(np.sum((v1 - v2) ** 2))

def classify_nearest_neighbor(feature_vector, templates):
    result_label = "_"
    min_dist = float('inf')
    for key, template_vector in templates.items():
        d = euclidean_distance(feature_vector, template_vector)
        if d < min_dist:
            result_label = key
            min_dist = d
    return result_label

alphabet_large_path = "alphabet.png"
try:
    alphabet_large = plt.imread("vector_recognition/alphabet.png")
    if alphabet_large.shape[2] > 3:
        alphabet_large = alphabet_large[:, :, :-1]
except FileNotFoundError:
    print(f"Error: File not found at {alphabet_large_path}")
    exit()
except Exception as e:
    print(f"Error loading {alphabet_large_path}: {e}")
    exit()

gray_large = alphabet_large.mean(axis=2)
binary_large = gray_large > 0.5
labeled_large = label(binary_large)
regions_large = regionprops(labeled_large)
print(f"Detected {len(regions_large)} regions in {alphabet_large_path}")


alphabet_small_path = "vector_recognition/alphabet-small.png"
try:
    alphabet_small = plt.imread(alphabet_small_path)
    if alphabet_small.shape[2] > 3:
        alphabet_small = alphabet_small[:, :, :-1]
except FileNotFoundError:
    print(f"Error: File not found at {alphabet_small_path}")
    exit()
except Exception as e:
    print(f"Error loading {alphabet_small_path}: {e}")
    exit()

gray_small = alphabet_small.mean(axis=2)
binary_small = gray_small < 0.5
labeled_small = label(binary_small)
regions_small = regionprops(labeled_small)
print(f"Detected {len(regions_small)} regions in {alphabet_small_path}")


templates = {
    "8": extractor(regions_small[0]),
    "0": extractor(regions_small[1]),
    "A": extractor(regions_small[2]),
    "B": extractor(regions_small[3]),
    "1": extractor(regions_small[4]),
    "W": extractor(regions_small[5]),
    "X": extractor(regions_small[6]),
    "*": extractor(regions_small[7]),
    "/": extractor(regions_small[8]),
    "-": extractor(regions_small[9])
}


results = {}
for region in regions_large:
    if region.area < 10:  # Skip noise
        continue
    v = extractor(region)
    symbol = classify_nearest_neighbor(v, templates)
    results[symbol] = results.get(symbol, 0) + 1

print("\nSymbol Counts:")
print(results)