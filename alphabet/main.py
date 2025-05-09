import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import regionprops, label
from skimage.morphology import binary_dilation
from pathlib import Path

def count_holes(region):
    shape = region.image.shape
    padded = np.zeros((shape[0] + 2, shape[1] + 2))
    padded[1:-1, 1:-1] = region.image
    inverted = np.logical_not(padded)
    labeled = label(inverted)
    return np.max(labeled) - 1

def count_vlines(region):
    return np.sum(np.all(region.image, axis=0))

def count_lgr_vlines(region):
    x = np.mean(region.image, axis=0) == 1
    mid = len(x) // 2
    return np.sum(x[:mid]) > np.sum(x[mid:])

def recognize(region):
    if np.all(region.image):
        return "-"
    
    holes = count_holes(region)
    
    if holes == 2:
        vlines = count_vlines(region)
        flag_lr = count_lgr_vlines(region)
        cy, cx = region.centroid_local
        cx /= region.image.shape[1]
        
        if flag_lr and cx < 0.44:
            return "B"
        return "8"
    
    elif holes == 1:
        cy, cx = region.centroid_local
        cx /= region.image.shape[1]
        cy /= region.image.shape[0]
        
        if count_lgr_vlines(region):
            if cx > 0.4 or cy > 0.4:
                return "D"
            return "P"
        
        if abs(cx - cy) < 0.04:
            return "0"
        return "A"
    
    else:
        if count_vlines(region) >= 3:
            return "1"
        
        if region.eccentricity < 0.5:
            return "*"
        
        inv_image = ~region.image
        dilated = binary_dilation(inv_image, np.ones((3, 3)))
        labeled = label(dilated, connectivity=1)
        num_labels = np.max(labeled)
        
        if num_labels == 2:
            return "/"
        elif num_labels == 4:
            return "X"
        return "W"


symbols = plt.imread(Path(__file__).parent / "symbols.png")[:, :, :-1]
gray = np.mean(symbols, axis=2)
binary = gray > 0
labeled = label(binary)
regions = regionprops(labeled)

result = {}
out_path = Path(__file__).parent / "out"
out_path.mkdir(exist_ok=True)

plt.figure()
for i, region in enumerate(regions, 1):
    print(f"Region {i}/{len(regions)}")
    symbol = recognize(region)
    result[symbol] = result.get(symbol, 0) + 1
    
    plt.clf()
    plt.title(f"Symbol: {symbol}")
    plt.imshow(region.image, cmap='gray')
    plt.savefig(out_path / f"img_{i:03d}.png")

print("Results:", result)