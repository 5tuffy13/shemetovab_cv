import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import regionprops, label
from skimage.morphology import binary_dilation
from pathlib import Path

def count_holes(region):
    try:
        shape = region.image.shape
        padded = np.zeros((shape[0] + 2, shape[1] + 2), dtype=bool)
        padded[1:-1, 1:-1] = region.image
        inverted = np.logical_not(padded)
        labeled = label(inverted, connectivity=2)
        return max(0, np.max(labeled) - 1)
    except:
        return 0

def count_vlines(region):
    try:
        return np.sum(np.all(region.image, axis=0))
    except:
        return 0

def has_more_left_vlines(region):
    try:
        vlines = np.mean(region.image, axis=0) == 1
        half = len(vlines) // 2
        return np.sum(vlines[:half]) > np.sum(vlines[half:])
    except:
        return False

def recognize_symbol(region):
    try:
        if np.all(region.image):
            return "-"
        
        holes = count_holes(region)
        
        if holes == 2:
            vlines = count_vlines(region)
            left_vlines = has_more_left_vlines(region)
            cy, cx = region.centroid_local
            cx_normalized = cx / region.image.shape[1]
            
            return "B" if left_vlines and cx_normalized < 0.44 else "8"
        
        elif holes == 1:
            cy, cx = region.centroid_local
            cx_normalized = cx / region.image.shape[1]
            cy_normalized = cy / region.image.shape[0]
            
            if has_more_left_vlines(region):
                return "D" if cx_normalized > 0.4 or cy_normalized > 0.4 else "P"
            
            return "0" if abs(cx_normalized - cy_normalized) < 0.04 else "A"
        
        else:
            vlines = count_vlines(region)
            if vlines >= 3:
                return "1"
            
            if region.eccentricity < 0.5:
                return "*"
            
            inv_image = ~region.image
            inv_image = binary_dilation(inv_image, np.ones((3, 3)))
            labeled = label(inv_image, connectivity=1)
            num_components = np.max(labeled)
            
            if num_components == 2:
                return "/"
            elif num_components == 4:
                return "X"
            return "W"
            
    except:
        return "#"

def process_image(image_path, output_dir):
    try:
        symbols = plt.imread(image_path)[:, :, :-1]
        gray = np.mean(symbols, axis=2)
        binary = gray > 0
        labeled = label(binary)
        regions = regionprops(labeled)
        
        output_dir.mkdir(exist_ok=True)
        
        symbol_counts = {}
        
        plt.figure(figsize=(4, 4))
        for i, region in enumerate(regions, 1):
            symbol = recognize_symbol(region)
            symbol_counts[symbol] = symbol_counts.get(symbol, 0) + 1
            
            plt.cla()
            plt.title(f"Symbol: {symbol}")
            plt.imshow(region.image, cmap='gray')
            plt.axis('off')
            plt.savefig(output_dir / f"{i:03d}.png", bbox_inches='tight', dpi=100)
        
        plt.close()
        return symbol_counts
    except:
        return {}

def main():
    script_dir = Path(__file__).parent
    input_path = script_dir / "symbols.png"
    output_path = script_dir / "out2"
    results = process_image(input_path, output_path)
    print(results)

if __name__ == "__main__":
    main()