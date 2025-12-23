"""
Visual comparison of old vs new approach for skeleton generation.
"""

import os
import tempfile
import subprocess
from PIL import Image
import numpy as np
from skimage.morphology import skeletonize, medial_axis
from scipy.ndimage import binary_opening, binary_closing, label
from freetype import Face, FT_LOAD_RENDER, FT_KERNING_DEFAULT
import xml.etree.ElementTree as ET

def render_text(font_path: str, text: str, size_pt: int, dpi: int):
    """Render text and return grayscale image."""
    face = Face(font_path)
    face.set_char_size(size_pt * 64, 0, dpi, 0)
    
    # Measure
    pen_x = 0
    prev_char_index = None
    max_above_baseline = 0
    max_below_baseline = 0
    
    for ch in text:
        face.load_char(ch, FT_LOAD_RENDER)
        slot = face.glyph
        if prev_char_index is not None:
            kerning = face.get_kerning(prev_char_index, face.get_char_index(ch), FT_KERNING_DEFAULT)
            pen_x += kerning.x >> 6
        pen_x += (slot.advance.x >> 6)
        metrics = slot.metrics
        above = metrics.horiBearingY >> 6
        below = (metrics.height >> 6) - above
        max_above_baseline = max(max_above_baseline, above)
        max_below_baseline = max(max_below_baseline, below)
        prev_char_index = face.get_char_index(ch)
    
    width = max(1, pen_x)
    height = max(1, max_above_baseline + max_below_baseline)
    img = Image.new("L", (width, height), color=0)
    
    # Render
    pen_x = 0
    prev_char_index = None
    for ch in text:
        face.load_char(ch, FT_LOAD_RENDER)
        slot = face.glyph
        if prev_char_index is not None:
            kerning = face.get_kerning(prev_char_index, face.get_char_index(ch), FT_KERNING_DEFAULT)
            pen_x += kerning.x >> 6
        bitmap = slot.bitmap
        top = slot.bitmap_top
        left = slot.bitmap_left
        if bitmap.width and bitmap.rows:
            glyph_img = Image.frombytes("L", (bitmap.width, bitmap.rows), bytes(bitmap.buffer))
            y = max_above_baseline - top
            x = pen_x + left
            img.paste(glyph_img, (x, y), glyph_img)
        pen_x += (slot.advance.x >> 6)
        prev_char_index = face.get_char_index(ch)
    
    return img

def old_approach(gray: np.ndarray):
    """Old approach: fixed threshold + aggressive skeletonize."""
    binary = gray > 128
    binary = binary_opening(binary, structure=np.ones((2, 2)))
    binary = binary_closing(binary, structure=np.ones((2, 2)))
    skel = skeletonize(binary)
    return skel, binary

def new_approach(gray: np.ndarray):
    """New approach: adaptive threshold + medial axis."""
    # Adaptive threshold
    hist, bin_edges = np.histogram(gray, bins=256, range=(0, 256))
    cumsum = np.cumsum(hist)
    target_mass = cumsum[-1] * 0.7
    threshold = np.argmax(cumsum >= target_mass)
    threshold = max(100, min(180, threshold))
    
    binary = gray > threshold
    binary = binary_opening(binary, structure=np.ones((2, 2)))
    binary = binary_closing(binary, structure=np.ones((2, 2)))
    
    # Medial axis with artifact removal
    skel, distance_transform = medial_axis(binary, return_distance=True)
    
    if np.max(distance_transform) > 1:
        min_distance = np.percentile(distance_transform[skel], 25)
        skel = skel & (distance_transform >= min_distance * 0.8)
    
    # Remove tiny components
    labeled, num_features = label(skel)
    component_sizes = np.bincount(labeled.ravel())
    for component_id in range(1, num_features + 1):
        if component_sizes[component_id] < 2:
            skel[labeled == component_id] = False
    
    return skel, binary

def count_potrace_paths(skel_img: Image.Image):
    """Run potrace and count output paths."""
    with tempfile.TemporaryDirectory() as tmpdir:
        pbm_path = os.path.join(tmpdir, "test.pbm")
        svg_path = os.path.join(tmpdir, "test.svg")
        
        pbm_img = skel_img.convert("1")
        pbm_img.save(pbm_path)
        
        potrace_exe = os.path.join(os.path.dirname(__file__), "bin", "potrace.exe")
        try:
            subprocess.run([
                potrace_exe, pbm_path, "-s", "-o", svg_path, "-i",
                "--turdsize", "0", "--opttolerance", "0.4", "--alphamax", "0.8",
            ], capture_output=True, check=True)
            
            tree = ET.parse(svg_path)
            root = tree.getroot()
            nsmap = {"svg": "http://www.w3.org/2000/svg"}
            paths = root.findall(".//svg:path", nsmap)
            return len(paths)
        except:
            return 0

def main():
    test_chars = [
        ("m", "fonts/CambriaMath.ttf", 28, 600),
        ("ff", "fonts/CambriaMath.ttf", 28, 600),
        ("&", "fonts/CambriaMath.ttf", 28, 600),
    ]
    
    print("\n" + "="*70)
    print("COMPARISON: Old Approach vs New Approach")
    print("="*70 + "\n")
    
    for char, font_path, size_pt, dpi in test_chars:
        if not os.path.exists(font_path):
            continue
        
        print(f"Character: '{char}'")
        print("-" * 70)
        
        # Render
        img = render_text(font_path, char, size_pt, dpi)
        gray = np.array(img.convert("L"), dtype=np.uint8)
        
        # Old approach
        old_skel, old_binary = old_approach(gray)
        old_paths = count_potrace_paths(Image.fromarray((old_skel * 255).astype(np.uint8)))
        
        # New approach
        new_skel, new_binary = new_approach(gray)
        new_paths = count_potrace_paths(Image.fromarray((new_skel * 255).astype(np.uint8)))
        
        # Save for visual inspection
        Image.fromarray((old_skel * 255).astype(np.uint8)).save(f"compare_old_{char}.png")
        Image.fromarray((new_skel * 255).astype(np.uint8)).save(f"compare_new_{char}.png")
        
        print(f"Old approach (fixed threshold 128 + skeletonize):")
        print(f"  Binary pixels:    {np.sum(old_binary)}")
        print(f"  Skeleton pixels:  {np.sum(old_skel)} ({np.sum(old_skel)/np.sum(old_binary)*100:.1f}%)")
        print(f"  Potrace paths:    {old_paths}")
        
        print(f"\nNew approach (adaptive threshold + medial_axis):")
        print(f"  Binary pixels:    {np.sum(new_binary)}")
        print(f"  Skeleton pixels:  {np.sum(new_skel)} ({np.sum(new_skel)/np.sum(new_binary)*100:.1f}%)")
        print(f"  Potrace paths:    {new_paths}")
        
        print(f"\nImprovement:")
        print(f"  Binary change:    {np.sum(new_binary) - np.sum(old_binary):+d} pixels")
        print(f"  Skeleton detail:  {np.sum(new_skel)} (medial_axis) vs {np.sum(old_skel)} (skeletonize)")
        print(f"  Same paths:       {old_paths == new_paths} (both should produce same # paths)")
        print()

if __name__ == "__main__":
    main()
