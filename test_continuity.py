"""
Test different skeletonization approaches for continuous centerline extraction.
Goal: Preserve stroke continuity while handling overlaps correctly.
"""

import os
from PIL import Image
import numpy as np
from skimage.morphology import skeletonize, medial_axis, thin, binary_erosion
from scipy.ndimage import binary_opening, binary_closing, label, distance_transform_edt
from freetype import Face, FT_LOAD_RENDER, FT_KERNING_DEFAULT

def render_text(font_path: str, text: str, size_pt: int, dpi: int):
    """Render text and return grayscale image."""
    face = Face(font_path)
    face.set_char_size(size_pt * 64, 0, dpi, 0)
    
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

def approach_1_standard_skeletonize(gray: np.ndarray):
    """Original: just standard skeletonize."""
    binary = gray > 128
    binary = binary_opening(binary, structure=np.ones((2, 2)))
    binary = binary_closing(binary, structure=np.ones((2, 2)))
    skel = skeletonize(binary)
    return skel

def approach_2_aggressive_morphology(gray: np.ndarray):
    """Try more aggressive morphology to close gaps before skeletonize."""
    binary = gray > 128
    # Dilate first to close gaps, then erode to restore size
    binary = binary_closing(binary, structure=np.ones((3, 3)))
    binary = binary_opening(binary, structure=np.ones((2, 2)))
    skel = skeletonize(binary)
    return skel

def approach_3_thin_instead_of_skeletonize(gray: np.ndarray):
    """Try thin() instead of skeletonize() - maintains more connectivity."""
    binary = gray > 128
    binary = binary_opening(binary, structure=np.ones((2, 2)))
    binary = binary_closing(binary, structure=np.ones((2, 2)))
    skel = thin(binary, max_iter=None)
    return skel

def approach_4_adaptive_threshold(gray: np.ndarray):
    """Adaptive threshold + standard skeletonize to avoid cutting through strokes."""
    hist, _ = np.histogram(gray, bins=256, range=(0, 256))
    cumsum = np.cumsum(hist)
    target_mass = cumsum[-1] * 0.7
    threshold = np.argmax(cumsum >= target_mass)
    threshold = max(100, min(180, threshold))
    
    binary = gray > threshold
    binary = binary_opening(binary, structure=np.ones((2, 2)))
    binary = binary_closing(binary, structure=np.ones((2, 2)))
    skel = skeletonize(binary)
    return skel

def approach_5_dilate_before_skeletonize(gray: np.ndarray):
    """Dilate the binary image first to merge nearby gaps, then skeletonize."""
    binary = gray > 128
    binary = binary_opening(binary, structure=np.ones((2, 2)))
    binary = binary_closing(binary, structure=np.ones((2, 2)))
    # Dilate slightly to connect broken strokes
    from scipy.ndimage import binary_dilation
    binary = binary_dilation(binary, iterations=1)
    skel = skeletonize(binary)
    return skel

def approach_6_medial_axis_no_filter(gray: np.ndarray):
    """Medial axis WITHOUT the distance filtering that was causing gaps."""
    binary = gray > 128
    binary = binary_opening(binary, structure=np.ones((2, 2)))
    binary = binary_closing(binary, structure=np.ones((2, 2)))
    skel, _ = medial_axis(binary, return_distance=True)
    return skel

def visualize_skeleton(skel: np.ndarray, label: str):
    """Save skeleton as image."""
    img = Image.fromarray((skel * 255).astype(np.uint8))
    img.save(f"skeleton_{label}.png")
    return np.sum(skel)

def main():
    test_chars = [
        ("m", "fonts/CambriaMath.ttf", 28, 600),
        ("ff", "fonts/CambriaMath.ttf", 28, 600),
    ]
    
    print("\n" + "="*80)
    print("TESTING CONTINUITY-PRESERVING SKELETONIZATION APPROACHES")
    print("="*80)
    
    for char, font_path, size_pt, dpi in test_chars:
        if not os.path.exists(font_path):
            continue
        
        print(f"\nCharacter: '{char}'")
        print("-" * 80)
        
        img = render_text(font_path, char, size_pt, dpi)
        gray = np.array(img.convert("L"), dtype=np.uint8)
        
        approaches = [
            ("1. Standard skeletonize (original)", approach_1_standard_skeletonize),
            ("2. Aggressive morphology before", approach_2_aggressive_morphology),
            ("3. Thin instead of skeletonize", approach_3_thin_instead_of_skeletonize),
            ("4. Adaptive threshold + skeletonize", approach_4_adaptive_threshold),
            ("5. Dilate then skeletonize", approach_5_dilate_before_skeletonize),
            ("6. Medial axis (no distance filter)", approach_6_medial_axis_no_filter),
        ]
        
        results = []
        for name, func in approaches:
            try:
                skel = func(gray)
                pixels = visualize_skeleton(skel, name.replace(" ", "_").replace(".", ""))
                results.append((name, pixels, True))
            except Exception as e:
                results.append((name, 0, False))
                print(f"  ✗ {name}: {str(e)[:50]}")
        
        for name, pixels, success in results:
            if success:
                print(f"  ✓ {name}: {pixels} skeleton pixels")

if __name__ == "__main__":
    main()
