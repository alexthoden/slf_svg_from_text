"""
Diagnostic script to identify sources of geometry issues in text vectorization.
"""

import os
import tempfile
import subprocess
from PIL import Image
import numpy as np
from skimage.morphology import skeletonize
from scipy.ndimage import binary_opening, binary_closing
from freetype import Face, FT_LOAD_RENDER, FT_KERNING_DEFAULT
import yaml

def load_config(path: str):
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    return data

def render_text_freetype(font_path: str, text: str, size_pt: int, dpi: int):
    """Render text with FreeType and return image with metadata."""
    face = Face(font_path)
    face.set_char_size(size_pt * 64, 0, dpi, 0)
    
    # First pass: measure dimensions
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
    
    # Second pass: render glyphs
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
    
    return img, width, height

def test_binarization_thresholds(img: Image.Image):
    """Test different binarization thresholds to see which works best."""
    gray = np.array(img.convert("L"), dtype=np.uint8)
    
    thresholds = [100, 128, 150, 180, 200]
    results = {}
    
    for threshold in thresholds:
        binary = gray > threshold
        # Apply morphological operations
        binary = binary_opening(binary, structure=np.ones((2, 2)))
        binary = binary_closing(binary, structure=np.ones((2, 2)))
        
        # Count foreground pixels
        fg_count = np.sum(binary)
        
        results[threshold] = {
            'binary': binary,
            'fg_count': fg_count
        }
    
    return results, gray

def test_skeletonization_methods(binary_img: np.ndarray):
    """Test skeletonization and show coverage."""
    skel = skeletonize(binary_img)
    
    # Calculate metrics
    original_pixels = np.sum(binary_img)
    skeleton_pixels = np.sum(skel)
    coverage_ratio = skeleton_pixels / original_pixels if original_pixels > 0 else 0
    
    return skel, coverage_ratio

def analyze_font_character(font_path: str, char: str, size_pt: int, dpi: int):
    """Detailed analysis of a single character."""
    print(f"\n{'='*60}")
    print(f"Analyzing character: '{char}' in {os.path.basename(font_path)} at {size_pt}pt, {dpi}dpi")
    print(f"{'='*60}")
    
    # Render text
    img, w, h = render_text_freetype(font_path, char, size_pt, dpi)
    img.save(f"diag_freetype_{char}.png")
    print(f"✓ Rendered text image: {w}x{h}px")
    
    # Test binarization thresholds
    results, gray = test_binarization_thresholds(img)
    print(f"\nBinarization threshold analysis:")
    for threshold, data in results.items():
        ratio = (data['fg_count'] / (gray.size)) * 100
        print(f"  Threshold {threshold}: {data['fg_count']} foreground pixels ({ratio:.1f}%)")
        
        # Save best result (threshold 128)
        if threshold == 128:
            skel, coverage = test_skeletonization_methods(data['binary'])
            print(f"\nSkeletonization (threshold 128):")
            print(f"  Original foreground pixels: {data['fg_count']}")
            print(f"  Skeleton pixels: {np.sum(skel)}")
            print(f"  Coverage ratio: {coverage:.2%}")
            
            skel_img = Image.fromarray((skel * 255).astype(np.uint8), mode="L")
            skel_img.save(f"diag_skeleton_{char}.png")
            
            # Test potrace with this
            with tempfile.TemporaryDirectory() as tmpdir:
                pbm_path = os.path.join(tmpdir, "test.pbm")
                svg_path = os.path.join(tmpdir, "test.svg")
                
                pbm_img = skel_img.convert("1")
                pbm_img.save(pbm_path)
                
                potrace_exe = os.path.join(os.path.dirname(__file__), "bin", "potrace.exe")
                try:
                    result = subprocess.run([
                        potrace_exe,
                        pbm_path,
                        "-s",
                        "-o", svg_path,
                        "-i",
                        "--turdsize", "0",
                        "--opttolerance", "0.5",
                        "--alphamax", "0.5",
                    ], capture_output=True, text=True, check=True)
                    print(f"✓ Potrace succeeded")
                except Exception as e:
                    print(f"✗ Potrace failed: {e}")

def main():
    config = load_config("config.yaml")
    
    # Test with problematic fonts/characters
    test_cases = [
        ("fonts/Hey Beauty.otf", "A", 36, config['canvas']['dpi']),
        ("fonts/CambriaMath.ttf", "a", 28, config['canvas']['dpi']),
        ("fonts/CambriaMath.ttf", "m", 28, config['canvas']['dpi']),  # Common overlap issue
        ("fonts/CambriaMath.ttf", "ff", 28, config['canvas']['dpi']),  # Ligatures
    ]
    
    for font_path, text, size_pt, dpi in test_cases:
        if os.path.exists(font_path):
            analyze_font_character(font_path, text, size_pt, dpi)
        else:
            print(f"Font not found: {font_path}")

if __name__ == "__main__":
    main()
