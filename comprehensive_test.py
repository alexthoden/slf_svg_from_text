"""
Advanced diagnostics to understand intersection artifacts and test solutions.
"""

import os
import tempfile
import subprocess
from PIL import Image
import numpy as np
from skimage.morphology import skeletonize, medial_axis, thin
from scipy.ndimage import binary_opening, binary_closing, binary_dilation
from freetype import Face, FT_LOAD_RENDER, FT_KERNING_DEFAULT
import xml.etree.ElementTree as ET

def render_text_freetype(font_path: str, text: str, size_pt: int, dpi: int, padding: int = 0):
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
    
    width = max(1, pen_x + padding * 2)
    height = max(1, max_above_baseline + max_below_baseline + padding * 2)
    img = Image.new("L", (width, height), color=0)
    
    # Second pass: render glyphs
    pen_x = padding
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
            y = max_above_baseline - top + padding
            x = pen_x + left
            img.paste(glyph_img, (x, y), glyph_img)
        
        pen_x += (slot.advance.x >> 6)
        prev_char_index = face.get_char_index(ch)
    
    return img, width, height

def test_solution_combination(test_name: str, binary: np.ndarray, morphology: str, skeleton_method: str, 
                             extra_dilation: int = 0, potrace_params: dict = None):
    """Test a specific combination of preprocessing and vectorization."""
    print(f"\n  Testing: {test_name}")
    
    # Apply morphological operations
    if morphology == "aggressive":
        # Aggressive: larger kernel to close gaps
        test_binary = binary_opening(binary, structure=np.ones((3, 3)))
        test_binary = binary_closing(test_binary, structure=np.ones((3, 3)))
    elif morphology == "conservative":
        # Conservative: smaller kernel to preserve detail
        test_binary = binary_opening(binary, structure=np.ones((2, 2)))
        test_binary = binary_closing(binary, structure=np.ones((2, 2)))
    else:
        test_binary = binary.copy()
    
    # Apply dilation if needed
    if extra_dilation > 0:
        test_binary = binary_dilation(test_binary, iterations=extra_dilation)
    
    # Skeletonize
    if skeleton_method == "skeletonize":
        skel = skeletonize(test_binary)
    elif skeleton_method == "medial_axis":
        skel, dist = medial_axis(test_binary, return_distance=True)
    elif skeleton_method == "thin":
        skel = thin(test_binary)
    else:
        skel = test_binary
    
    # Try potrace
    skel_img = Image.fromarray((skel * 255).astype(np.uint8), mode="L")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        pbm_path = os.path.join(tmpdir, "test.pbm")
        svg_path = os.path.join(tmpdir, "test.svg")
        
        pbm_img = skel_img.convert("1")
        pbm_img.save(pbm_path)
        
        potrace_exe = os.path.join(os.path.dirname(__file__), "bin", "potrace.exe")
        
        # Default potrace params
        turdsize = potrace_params.get('turdsize', 0) if potrace_params else 0
        opttolerance = potrace_params.get('opttolerance', 0.5) if potrace_params else 0.5
        alphamax = potrace_params.get('alphamax', 0.5) if potrace_params else 0.5
        
        try:
            result = subprocess.run([
                potrace_exe,
                pbm_path,
                "-s",
                "-o", svg_path,
                "-i",
                "--turdsize", str(turdsize),
                "--opttolerance", str(opttolerance),
                "--alphamax", str(alphamax),
            ], capture_output=True, text=True, check=True)
            
            # Count paths in output
            tree = ET.parse(svg_path)
            root = tree.getroot()
            nsmap = {"svg": "http://www.w3.org/2000/svg"}
            paths = root.findall(".//svg:path", nsmap)
            
            # Measure skeleton quality
            skel_pixels = np.sum(skel)
            orig_pixels = np.sum(test_binary)
            
            print(f"    ✓ Potrace succeeded, {len(paths)} paths")
            print(f"      Skeleton: {skel_pixels} pixels from {orig_pixels} ({skel_pixels/orig_pixels*100:.1f}%)")
            return True, len(paths), skel_pixels
            
        except Exception as e:
            print(f"    ✗ Potrace failed: {str(e)[:60]}")
            return False, 0, 0

def comprehensive_test(text: str, font_path: str, size_pt: int, dpi: int):
    """Test various combinations."""
    print(f"\n{'='*70}")
    print(f"Comprehensive test: '{text}' in {os.path.basename(font_path)}")
    print(f"{'='*70}")
    
    # Render with padding
    img, w, h = render_text_freetype(font_path, text, size_pt, dpi, padding=10)
    img.save(f"comprehensive_{text}.png")
    
    gray = np.array(img.convert("L"), dtype=np.uint8)
    binary = gray > 128
    
    # Test combinations
    test_params = [
        ("Conservative morphology + Skeletonize", "conservative", "skeletonize", 0, None),
        ("Conservative morphology + Medial axis", "conservative", "medial_axis", 0, None),
        ("Conservative morphology + Thin", "conservative", "thin", 0, None),
        ("Aggressive morphology + Skeletonize", "aggressive", "skeletonize", 0, None),
        ("Conservative + Dilation(1) + Skeletonize", "conservative", "skeletonize", 1, None),
        ("Conservative + Skeletonize + Lower opttol", "conservative", "skeletonize", 0, 
         {"turdsize": 0, "opttolerance": 0.2, "alphamax": 0.5}),
        ("Conservative + Skeletonize + Higher alphamax", "conservative", "skeletonize", 0, 
         {"turdsize": 0, "opttolerance": 0.5, "alphamax": 1.0}),
    ]
    
    results = []
    for test_name, morph, skel, dil, ptrace_params in test_params:
        success, path_count, skel_pixels = test_solution_combination(
            test_name, binary, morph, skel, dil, ptrace_params
        )
        results.append((test_name, success, path_count, skel_pixels))
    
    return results

def main():
    # Test problematic cases
    test_cases = [
        ("m", "fonts/CambriaMath.ttf", 28, 600),  # Common overlap
        ("ff", "fonts/CambriaMath.ttf", 28, 600),  # Ligature
        ("&", "fonts/CambriaMath.ttf", 28, 600),  # Complex shape
    ]
    
    all_results = {}
    for text, font_path, size_pt, dpi in test_cases:
        if os.path.exists(font_path):
            results = comprehensive_test(text, font_path, size_pt, dpi)
            all_results[text] = results
    
    # Summary
    print(f"\n\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    for char, results in all_results.items():
        print(f"\nCharacter '{char}':")
        for test_name, success, path_count, skel_pixels in results:
            status = "✓" if success else "✗"
            print(f"  {status} {test_name}: {path_count} paths, {skel_pixels} skel pixels")

if __name__ == "__main__":
    main()
