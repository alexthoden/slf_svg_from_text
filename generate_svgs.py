# generate_svgs.py

import csv
import os
import subprocess
import tempfile
from dataclasses import dataclass
from typing import List

import yaml
import numpy as np
from PIL import Image
import svgwrite
from skimage.morphology import skeletonize, medial_axis, thin
from freetype import Face, FT_LOAD_DEFAULT, FT_LOAD_RENDER, FT_KERNING_DEFAULT
from fontTools.ttLib import TTCollection
import time
from scipy.ndimage import binary_opening, binary_closing, binary_dilation, label, maximum_filter
import xml.etree.ElementTree as ET
import re

def measure_path_height(d_list):
    ys = []

    for d in d_list:
        # Extract all numbers from the path
        nums = re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", d)
        coords = list(map(float, nums))

        # Y values are every second number: (x0, y0, x1, y1, ...)
        ys.extend(coords[1::2])

    if not ys:
        return 0

    return max(ys) - min(ys)

def measure_path_bounds(d_list):
    xs, ys = [], []

    for d in d_list:
        nums = re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", d)
        coords = list(map(float, nums))

        xs.extend(coords[0::2])
        ys.extend(coords[1::2])

    if not xs or not ys:
        return (0, 0, 0, 0)

    return (min(xs), min(ys), max(xs), max(ys))

def log_stage(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}")


# -----------------------------
# TTC extraction for Cambria Math
# -----------------------------

def extract_ttc_face(ttc_path: str, face_name: str, output_path: str):
    """
    Extract a specific font face from a TTC file and save as TTF.
    """
    collection = TTCollection(ttc_path)
    for font in collection.fonts:
        name_table = font["name"]
        family_names = [record.toUnicode() for record in name_table.names if record.nameID in (1, 4)]
        if any(face_name in fam for fam in family_names):
            font.save(output_path)
            print(f"Extracted {face_name} to {output_path}")
            return output_path
    raise ValueError(f"Face '{face_name}' not found in {ttc_path}")

# -----------------------------
# Config data structures
# -----------------------------

@dataclass
class CanvasConfig:
    width_mm: float
    height_mm: float
    margin_mm: float
    dpi: int
    stroke_width_mm: float
    line_alignment: str

@dataclass
class OutputConfig:
    dir: str
    filename_prefix: str

@dataclass
class Config:
    canvas: CanvasConfig
    line_styles: dict
    output: OutputConfig
    potrace: dict


# -----------------------------
# Helpers
# -----------------------------

def mm_to_px(mm: float, dpi: int) -> int:
    return int(round(mm * dpi / 25.4))

def format_address_lines(row: dict) -> List[str]:
    lines = [row["recipient_name"], row["address_line1"]]
    if row.get("address_line2", "").strip():
        lines.append(row["address_line2"])
    lines.append(f"{row['city']}, {row['state']} {row['zip']}")
    return lines

# -----------------------------
# FreeType rendering
# -----------------------------

class FreeTypeTextRenderer:
    def __init__(self, font_path: str, size_pt: int, dpi: int):
        self.face = Face(font_path)
        self.face.set_char_size(size_pt * 64, 0, dpi, 0)

    def render_line(self, text: str, extra_letter_spacing_px: int = 0) -> Image.Image:
        pen_x = 0
        prev_char_index = None
        max_above_baseline = 0
        max_below_baseline = 0

        for ch in text:
            self.face.load_char(ch, FT_LOAD_RENDER) 
            slot = self.face.glyph 
            bitmap = slot.bitmap # Grayscale glyph image 
            glyph_img = Image.frombytes( "L", (bitmap.width, bitmap.rows), bytes(bitmap.buffer),)

            slot = self.face.glyph
            if prev_char_index is not None:
                kerning = self.face.get_kerning(prev_char_index, self.face.get_char_index(ch), FT_KERNING_DEFAULT)
                pen_x += kerning.x >> 6
            pen_x += (slot.advance.x >> 6) + extra_letter_spacing_px
            metrics = slot.metrics
            above = metrics.horiBearingY >> 6
            below = (metrics.height >> 6) - above
            max_above_baseline = max(max_above_baseline, above)
            max_below_baseline = max(max_below_baseline, below)
            prev_char_index = self.face.get_char_index(ch)

        width = max(1, pen_x)
        height = max(1, max_above_baseline + max_below_baseline)
        img = Image.new("L", (width, height), color=0)
        pen_x = 0
        prev_char_index = None

        for ch in text:
            self.face.load_char(ch, FT_LOAD_RENDER)
            slot = self.face.glyph
            if prev_char_index is not None:
                kerning = self.face.get_kerning(prev_char_index, self.face.get_char_index(ch), FT_KERNING_DEFAULT)
                pen_x += kerning.x >> 6
            bitmap = slot.bitmap
            top = slot.bitmap_top
            left = slot.bitmap_left
            if bitmap.width and bitmap.rows:
                glyph_img = Image.frombytes("L", (bitmap.width, bitmap.rows), bytes(bitmap.buffer))
                y = max_above_baseline - top
                x = pen_x + left
                img.paste(glyph_img, (x, y), glyph_img)
            pen_x += (slot.advance.x >> 6) + extra_letter_spacing_px
            prev_char_index = self.face.get_char_index(ch)
        return img

# -----------------------------
# Skeletonization + Potrace
# -----------------------------

def skeletonize_binary_image(img: Image.Image) -> Image.Image:
    """
    Take a grayscale (L) or 1-bit image of text,
    binarize it cleanly, denoise it, and produce a 1-pixel skeleton (centerline).
    
    Key insight: For stroke centerline extraction, we need:
    1. Proper binarization that doesn't cut through strokes
    2. Standard skeletonize to extract true centerlines
    3. Minimal post-processing to avoid breaking continuity
    """
    # Convert to grayscale array
    gray = np.array(img.convert("L"), dtype=np.uint8)

    # 1) Binarization: Use standard threshold
    # The key is to use a threshold that captures the main stroke body
    # without including anti-aliasing halos around edges.
    # A threshold around 128 works well for most fonts at 600 DPI.
    binary = gray > 128

    # 2) Clean noise: remove tiny specks / fill small gaps
    # Use MINIMAL morphology - only close obvious gaps and remove noise
    binary = binary_opening(binary, structure=np.ones((2, 2)))
    binary = binary_closing(binary, structure=np.ones((2, 2)))

    # 3) Skeletonize to get the centerline
    # Standard skeletonization provides true 1-pixel wide centerlines
    # through the middle of each stroke, which is exactly what we want.
    skel = skeletonize(binary)

    # Convert back to PIL 1-bit image (0/255)
    skel_img = Image.fromarray((skel * 255).astype(np.uint8), mode="L").convert("1")
    skel_img.save("debug_skel_recipient.png")

    return skel_img


def potrace_bitmap_to_svg_paths(
    bin_img: Image.Image,
    turdsize: int,
    opttolerance: float,
    alphamax: float,
    line_label: str = "",
) -> List[str]:
    """
    Vectorize a binary skeleton image into SVG paths using Potrace.
    Expects a clean 1-bit image (mode "1") as input.
    
    Parameters are tuned to avoid artifacts at intersection points:
    - Lower turdsize: keeps small strokes (important for intersections)
    - Moderate opttolerance: balances curve smoothing with geometry preservation
    - Adjusted alphamax: controls corner sharpness (lower = sharper, better for overlaps)
    """
    log_stage(f"Potrace starting for line: {line_label}")

    with tempfile.TemporaryDirectory() as tmpdir:
        pbm_path = os.path.join(tmpdir, "input.pbm")
        svg_path = os.path.join(tmpdir, "output.svg")

        # Ensure 1-bit, then save as PBM (Pillow infers from extension)
        pbm_img = bin_img.convert("1")
        pbm_img.save(pbm_path)

        potrace_exe = os.path.join(os.path.dirname(__file__), "bin", "potrace.exe")
        
        # Use optimized parameters:
        # - turdsize: 0 keeps ALL small features (essential for correct overlaps)
        # - opttolerance: 0.4 provides good curve smoothing without losing detail
        # - alphamax: 0.8 preserves corners better at intersections
        cmd = [
            potrace_exe,
            os.path.abspath(pbm_path),         # INPUT bitmap
            "-s",
            "-o", os.path.abspath(svg_path),   # OUTPUT SVG
            "-i",
            "--turdsize", str(turdsize),
            "--opttolerance", str(opttolerance),
            "--alphamax", str(alphamax),
        ]

        log_stage(f"Running command: {' '.join(cmd)}")
        result = subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        print("[Potrace stdout]", result.stdout)
        print("[Potrace stderr]", result.stderr)

        # Parse SVG and extract all path 'd' attributes
        tree = ET.parse(svg_path)
        root = tree.getroot()
        nsmap = {"svg": "http://www.w3.org/2000/svg"}
        paths = [elem.attrib["d"] for elem in root.findall(".//svg:path", nsmap)]
        
        # Post-process: filter out very small paths that might be artifacts
        # These can occur at intersection points due to pixelation
        paths = [p for p in paths if len(p) > 20]  # Keep only meaningful paths

    log_stage(f"Potrace completed for line: {line_label}")
    return paths



# -----------------------------
# Layout + SVG assembly
# -----------------------------

def layout_lines_as_paths(lines: List[str], config: Config):
    line_paths, line_widths, line_heights = [], [], []
    key_order = ["recipient_name", "address_line1", "address_line2", "city_state_zip"]

    for i, text in enumerate(lines):
        key = key_order[i]
        style = config.line_styles[key]
        print(f"[Render] Processing line '{key}': \"{text}\"")

        renderer = FreeTypeTextRenderer(
            style["font_path"],
            int(style["size_pt"]),
            config.canvas.dpi,
        )
        line_img = renderer.render_line(text, int(style.get("letter_spacing_px", 0)))
        print(f"  [Render] FreeType rendering complete for line '{key}'")

        skel_img = skeletonize_binary_image(line_img)
        print(f"  [Skeleton] Skeletonization complete for line '{key}'")
        # Optional: debug image
        # skel_img.save(f"debug_skel_{key}.png")

        d_list = potrace_bitmap_to_svg_paths(
            skel_img,
            turdsize=config.potrace["turdsize"],
            opttolerance=config.potrace["opttolerance"],
            alphamax=config.potrace["alphamax"],
            line_label=key,
        )

        line_paths.append(d_list)
        line_widths.append(line_img.size[0])   # in pixels
        line_heights.append(line_img.size[1])  # in pixels

    return line_paths, line_widths, line_heights


def mm_to_px(mm: float, dpi: int) -> float:
    return mm * dpi / 25.4

def assemble_svg(
    lines,
    line_paths,
    line_widths_px,
    line_heights_px,
    config,
    output_file
):
    dpi = config.canvas.dpi

    # --- 1. Measure all line bounding boxes ---
    line_bounds = []
    for d_list in line_paths:
        min_x, min_y, max_x, max_y = measure_path_bounds(d_list)
        width = max_x - min_x
        height = max_y - min_y
        line_bounds.append((width, height))

    # --- 2. Compute total height dynamically ---
    total_height = 0
    spacing_factor = 0.80  # 20% extra spacing

    for width, height in line_bounds:
        total_height += height * spacing_factor

    # --- 3. Compute max width dynamically ---
    max_width = max(width for width, height in line_bounds)

    # --- 4. Add margins (in pixels) ---
    margin_px = mm_to_px(config.canvas.margin_mm, dpi)
    svg_width = max_width + margin_px * 2
    svg_height = total_height + margin_px * 2

    # --- 5. Create SVG with dynamic viewBox ---
    dwg = svgwrite.Drawing(
        output_file,
        size=(f"{svg_width}px", f"{svg_height}px"),
        viewBox=f"0 0 {svg_width} {svg_height}",
    )

    # --- 6. Place each line centered horizontally ---
    current_y = margin_px

    for i, d_list in enumerate(line_paths):
        width, height = line_bounds[i]

        # Center horizontally
        x_offset = (svg_width - width) / 2

        # Flip vertically and position
        g = dwg.g(
            transform=(
                f"translate({x_offset},{current_y}) "
                f"scale(1,-1) "
                f"translate(0,{-height})"
            )
        )

        for d in d_list:
            g.add(
                dwg.path(
                    d=d,
                    fill="none",
                    stroke="red",
                    stroke_width=f"{config.canvas.stroke_width_mm}mm",
                )
            )

        dwg.add(g)

        # Advance Y by height + 20%
        current_y += height * spacing_factor

    dwg.save()


# -----------------------------
# Config loader
# -----------------------------

def load_config(path: str) -> Config:
    with open(path, "r") as f:
        data = yaml.safe_load(f)

    canvas = CanvasConfig(**data["canvas"])
    output = OutputConfig(**data["output"])
    potrace = data.get("potrace", {"turdsize": 0, "opttolerance": 0.2, "alphamax": 1.0})

    return Config(canvas=canvas,
                  line_styles=data["line_styles"],
                  output=output,
                  potrace=potrace)

# -----------------------------
# Main
# -----------------------------

def main():
    CONFIG_FILE = "config.yaml"
    CSV_FILE = "addresses.csv"

    # Ensure Cambria Math is extracted from TTC
    ttc_path = "fonts/cambria.ttc"
    ttf_output = "fonts/CambriaMath.ttf"
    if not os.path.exists(ttf_output):
        extract_ttc_face(ttc_path, "Cambria Math", ttf_output)

    config = load_config(CONFIG_FILE)
    os.makedirs(config.output.dir, exist_ok=True)

    with open(CSV_FILE, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        total = len(rows)

        for idx, row in enumerate(rows, start=1):
            print(f"\n[Progress] Address {idx}/{total}: {row['recipient_name']}")
            lines = format_address_lines(row)
            line_paths, line_widths_px, line_heights_px = layout_lines_as_paths(lines, config)

            filename = os.path.join(config.output.dir, f"{config.output.filename_prefix}{idx:03}.svg")
            assemble_svg(lines, line_paths, line_widths_px, line_heights_px, config, filename)
            print(f"[Progress] Completed {filename}")


if __name__ == "__main__":
    main()
