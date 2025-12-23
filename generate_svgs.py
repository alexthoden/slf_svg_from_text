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
from scipy.spatial import distance

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
    """
    # Convert to grayscale array
    gray = np.array(img.convert("L"), dtype=np.uint8)

    # 1) Binarization: threshold at 128
    binary = gray > 10

    # 2) Clean noise: remove tiny specks / fill small gaps
    binary = binary_opening(binary, structure=np.ones((2, 2)))
    binary = binary_closing(binary, structure=np.ones((2, 2)))

    # 3) Skeletonize to get the centerline
    skel = skeletonize(binary)

    # Convert back to PIL 1-bit image (0/255)
    skel_img = Image.fromarray((skel * 255).astype(np.uint8), mode="L").convert("1")
    skel_img.save("debug_skel_recipient.png")

    return skel_img


def _simplify_and_smooth_path(path_points: List[tuple], opttolerance: float, alphamax: float) -> str:
    """
    Simplify and smooth a path using Ramer-Douglas-Peucker algorithm.
    Falls back to direct pixel representation if simplification fails.
    
    Args:
        path_points: List of (y, x) tuples from skeleton tracing
        opttolerance: Simplification tolerance (0.2-1.0, higher = more aggressive)
        alphamax: Corner detection threshold
    
    Returns:
        SVG path string (either simplified or direct pixel representation)
    """
    try:
        if len(path_points) < 3:
            # Too few points, use direct representation
            return _points_to_svg_path(path_points)
        
        # Convert to (x, y) for distance calculation
        xy_points = np.array([(px, py) for py, px in path_points])
        
        # Apply Ramer-Douglas-Peucker simplification
        simplified = _rdp_simplify(xy_points, opttolerance)
        
        if len(simplified) < 2:
            # Simplification too aggressive, fall back
            return _points_to_svg_path(path_points)
        
        # Convert back to (y, x) tuples
        simplified_points = [(y, x) for x, y in simplified]
        
        # Build SVG path with Line commands
        if len(simplified_points) > 0:
            path_d = f"M{simplified_points[0][1]},{simplified_points[0][0]}"
            for py, px in simplified_points[1:]:
                path_d += f"L{px},{py}"
            return path_d
        else:
            return ""
    
    except Exception as e:
        log_stage(f"Curve fitting failed ({e}), falling back to direct pixel representation")
        return _points_to_svg_path(path_points)


def _rdp_simplify(points: np.ndarray, tolerance: float) -> np.ndarray:
    """
    Ramer-Douglas-Peucker path simplification algorithm.
    Reduces number of points while maintaining path shape.
    """
    if len(points) < 3:
        return points
    
    # Find point with maximum distance from line
    dmax = 0
    index = 0
    start = points[0]
    end = points[-1]
    
    # Vector from start to end
    line_vec = end - start
    line_len = np.linalg.norm(line_vec)
    
    if line_len < 1e-10:
        # Start and end are the same, return them
        return np.array([start, end])
    
    line_unitvec = line_vec / line_len
    
    for i in range(1, len(points) - 1):
        point = points[i]
        # Vector from start to point
        point_vec = point - start
        # Project onto line
        proj_length = np.dot(point_vec, line_unitvec)
        proj = start + proj_length * line_unitvec
        # Distance from point to line
        dist = np.linalg.norm(point - proj)
        
        if dist > dmax:
            dmax = dist
            index = i
    
    # If max distance is greater than tolerance, recursively simplify
    if dmax > tolerance:
        # Recursive call
        rec1 = _rdp_simplify(points[:index + 1], tolerance)
        rec2 = _rdp_simplify(points[index:], tolerance)
        # Build result, avoiding duplicating the middle point
        result = np.vstack([rec1[:-1], rec2])
    else:
        # Just return start and end
        result = np.array([start, end])
    
    return result


def _points_to_svg_path(path_points: List[tuple]) -> str:
    """
    Convert path points directly to SVG path string (no simplification).
    Fallback when curve fitting fails.
    """
    if len(path_points) < 1:
        return ""
    
    path_d = f"M{path_points[0][1]},{path_points[0][0]}"
    for py, px in path_points[1:]:
        path_d += f"L{px},{py}"
    return path_d


def _strip_close_path_commands(path_d: str) -> str:
    """
    Remove close path commands (Z or z) from SVG path string to keep paths open.
    Also cleans up unnecessary whitespace and formatting.
    """
    # Remove Z or z commands (close path)
    path_d = re.sub(r'\s*[Zz]\s*', '', path_d)
    # Clean up extra spaces
    path_d = re.sub(r'\s+', ' ', path_d).strip()
    return path_d


def potrace_with_fallback(
    bin_img: Image.Image,
    turdsize: int,
    opttolerance: float,
    alphamax: float,
    line_label: str = "",
) -> List[str]:
    """
    Attempt to use potrace binary for vectorization, fall back to custom path tracing.
    
    Args:
        bin_img: Binary skeleton image
        turdsize: Potrace turdsize parameter
        opttolerance: Curve fitting tolerance
        alphamax: Corner detection threshold
        line_label: Label for logging
    
    Returns:
        List of SVG path strings
    """
    log_stage(f"Skeleton-to-path conversion for line: {line_label}")
    
    # Try potrace first
    potrace_path = "bin/potrace.exe"
    if os.path.exists(potrace_path):
        try:
            return _try_potrace(bin_img, potrace_path, turdsize, opttolerance, alphamax, line_label)
        except Exception as e:
            log_stage(f"Potrace failed ({e}), falling back to custom path tracing")
    else:
        log_stage(f"Potrace not found at {potrace_path}, using custom path tracing")
    
    # Fallback: use custom path tracing
    return _custom_path_tracing(bin_img, opttolerance, alphamax, line_label)


def _try_potrace(
    bin_img: Image.Image,
    potrace_exe: str,
    turdsize: int,
    opttolerance: float,
    alphamax: float,
    line_label: str,
) -> List[str]:
    """
    Use potrace binary to vectorize skeleton.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        # Save skeleton as PBM
        pbm_path = os.path.join(tmpdir, "skeleton.pbm")
        svg_path = os.path.join(tmpdir, "skeleton.svg")
        
        bin_img.save(pbm_path)
        
        # Run potrace (no -i flag to generate open paths instead of closed fills)
        cmd = [
            potrace_exe,
            "-s",  # Smooth
            f"--turdsize={turdsize}",
            f"--opttolerance={opttolerance}",
            f"--alphamax={alphamax}",
            "-o", svg_path,
            # "-i", # Uncomment to remove border. Note this seems to be the issue with open-path generation
            pbm_path,
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"Potrace failed: {result.stderr}")
        
        # Parse SVG output
        tree = ET.parse(svg_path)
        root = tree.getroot()
        
        paths = []
        ns = {"svg": "http://www.w3.org/2000/svg"}
        
        for path_elem in root.findall(".//svg:path", ns):
            d = path_elem.get("d", "")
            # Remove close path commands (Z or z) to keep paths open
            d = _strip_close_path_commands(d)
            if len(d) > 20:
                paths.append(d)
        
        log_stage(f"Potrace vectorization successful for line: {line_label} ({len(paths)} paths)")
        return paths


def _custom_path_tracing(
    bin_img: Image.Image,
    opttolerance: float,
    alphamax: float,
    line_label: str,
) -> List[str]:
    """
    Use custom smart path tracing to vectorize skeleton.
    """
    # Convert binary image to array
    skel = np.array(bin_img.convert("L"), dtype=np.uint8) > 128
    
    paths = []
    
    if np.sum(skel) > 0:
        # Label connected components in skeleton
        from scipy.ndimage import label as label_components
        labeled, num_features = label_components(skel)
        
        # For each connected component, create an SVG path
        for component_id in range(1, num_features + 1):
            component = (labeled == component_id)
            path_d = _trace_component_to_svg(component, opttolerance, alphamax)
            if len(path_d) > 20:
                paths.append(path_d)
    
    log_stage(f"Custom path tracing completed for line: {line_label} ({len(paths)} paths)")
    return paths


def potrace_bitmap_to_svg_paths(
    bin_img: Image.Image,
    turdsize: int,
    opttolerance: float,
    alphamax: float,
    line_label: str = "",
) -> List[str]:
    """
    Main entry point for skeleton-to-path conversion.
    Uses potrace with fallback to custom path tracing.
    """
    return potrace_with_fallback(bin_img, turdsize, opttolerance, alphamax, line_label)


def _find_endpoints(component: np.ndarray) -> List[tuple]:
    """
    Find endpoints (pixels with only 1 neighbor) in skeleton component.
    These are good starting points for path tracing.
    """
    endpoints = []
    pixels = np.argwhere(component)
    
    for py, px in pixels:
        neighbors = []
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dy == 0 and dx == 0:
                    continue
                ny, nx = py + dy, px + dx
                if 0 <= ny < component.shape[0] and 0 <= nx < component.shape[1]:
                    if component[ny, nx]:
                        neighbors.append((ny, nx))
        
        # Endpoint has only 1 neighbor
        if len(neighbors) == 1:
            endpoints.append((py, px))
    
    return endpoints


def _get_direction_vector(prev_pos: tuple, curr_pos: tuple) -> tuple:
    """
    Calculate direction vector from previous to current position.
    """
    py, px = curr_pos
    ppy, ppx = prev_pos
    dy = py - ppy
    dx = px - ppx
    norm = np.sqrt(dy**2 + dx**2)
    if norm == 0:
        return (0, 0)
    return (dy / norm, dx / norm)


def _select_next_neighbor(neighbors: List[tuple], curr_pos: tuple, prev_pos: tuple) -> tuple:
    """
    Select next neighbor based on direction continuity (choose neighbor closest to current direction).
    """
    if not neighbors:
        return None
    
    if len(neighbors) == 1:
        return neighbors[0]
    
    # Current direction
    curr_dir = _get_direction_vector(prev_pos, curr_pos)
    
    # Find neighbor that continues the direction best
    best_neighbor = None
    best_dot = -2.0
    
    for neighbor in neighbors:
        neighbor_dir = _get_direction_vector(curr_pos, neighbor)
        dot_product = curr_dir[0] * neighbor_dir[0] + curr_dir[1] * neighbor_dir[1]
        if dot_product > best_dot:
            best_dot = dot_product
            best_neighbor = neighbor
    
    return best_neighbor


def _trace_component_to_svg(component: np.ndarray, opttolerance: float, alphamax: float) -> str:
    """
    Trace a single connected component (skeleton) to an SVG path string using smart path following.
    Detects endpoints and traces intelligently by following direction continuity.
    """
    pixels_set = set(map(tuple, np.argwhere(component)))
    
    if len(pixels_set) == 0:
        return ""
    
    # Find endpoints or use any pixel if no clear endpoints
    endpoints = _find_endpoints(component)
    start_pixel = endpoints[0] if endpoints else list(pixels_set)[0]
    
    # Trace from start point
    path_points = [start_pixel]
    visited = {start_pixel}
    current = start_pixel
    
    while len(visited) < len(pixels_set):
        py, px = current
        neighbors = []
        
        # Get all unvisited neighbors
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dy == 0 and dx == 0:
                    continue
                ny, nx = py + dy, px + dx
                if (ny, nx) in pixels_set and (ny, nx) not in visited:
                    neighbors.append((ny, nx))
        
        if not neighbors:
            break
        
        # Choose next neighbor based on direction continuity
        if len(path_points) >= 2:
            next_pos = _select_next_neighbor(neighbors, current, path_points[-2])
        else:
            next_pos = neighbors[0]
        
        if next_pos is None:
            next_pos = neighbors[0]
        
        visited.add(next_pos)
        path_points.append(next_pos)
        current = next_pos
    
    # Convert to SVG path
    if len(path_points) > 1:
        smoothed_path = _simplify_and_smooth_path(path_points, opttolerance, alphamax)
        return smoothed_path
    
    return ""



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
    spacing_factor = 0.95  # 20% extra spacing

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

        # Position without flip (skeleton is already correctly oriented)
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
