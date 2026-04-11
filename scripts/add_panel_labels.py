"""
Add panel labels (A-F, G-L) to Figures 1 and 2 for Frontiers revision.
Also fix axis labels where needed.
"""

from PIL import Image, ImageDraw, ImageFont
import os

IMG_DIR = '/Users/neurokinetikz/Code/research/papers/images'


def add_labels_to_grid(input_path, output_path, labels, ncols=2, nrows=3):
    """Add panel labels to a grid figure."""
    img = Image.open(input_path)
    draw = ImageDraw.Draw(img)
    w, h = img.size

    # Try to get a bold font; fall back to default
    font_size = int(min(w, h) * 0.03)
    try:
        font = ImageFont.truetype('/System/Library/Fonts/Helvetica.ttc', font_size)
    except (IOError, OSError):
        try:
            font = ImageFont.truetype('/System/Library/Fonts/HelveticaNeue.ttc', font_size)
        except (IOError, OSError):
            font = ImageFont.load_default()

    # Calculate panel positions (approximate grid cells)
    # Skip top ~8% for the title area
    title_frac = 0.08
    bottom_frac = 0.04
    y_start = int(h * title_frac)
    y_end = int(h * (1 - bottom_frac))
    cell_h = (y_end - y_start) / nrows
    cell_w = w / ncols

    for idx, label in enumerate(labels):
        row = idx // ncols
        col = idx % ncols
        x = int(col * cell_w + cell_w * 0.02)
        y = int(y_start + row * cell_h + cell_h * 0.02)

        # White background box for readability
        bbox = font.getbbox(label)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        padding = 4
        draw.rectangle(
            [x - padding, y - padding, x + tw + padding, y + th + padding],
            fill='white', outline='black', width=1
        )
        draw.text((x, y), label, fill='black', font=font)

    img.save(output_path, dpi=(300, 300))
    print(f"Saved {output_path}")


# Figure 1: panels A-F
add_labels_to_grid(
    os.path.join(IMG_DIR, 'download (5).png'),
    os.path.join(IMG_DIR, 'figure1_labeled.png'),
    ['A', 'B', 'C', 'D', 'E', 'F']
)

# Figure 2: panels G-L
add_labels_to_grid(
    os.path.join(IMG_DIR, 'download (6).png'),
    os.path.join(IMG_DIR, 'figure2_labeled.png'),
    ['G', 'H', 'I', 'J', 'K', 'L']
)
