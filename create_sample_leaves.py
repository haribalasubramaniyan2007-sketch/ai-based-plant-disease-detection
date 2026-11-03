"""
Script to generate realistic botanical leaf images for sample testing.
Creates high-definition sample leaf representations with realistic veins,
pigmentation gradients, chlorosis halos, and disease lesions using PIL & OpenCV.
"""

import os
import math
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

try:
    import cv2
except ImportError:
    cv2 = None

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static', 'images', 'sample_leaves')
os.makedirs(OUTPUT_DIR, exist_ok=True)


def draw_leaf_base(width, height, base_color=(50, 140, 45), bg_color=(245, 248, 245)):
    """Generate a high-res base canvas with background and leafy gradient."""
    img = Image.new('RGB', (width, height), bg_color)
    draw = ImageDraw.Draw(img)
    return img, draw


def create_tomato_early_blight():
    """Create a realistic tomato leaf with concentric brown target spots and yellow chlorotic halos."""
    w, h = 600, 600
    img = Image.new('RGB', (w, h), (242, 245, 242))
    draw = ImageDraw.Draw(img)

    # Stem
    draw.line([(300, 560), (300, 320)], fill=(70, 110, 45), width=8)

    # Leaf blade contour points (compound lobed leaf)
    blade_pts = [
        (300, 80), (330, 140), (370, 160), (350, 200), (410, 240),
        (380, 290), (420, 340), (370, 390), (340, 430), (300, 470),
        (260, 430), (230, 390), (180, 340), (220, 290), (190, 240),
        (250, 200), (230, 160), (270, 140), (300, 80)
    ]
    draw.polygon(blade_pts, fill=(58, 138, 52), outline=(42, 102, 38))

    # Leaf veins
    draw.line([(300, 100), (300, 470)], fill=(85, 165, 75), width=4)
    for y, vx in [(180, 70), (250, 90), (320, 100), (390, 65)]:
        draw.line([(300, y), (300 + vx, y - 35)], fill=(80, 155, 70), width=2)
        draw.line([(300, y), (300 - vx, y - 35)], fill=(80, 155, 70), width=2)

    # Concentric target lesions (Early Blight)
    lesion_centers = [(270, 220), (340, 280), (250, 340), (350, 370), (280, 410)]
    for cx, cy in lesion_centers:
        # Yellow chlorotic halo
        draw.ellipse([cx - 36, cy - 36, cx + 36, cy + 36], fill=(195, 190, 45))
        # Outer dark brown ring
        draw.ellipse([cx - 26, cy - 26, cx + 26, cy + 26], fill=(75, 45, 20))
        # Middle ring
        draw.ellipse([cx - 18, cy - 18, cx + 18, cy + 18], fill=(110, 68, 30))
        # Inner dark necrotic center
        draw.ellipse([cx - 9, cy - 9, cx + 9, cy + 9], fill=(45, 25, 12))

    img = img.filter(ImageFilter.GaussianBlur(0.8))
    img.save(os.path.join(OUTPUT_DIR, 'tomato_early_blight.jpg'), quality=92)


def create_tomato_healthy():
    """Create a healthy, vibrant green tomato leaf."""
    w, h = 600, 600
    img = Image.new('RGB', (w, h), (242, 245, 242))
    draw = ImageDraw.Draw(img)

    draw.line([(300, 560), (300, 320)], fill=(65, 125, 42), width=8)

    blade_pts = [
        (300, 70), (335, 130), (380, 155), (355, 200), (420, 240),
        (390, 290), (430, 340), (380, 390), (345, 435), (300, 480),
        (255, 435), (220, 390), (170, 340), (210, 290), (180, 240),
        (245, 200), (220, 155), (265, 130), (300, 70)
    ]
    draw.polygon(blade_pts, fill=(46, 142, 54), outline=(32, 105, 38))

    # Radiant healthy veins
    draw.line([(300, 90), (300, 480)], fill=(75, 175, 78), width=5)
    for y, vx in [(170, 75), (240, 95), (315, 105), (390, 70)]:
        draw.line([(300, y), (300 + vx, y - 40)], fill=(72, 168, 75), width=3)
        draw.line([(300, y), (300 - vx, y - 40)], fill=(72, 168, 75), width=3)

    img = img.filter(ImageFilter.GaussianBlur(0.6))
    img.save(os.path.join(OUTPUT_DIR, 'tomato_healthy.jpg'), quality=92)


def create_potato_late_blight():
    """Create a potato leaflet with large water-soaked dark brown-to-black necrotic margins."""
    w, h = 600, 600
    img = Image.new('RGB', (w, h), (242, 245, 242))
    draw = ImageDraw.Draw(img)

    draw.line([(300, 560), (300, 420)], fill=(70, 110, 45), width=7)

    # Broad oval potato leaflet
    blade_pts = [
        (300, 90), (370, 150), (420, 240), (430, 330), (400, 410),
        (350, 470), (300, 500), (250, 470), (200, 410), (170, 330),
        (180, 240), (230, 150), (300, 90)
    ]
    draw.polygon(blade_pts, fill=(62, 128, 55), outline=(45, 95, 40))

    # Veins
    draw.line([(300, 110), (300, 500)], fill=(88, 155, 75), width=4)

    # Spreading irregular Late Blight necrosis on edges
    draw.polygon([(300, 90), (370, 150), (410, 220), (330, 200), (300, 140)], fill=(40, 32, 25))
    draw.polygon([(420, 240), (430, 330), (390, 360), (360, 280)], fill=(48, 38, 28))
    draw.polygon([(170, 330), (200, 410), (260, 380), (220, 320)], fill=(38, 30, 22))

    # White fuzzy mold rim on margin
    draw.arc([310, 180, 390, 260], start=10, end=170, fill=(215, 220, 215), width=3)
    draw.arc([190, 310, 270, 390], start=30, end=190, fill=(215, 220, 215), width=3)

    img = img.filter(ImageFilter.GaussianBlur(0.8))
    img.save(os.path.join(OUTPUT_DIR, 'potato_late_blight.jpg'), quality=92)


def create_apple_scab():
    """Create an apple leaf with velvety olive-green/dark brown scab lesions."""
    w, h = 600, 600
    img = Image.new('RGB', (w, h), (242, 245, 242))
    draw = ImageDraw.Draw(img)

    draw.line([(300, 560), (300, 460)], fill=(80, 105, 45), width=6)

    # Elliptical serrated leaf
    blade_pts = [
        (300, 80), (360, 140), (400, 220), (410, 320), (380, 410),
        (340, 470), (300, 500), (260, 470), (220, 410), (190, 320),
        (200, 220), (240, 140), (300, 80)
    ]
    draw.polygon(blade_pts, fill=(52, 130, 48), outline=(38, 98, 35))

    # Main midrib
    draw.line([(300, 95), (300, 500)], fill=(82, 160, 72), width=4)

    # Scab patches
    scab_locs = [(270, 180), (340, 220), (260, 280), (330, 320), (280, 380), (350, 420)]
    for sx, sy in scab_locs:
        draw.ellipse([sx - 24, sy - 18, sx + 24, sy + 18], fill=(62, 58, 30))
        draw.ellipse([sx - 15, sy - 10, sx + 15, sy + 10], fill=(45, 40, 20))

    img = img.filter(ImageFilter.GaussianBlur(0.7))
    img.save(os.path.join(OUTPUT_DIR, 'apple_scab.jpg'), quality=92)


def create_corn_rust():
    """Create a corn leaf blade with small reddish-brown powdery rust pustules."""
    w, h = 600, 600
    img = Image.new('RGB', (w, h), (242, 245, 242))
    draw = ImageDraw.Draw(img)

    # Long corn blade
    blade_pts = [
        (300, 50), (360, 180), (380, 340), (370, 480), (300, 560),
        (230, 480), (220, 340), (240, 180), (300, 50)
    ]
    draw.polygon(blade_pts, fill=(65, 145, 45), outline=(48, 110, 32))

    # Prominent parallel venation
    draw.line([(300, 60), (300, 560)], fill=(130, 195, 95), width=6)
    for x in [260, 280, 320, 340]:
        draw.line([(x, 100), (x, 530)], fill=(85, 165, 60), width=2)

    # Cinnamon-brown rust pustules
    random.seed(42)
    for _ in range(35):
        rx = random.randint(250, 350)
        ry = random.randint(120, 500)
        draw.ellipse([rx - 6, ry - 3, rx + 6, ry + 3], fill=(168, 65, 22), outline=(115, 40, 15))

    img = img.filter(ImageFilter.GaussianBlur(0.6))
    img.save(os.path.join(OUTPUT_DIR, 'corn_rust.jpg'), quality=92)


def create_grape_black_rot():
    """Create a palmate grape leaf with reddish-brown circular spots and black pycnidia."""
    w, h = 600, 600
    img = Image.new('RGB', (w, h), (242, 245, 242))
    draw = ImageDraw.Draw(img)

    draw.line([(300, 560), (300, 440)], fill=(75, 115, 50), width=7)

    # Palmate grape leaf lobes
    blade_pts = [
        (300, 100), (360, 130), (410, 110), (440, 180), (470, 240),
        (430, 280), (450, 360), (390, 400), (330, 450), (300, 440),
        (270, 450), (210, 400), (150, 360), (170, 280), (130, 240),
        (160, 180), (190, 110), (240, 130), (300, 100)
    ]
    draw.polygon(blade_pts, fill=(50, 135, 55), outline=(35, 100, 40))

    # Palmate main veins
    for ex, ey in [(300, 115), (420, 130), (460, 245), (180, 130), (140, 245)]:
        draw.line([(300, 440), (ex, ey)], fill=(80, 160, 75), width=3)

    # Black rot spots with pycnidia specks
    rot_spots = [(260, 230), (350, 250), (240, 320), (340, 340), (290, 380)]
    for rx, ry in rot_spots:
        draw.ellipse([rx - 25, ry - 25, rx + 25, ry + 25], fill=(145, 55, 30), outline=(70, 25, 15))
        draw.ellipse([rx - 16, ry - 16, rx + 16, ry + 16], fill=(120, 42, 22))
        # Pycnidia black specks
        for dx, dy in [(-5, -4), (4, -5), (0, 3), (6, 4), (-6, 5)]:
            draw.point((rx + dx, ry + dy), fill=(15, 10, 10))

    img = img.filter(ImageFilter.GaussianBlur(0.7))
    img.save(os.path.join(OUTPUT_DIR, 'grape_black_rot.jpg'), quality=92)


if __name__ == '__main__':
    create_tomato_early_blight()
    create_tomato_healthy()
    create_potato_late_blight()
    create_apple_scab()
    create_corn_rust()
    create_grape_black_rot()
    print("Successfully generated 6 botanical leaf sample images in", OUTPUT_DIR)
