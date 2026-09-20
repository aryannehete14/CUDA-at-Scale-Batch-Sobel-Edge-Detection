#!/usr/bin/env python3
"""Generates synthetic grayscale PGM images for testing the batch CUDA
Sobel edge-detection program.

No third-party dependencies (no PIL/numpy required) so it runs anywhere
Python 3 is available, including a bare lab environment.

Usage:
    python3 generate_data.py --count 200 --size 256 --out data/
"""
import argparse
import math
import os
import random


def make_image(width: int, height: int, seed: int) -> bytearray:
    """Draws a few random circles/lines on a noisy background so the
    Sobel filter has real edges to detect."""
    rng = random.Random(seed)
    pixels = bytearray(width * height)

    # Noisy background.
    for i in range(width * height):
        pixels[i] = rng.randint(0, 40)

    # A few random filled circles (clear edges for Sobel to find).
    num_shapes = rng.randint(2, 5)
    for _ in range(num_shapes):
        cx = rng.randint(0, width - 1)
        cy = rng.randint(0, height - 1)
        radius = rng.randint(10, min(width, height) // 4)
        brightness = rng.randint(180, 255)
        r2 = radius * radius
        for y in range(max(0, cy - radius), min(height, cy + radius)):
            for x in range(max(0, cx - radius), min(width, cx + radius)):
                if (x - cx) ** 2 + (y - cy) ** 2 <= r2:
                    pixels[y * width + x] = brightness

    # A diagonal stripe for extra edge content.
    stripe_offset = rng.randint(0, width)
    for y in range(height):
        x = (y + stripe_offset) % width
        for dx in range(-2, 3):
            xi = x + dx
            if 0 <= xi < width:
                pixels[y * width + xi] = 220

    return pixels


def write_pgm(path: str, width: int, height: int, pixels: bytearray) -> None:
    with open(path, "wb") as f:
        f.write(f"P5\n{width} {height}\n255\n".encode("ascii"))
        f.write(pixels)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=200,
                         help="Number of images to generate (default: 200)")
    parser.add_argument("--size", type=int, default=256,
                         help="Width/height in pixels (default: 256)")
    parser.add_argument("--out", type=str, default="data",
                         help="Output directory (default: data)")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    for i in range(args.count):
        pixels = make_image(args.size, args.size, seed=i)
        path = os.path.join(args.out, f"image_{i:04d}.pgm")
        write_pgm(path, args.size, args.size, pixels)

    print(f"Generated {args.count} synthetic {args.size}x{args.size} "
          f"PGM images in '{args.out}/'")


if __name__ == "__main__":
    main()