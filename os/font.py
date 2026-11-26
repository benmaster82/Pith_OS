"""Rasterize a build-time DejaVu font into 256 Latin-1 glyphs, 8x16 pixels."""
import argparse
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

parser = argparse.ArgumentParser()
parser.add_argument("output", type=Path)
parser.add_argument("--font", default="/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf")
args = parser.parse_args()
font = ImageFont.truetype(args.font, 13)
result = bytearray()
for code in range(256):
    tile = Image.new("1", (8, 16))
    if code >= 32 and not 127 <= code < 160:
        ImageDraw.Draw(tile).text((0, -1), chr(code), font=font, fill=1)
    for y in range(16):
        result.append(sum(bool(tile.getpixel((x, y))) << (7-x) for x in range(8)))
args.output.write_bytes(result)
