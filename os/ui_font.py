"""Build the Pithos CORE console glyphs and logo from system fonts at build time.

Cells are 8x16 pixels; wide glyphs take two cells. Latin, Greek, Cyrillic and common
symbols come antialiased from DejaVu Sans Mono, the rest of the Basic Multilingual Plane
(CJK, Hangul, Arabic, Devanagari, Thai, ...) from GNU Unifont, and emoji in colour from
Noto Color Emoji, reduced to 16x16.

Layout, little endian:
  header   "PFONT001", u32 page table offset, u32 logo offset
  pages    4352 u32: offset of the page record for codepoints page*256.., or 0
  record   256 u32: glyph kind << 24 | glyph data offset, or 0 when absent
  glyphs   kind 1: 8x16 1-bit (16 bytes)    kind 2: 16x16 1-bit (32 bytes)
           kind 3: 8x16 coverage (128)      kind 4: 16x16 RGBA (1024)
  logo     u32 width, u32 height, coverage bytes
"""
import argparse
import struct
import unicodedata
from pathlib import Path

from fontTools.ttLib import TTFont
from PIL import Image, ImageDraw, ImageFont

DEJAVU = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
UNIFONT = "/usr/share/unifont/unifont.hex"
EMOJI = "/usr/share/fonts/truetype/noto/NotoColorEmoji.ttf"
# Antialiased DejaVu for these blocks; it covers them in one 8-pixel cell.
DEJAVU_BLOCKS = [(0x20, 0x24f), (0x250, 0x2ff), (0x370, 0x3ff), (0x400, 0x4ff), (0x1e00, 0x1fff),
                 (0x2000, 0x206f), (0x20a0, 0x20cf), (0x2100, 0x214f), (0x2190, 0x21ff),
                 (0x2200, 0x22ff), (0x2500, 0x25ff)]
LOGO_SIZE = 24
# Basic Multilingual Plane emoji shown in colour by default (Emoji_Presentation=Yes),
# and the red heart, which chat answers use as an emoji.
BMP_EMOJI = [(0x231a, 0x231b), (0x23e9, 0x23ec), (0x23f0, 0x23f0), (0x23f3, 0x23f3), (0x25fd, 0x25fe),
             (0x2614, 0x2615), (0x2648, 0x2653), (0x267f, 0x267f), (0x2693, 0x2693), (0x26a1, 0x26a1),
             (0x26aa, 0x26ab), (0x26bd, 0x26be), (0x26c4, 0x26c5), (0x26ce, 0x26ce), (0x26d4, 0x26d4),
             (0x26ea, 0x26ea), (0x26f2, 0x26f3), (0x26f5, 0x26f5), (0x26fa, 0x26fa), (0x26fd, 0x26fd),
             (0x2705, 0x2705), (0x270a, 0x270b), (0x2728, 0x2728), (0x274c, 0x274c), (0x274e, 0x274e),
             (0x2753, 0x2755), (0x2757, 0x2757), (0x2764, 0x2764), (0x2795, 0x2797), (0x27b0, 0x27b0),
             (0x27bf, 0x27bf), (0x2b1b, 0x2b1c), (0x2b50, 0x2b50), (0x2b55, 0x2b55)]


def zero_width(code):
    return unicodedata.category(chr(code)) in ("Mn", "Me", "Cf") or 0x1160 <= code <= 0x11ff


def dejavu_glyphs():
    cmap = TTFont(DEJAVU)["cmap"].getBestCmap()
    font = ImageFont.truetype(DEJAVU, 13)
    for low, high in DEJAVU_BLOCKS:
        for code in range(low, high + 1):
            if code in cmap and not zero_width(code) and unicodedata.east_asian_width(chr(code)) not in "WF":
                tile = Image.new("L", (8, 16))
                ImageDraw.Draw(tile).text((0, -1), chr(code), font=font, fill=255)
                yield code, 3, tile.tobytes()


def unifont_glyphs():
    for line in Path(UNIFONT).read_text(encoding="ascii").splitlines():
        code, bits = line.split(":")
        code = int(code, 16)
        if zero_width(code) or code < 0x20 or 0x7f <= code < 0xa0:
            continue
        yield code, 1 if len(bits) == 32 else 2, bytes.fromhex(bits)


def emoji_glyphs():
    cmap = TTFont(EMOJI)["cmap"].getBestCmap()
    font = ImageFont.truetype(EMOJI, 109)
    for code in sorted(cmap):
        if code < 0x1f000 and not any(low <= code <= high for low, high in BMP_EMOJI):
            continue
        canvas = Image.new("RGBA", (160, 160))
        ImageDraw.Draw(canvas).text((8, 8), chr(code), font=font, embedded_color=True)
        box = canvas.getbbox()
        if not box:
            continue
        canvas = canvas.crop(box)
        side = max(canvas.size)
        square = Image.new("RGBA", (side, side))
        square.paste(canvas, ((side - canvas.width) // 2, (side - canvas.height) // 2))
        yield code, 4, square.resize((16, 16), Image.LANCZOS).tobytes()


def logo(path):
    image = Image.open(path).convert("RGBA")
    coverage = Image.new("L", image.size)
    coverage.putdata([a * max(r, g, b) // 255 for r, g, b, a in image.getdata()])
    coverage = coverage.crop(coverage.getbbox())
    side = max(coverage.size)
    square = Image.new("L", (side, side))
    square.paste(coverage, ((side - coverage.width) // 2, (side - coverage.height) // 2))
    square = square.resize((LOGO_SIZE, LOGO_SIZE), Image.LANCZOS)
    return struct.pack("<II", LOGO_SIZE, LOGO_SIZE) + square.tobytes()


def build(logo_path):
    glyphs = {}
    for source in (unifont_glyphs(), dejavu_glyphs(), emoji_glyphs()):
        for code, kind, data in source:
            glyphs[code] = (kind, data)    # later sources take precedence
    pages = sorted({code >> 8 for code in glyphs})
    header = 16
    table = header
    records = table + 4352 * 4
    data_start = records + len(pages) * 1024
    page_table = [0] * 4352
    record_bytes = bytearray(len(pages) * 1024)
    data = bytearray()
    for index, page in enumerate(pages):
        page_table[page] = records + index * 1024
        for low in range(256):
            code = page * 256 + low
            if code in glyphs:
                kind, bits = glyphs[code]
                offset = data_start + len(data)
                if offset >= 1 << 24:
                    raise ValueError("Glyph data exceeds 16 MiB")
                struct.pack_into("<I", record_bytes, index * 1024 + low * 4, kind << 24 | offset)
                data.extend(bits)
    logo_offset = data_start + len(data)
    output = bytearray(b"PFONT001" + struct.pack("<II", table, logo_offset))
    output += struct.pack("<4352I", *page_table) + record_bytes + data + logo(logo_path)
    kinds = [kind for kind, _ in glyphs.values()]
    return output, {kind: kinds.count(kind) for kind in (1, 2, 3, 4)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("logo", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    output, counts = build(args.logo)
    args.output.write_bytes(output)
    print(f"Console glyphs: {len(output):,} bytes; {counts[3]} antialiased, {counts[1] + counts[2]} Unifont "
          f"({counts[2]} wide), {counts[4]} colour emoji; {LOGO_SIZE}x{LOGO_SIZE} logo")
