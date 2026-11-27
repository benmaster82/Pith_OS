"""Generate docs/banner.svg: the Pithos mark and text outlined from DejaVu, so the banner
renders the same everywhere without fonts. Needs fontTools and the DejaVu fonts."""
import argparse
import re
from pathlib import Path

from fontTools.pens.svgPathPen import SVGPathPen
from fontTools.pens.transformPen import TransformPen
from fontTools.ttLib import TTFont

FONTS = Path("/usr/share/fonts/truetype/dejavu")
WIDTH, HEIGHT = 1280, 400
TEXT_X, RIGHT = 340, 1184


def outline(font_file, text, size, x, baseline, max_width=None):
    """SVG path data for text set at (x, baseline); returns the path and its advance."""
    font = TTFont(FONTS / font_file)
    cmap, glyphs, metrics = font.getBestCmap(), font.getGlyphSet(), font["hmtx"].metrics
    scale = size / font["head"].unitsPerEm
    advance = sum(metrics[cmap[ord(char)]][0] for char in text) * scale
    if max_width and advance > max_width:
        scale *= max_width / advance
        advance = max_width
    pen = SVGPathPen(glyphs)
    cursor = 0.0
    for char in text:
        name = cmap[ord(char)]
        glyphs[name].draw(TransformPen(pen, (scale, 0, 0, -scale, x + cursor, baseline)))
        cursor += metrics[name][0] * scale
    data = re.sub(r"(\d+\.\d{2})\d+", r"\1", pen.getCommands())
    return data, advance


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mark", type=Path, help="the Pithos mark, a 100x100 SVG path")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    mark = re.search(r' d="([^"]+)"', args.mark.read_text()).group(1)
    pithos, pithos_width = outline("DejaVuSans-Bold.ttf", "Pithos", 92, TEXT_X, 200)
    core, core_width = outline("DejaVuSans-Bold.ttf", "CORE", 92, TEXT_X + pithos_width + 26, 200)
    subtitle, _ = outline("DejaVuSans.ttf", "Language models in x86-64 assembly, on Linux and bare metal",
                          27, TEXT_X, 300, RIGHT - TEXT_X)
    details, _ = outline("DejaVuSansMono.ttf", "LFM3-1.7B · LFM2.5-350M · BIOS/UEFI · up to 16 CPUs · no libc",
                         19, TEXT_X, 344, RIGHT - TEXT_X)
    trail_end = TEXT_X + pithos_width + 26 + core_width
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {WIDTH} {HEIGHT}" width="{WIDTH}" height="{HEIGHT}" role="img" aria-label="Pithos CORE: language models in x86-64 assembly, on Linux and bare metal">
  <defs>
    <linearGradient id="trail" gradientUnits="userSpaceOnUse" x1="{TEXT_X}" y1="0" x2="{trail_end:.1f}" y2="0"><stop offset="0" stop-color="#7c5cf6" stop-opacity="0"/><stop offset=".35" stop-color="#7c5cf6"/><stop offset=".7" stop-color="#ec5c8c"/><stop offset="1" stop-color="#ffc460"/></linearGradient>
    <linearGradient id="core" gradientUnits="userSpaceOnUse" x1="{TEXT_X + pithos_width + 26:.1f}" y1="0" x2="{trail_end:.1f}" y2="0"><stop offset="0" stop-color="#fbbf24"/><stop offset="1" stop-color="#ffc460"/></linearGradient>
    <pattern id="dots" width="24" height="24" patternUnits="userSpaceOnUse"><circle cx="1" cy="1" r="1" fill="#ffffff" fill-opacity=".045"/></pattern>
  </defs>
  <rect width="{WIDTH}" height="{HEIGHT}" rx="22" fill="#08090b"/>
  <rect width="{WIDTH}" height="{HEIGHT}" rx="22" fill="url(#dots)"/>
  <rect x=".5" y=".5" width="{WIDTH - 1}" height="{HEIGHT - 1}" rx="21.5" fill="none" stroke="#ffffff" stroke-opacity=".08"/>
  <line x1="{TEXT_X}" y1="246" x2="{RIGHT}" y2="246" stroke="#ffffff" stroke-opacity=".14"/>
  <line x1="{TEXT_X}" y1="246" x2="{trail_end:.1f}" y2="246" stroke="url(#trail)" stroke-width="2">
    <animate attributeName="x2" from="{TEXT_X}" to="{trail_end:.1f}" dur="1.4s" calcMode="spline" keySplines=".2 .8 .2 1" fill="freeze"/>
  </line>
  <g transform="translate(110 112) scale(1.76)">
    <path fill="#ffffff" fill-rule="evenodd" d="{mark}"/>
  </g>
  <path fill="#ffffff" d="{pithos}"/>
  <path fill="url(#core)" d="{core}"/>
  <path fill="#8b949e" d="{subtitle}"/>
  <path fill="#5c646d" d="{details}"/>
</svg>
"""
    args.output.write_text(svg, encoding="ascii")
    print(f"Banner: {len(svg):,} bytes")


if __name__ == "__main__":
    main()
