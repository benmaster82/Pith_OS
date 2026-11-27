"""Create a GRUB hybrid BIOS/UEFI ISO, also bootable as a raw USB disk image."""
import argparse
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("kernel", type=Path)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--options", default="mode=chat tokens=64 context=2048 threads=6")
    parser.add_argument("--title", default="LFM2.5 BF16", help="GRUB menu entry")
    parser.add_argument("--license", type=Path, action="append", default=[],
                        help="additional license file copied next to the kernel")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9 ._()+-]+", args.title):
        parser.error("The title may contain letters, digits, spaces and ._()+-")
    option = r"(?:mode=(?:chat|ids|text)|(?:tokens|context|threads)=[0-9]+|keymap=[a-z]{2})"
    if not re.fullmatch(rf"{option}(?: {option})*", args.options):
        parser.error("Options must contain only mode, tokens, context, threads and keymap assignments")
    options = dict(part.split("=", 1) for part in args.options.split())
    if not 1 <= int(options.get("tokens", 32)) <= int(options.get("context", 2048)) <= 32768:
        parser.error("Require 1 <= tokens <= context <= 32768")
    if not 1 <= int(options.get("threads", 6)) <= 16:
        parser.error("Require 1 <= threads <= 16")
    output = args.output.resolve()
    if output in (args.kernel.resolve(), args.bundle.resolve()):
        parser.error("Output must differ from kernel and bundle")
    for platform in ("i386-pc", "x86_64-efi"):
        if not Path("/usr/lib/grub", platform).is_dir():
            parser.error(f"Missing GRUB platform: {platform}")
    subprocess.run(["grub-file", "--is-x86-multiboot2", str(args.kernel)], check=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="lfm-image-") as directory:
        directory = Path(directory)
        root = directory / "root"
        boot = root / "boot"
        (boot / "grub").mkdir(parents=True)
        files = [(args.kernel, "lfm-os.elf"), (args.bundle, "lfm-boot.bundle"),
                 (Path("os/FONT-LICENSE.txt"), "FONT-LICENSE.txt")]
        files += [(path, path.name) for path in args.license]
        for source, name in files:
            try:
                os.link(source, boot / name)
            except OSError:
                shutil.copyfile(source, boot / name)
        (boot / "grub/grub.cfg").write_text(
            'set timeout=0\nset default=0\ninsmod all_video\n'
            'set gfxmode=1024x768x32,auto\nset gfxpayload=keep\n'
            f'menuentry "{args.title}" {{\n'
            f'  multiboot2 /boot/lfm-os.elf {args.options}\n'
            '  module2 /boot/lfm-boot.bundle\n  boot\n}\n')
        temporary = directory / "image.iso"
        subprocess.run(["grub-mkrescue", "-o", str(temporary), str(root)], check=True)
        # Build on the native filesystem; copying in large blocks is much faster
        # than issuing the ISO writer's small writes through a Windows mount.
        destination = None
        try:
            with tempfile.NamedTemporaryFile(dir=output.parent, delete=False) as stream:
                destination = Path(stream.name)
                with temporary.open("rb") as source:
                    shutil.copyfileobj(source, stream, 1024 * 1024)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(destination, output)
            destination = None
        finally:
            if destination is not None:
                destination.unlink(missing_ok=True)
    print(f"BIOS/UEFI hybrid image: {output} ({output.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
