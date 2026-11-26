"""Run the kernel or hybrid image, normalizing QEMU test exit status."""
import argparse
import os
import subprocess
from pathlib import Path


def base_command(accel="kvm", memory="2G", cpus=1, display="none"):
    return ["qemu-system-x86_64", "-accel", accel, "-cpu", "host" if accel == "kvm" else "max",
            "-m", memory, "-smp", str(cpus), "-display", display, "-monitor", "none", "-serial", "stdio",
            "-nic", "none", "-no-reboot", "-device", "isa-debug-exit,iobase=0xf4,iosize=4"]


def command(kernel, bundle, options, accel="kvm", memory="2G", cpus=1, display="none"):
    return base_command(accel, memory, cpus, display) + [
        "-kernel", str(kernel), "-initrd", str(bundle), "-append", options]


def media_command(path, accel="kvm", memory="2G", cpus=6, display="none", uefi=False, usb=False):
    argv = base_command(accel, memory, cpus, display)
    argv += ["-machine", "q35"]  # AHCI/DMA avoids loading 700 MB through legacy IDE PIO
    if usb:
        argv += ["-drive", f"if=none,id=bootdisk,format=raw,readonly=on,file={str(path).replace(',', ',,')}",
                 "-device", "ich9-usb-ehci1,id=ehci",
                 "-device", "usb-storage,bus=ehci.0,drive=bootdisk,bootindex=1"]
    else:
        argv += ["-cdrom", str(path), "-boot", "d"]
    if uefi:
        # Match the kernel's xAPIC backend throughout firmware initialization.
        argv[argv.index("-cpu") + 1] += ",-x2apic"
        # Keep PCI MMIO in the address range supported by this kernel.
        argv += ["-fw_cfg", "name=opt/ovmf/X-PciMmio64Mb,string=0"]
        argv += ["-drive", "if=pflash,format=raw,readonly=on,file=/usr/share/OVMF/OVMF_CODE_4M.fd",
                 "-drive", "if=pflash,format=raw,snapshot=on,file=/usr/share/OVMF/OVMF_VARS_4M.fd"]
    return argv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("bundle", type=Path, nargs="?")
    parser.add_argument("--iso", type=Path, help="Boot the image's embedded configuration")
    parser.add_argument("--uefi", action="store_true", help="Use OVMF firmware with --iso")
    parser.add_argument("--usb", action="store_true", help="Attach --iso as an emulated USB disk")
    parser.add_argument("--cpus", type=int, default=6)
    parser.add_argument("--threads", type=int)
    parser.add_argument("--display", choices=("none", "gtk", "sdl"), default="none")
    parser.add_argument("--kernel", type=Path)
    parser.add_argument("--mode", choices=("chat", "text", "ids"))
    parser.add_argument("--prompt")
    parser.add_argument("--tokens", type=int)
    parser.add_argument("--context", type=int)
    parser.add_argument("--accel", choices=("kvm", "tcg"), default="kvm")
    parser.add_argument("--memory", default="2G")
    args = parser.parse_args()
    if (args.bundle is None) == (args.iso is None):
        parser.error("Specify either a bundle or --iso")
    if (args.uefi or args.usb) and not args.iso:
        parser.error("--uefi and --usb require --iso")
    if args.iso and any(getattr(args, name) is not None for name in
                        ("kernel", "mode", "prompt", "tokens", "context", "threads")):
        parser.error("Kernel settings are embedded in the ISO; rebuild it with os/image.py --options")
    for name, default in (("kernel", Path("build/lfm-os.elf")), ("mode", "chat"),
                          ("prompt", "What is the capital of France?"), ("tokens", 32),
                          ("context", 2048), ("threads", 6)):
        if getattr(args, name) is None:
            setattr(args, name, default)
    if not 1 <= args.cpus <= 16 or not 1 <= args.threads <= 16:
        parser.error("cpus and threads must be between 1 and 16")
    if not 1 <= args.tokens <= args.context <= 32768:
        parser.error("Require 1 <= tokens <= context <= 32768")
    if any(char in args.prompt for char in "\x00\r\n"):
        parser.error("The boot prompt must be a single line without NUL")
    options = f"mode={args.mode} tokens={args.tokens} context={args.context} threads={args.threads}"
    if args.mode != "chat":
        options += " prompt=" + args.prompt
    if len(options.encode()) + len(str(args.kernel).encode()) + 2 > 8192:
        parser.error("Boot command line exceeds 8192 bytes")
    if args.accel == "kvm" and not os.access("/dev/kvm", os.R_OK | os.W_OK):
        parser.error("KVM unavailable; use --accel tcg for slower software emulation")
    try:
        argv = (media_command(args.iso, args.accel, args.memory, args.cpus, args.display, args.uefi, args.usb)
                if args.iso else command(args.kernel, args.bundle, options, args.accel, args.memory,
                                         args.cpus, args.display))
        result = subprocess.run(argv)
    except KeyboardInterrupt:
        raise SystemExit(130)
    raise SystemExit(0 if result.returncode == 33 else result.returncode or 1)


if __name__ == "__main__":
    main()
