"""Exercise SMP, graphical consoles, PS/2, BIOS/UEFI and hybrid USB boot in QEMU."""
import argparse
import json
import os
import re
from pathlib import Path
import selectors
import socket
import subprocess
import sys
import tempfile
import time

from run import command, media_command

# The Pithos CORE prompt: an amber mark, then bright user text. COM1 carries
# the same colour sequences as the screen.
USER_LABEL = b"\n\x1b[33m\xe2\x80\xba\x1b[1m "
PLAIN_LABEL = b"\n\xe2\x80\xba "


def plain(data):
    return re.sub(rb"\x1b\[[0-9;]*m", b"", data)


class Guest:
    def __init__(self, argv):
        self.directory = tempfile.TemporaryDirectory(prefix="lfm-qmp-")
        self.path = Path(self.directory.name) / "qmp"
        self.stderr = tempfile.TemporaryFile()
        self.process = subprocess.Popen(argv + ["-qmp", f"unix:{self.path},server=on,wait=off"],
                                        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.stderr)
        self.selector = selectors.DefaultSelector()
        self.selector.register(self.process.stdout, selectors.EVENT_READ)
        self.pending = bytearray()
        self.qmp = None
        self.socket = None

    def __enter__(self):
        try:
            deadline = time.monotonic() + 10
            while not self.path.exists():
                if time.monotonic() >= deadline or self.process.poll() is not None:
                    self.stderr.seek(0)
                    raise AssertionError(self.stderr.read())
                time.sleep(.02)
            self.socket = socket.socket(socket.AF_UNIX)
            self.socket.settimeout(10)
            self.socket.connect(str(self.path))
            self.qmp = self.socket.makefile("rwb", buffering=0)
            assert "QMP" in json.loads(self.qmp.readline())
            self.execute("qmp_capabilities")
            return self
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def __exit__(self, *unused):
        if self.process.poll() is None:
            self.process.kill()
        self.process.wait()
        self.selector.close()
        for stream in (self.process.stdin, self.process.stdout, self.stderr, self.qmp, self.socket):
            if stream is not None:
                stream.close()
        self.directory.cleanup()

    def execute(self, name, arguments=None):
        self.qmp.write(json.dumps({"execute": name, "arguments": arguments or {}}).encode() + b"\n")
        while True:
            line = self.qmp.readline()
            assert line, (name, self.process.poll())
            response = json.loads(line)
            assert "error" not in response, response
            if "return" in response:
                return response["return"]

    def until(self, marker=USER_LABEL, timeout=180):
        deadline = time.monotonic() + timeout
        while marker not in self.pending:
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not self.selector.select(max(0, remaining)):
                self.screenshot("build/os-failed.png")
                registers = self.execute("human-monitor-command", {"command-line": "info registers"})
                Path("build/os-failed-registers.txt").write_text(registers)
                raise AssertionError(bytes(self.pending[-1000:]))
            chunk = os.read(self.process.stdout.fileno(), 65536)
            assert chunk, (self.process.poll(), bytes(self.pending[-1000:]))
            self.pending.extend(chunk)
        end = self.pending.index(marker) + len(marker)
        output = bytes(self.pending[:end])
        del self.pending[:end]
        return output

    def serial(self, data):
        self.process.stdin.write(data)
        self.process.stdin.flush()

    def key(self, text):
        punctuation = {" ": "spc", "\n": "ret", "\b": "backspace", "/": "slash", "?": "shift-slash"}
        for char in text:
            key = punctuation.get(char, "shift-" + char.lower() if char.isupper() else char)
            self.execute("human-monitor-command", {"command-line": f"sendkey {key} 15"})
            time.sleep(.04)

    def screenshot(self, path):
        self.execute("screendump", {"filename": str(Path(path).resolve()), "format": "png"})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("image", type=Path)
    parser.add_argument("--only", choices=("smp", "media", "console", "uefi-usb"))
    parser.add_argument("--repeats", type=int, default=1, help="Repeat media boot checks")
    args = parser.parse_args()
    if not 1 <= args.repeats <= 10:
        parser.error("repeats must be in 1..10")
    france = "What is the capital of France?"
    options = f"mode=ids tokens=16 threads=16 prompt={france}"
    expected = b"1098\n5706\n803\n4481\n856\n5242\n523\n7\n"

    def payload(output):
        return output.split(b"<lfm-output>\n", 1)[1].rsplit(b"\n</lfm-output>\n", 1)[0]

    if args.only in (None, "smp"):
        for cores in (1, 2, 3, 4, 6, 16):
            result = subprocess.run(command("build/lfm-os.elf", args.bundle, options, cpus=cores),
                                    capture_output=True, timeout=180)
            assert result.returncode == 33, (cores, result.stdout, result.stderr)
            assert f"Active CPU cores: {cores}\n".encode() in result.stdout, result.stdout
            assert payload(result.stdout) == expected, result.stdout
        print("Greedy generation matches on 1, 2, 3, 4, 6 and 16 active cores", flush=True)
        template = "<|startoftext|><|im_start|>user\n" + france + "<|im_end|>\n<|im_start|>assistant\n"
        ids = subprocess.check_output(["build/lfm-next", "_", "--encode", template]).split()
        reference = subprocess.check_output(["build/lfm-logits", str(args.checkpoint), *[x.decode() for x in ids]],
                                            env=dict(os.environ, LFM_THREADS="1"))
        for cores in (3, 6):
            result = subprocess.run(command("build/lfm-os-logits.elf", args.bundle, options, cpus=cores),
                                    capture_output=True, timeout=180)
            assert result.returncode == 33 and payload(result.stdout) == reference, cores
        print("All logits are bitwise identical to Linux with 3 and 6 cores", flush=True)
        result = subprocess.run(command("build/lfm-os.elf", args.bundle, options, cpus=4) + ["-machine", "pc,acpi=off"],
                                capture_output=True, timeout=60)
        assert result.returncode == 33 and b"Active CPU cores: 1\n" in result.stdout, (result.stdout, result.stderr)
        assert payload(result.stdout) == expected
        print("Missing ACPI falls back to one core", flush=True)

    if args.only in (None, "console"):
        argv = command("build/lfm-os.elf", args.bundle, "mode=chat tokens=16 threads=4", cpus=4)
        with Guest(argv) as guest:
            assert b"Active CPU cores: 4\n" in guest.until()
            guest.key("What is the capital of Francx\be?\n")
            output = guest.until()
            assert b"The capital of France is Paris.\n" in plain(output), output
            guest.screenshot("build/os-vga.png")
            guest.key("/clear\n")
            assert b"Conversation cleared" in guest.until()
            guest.key("/quit\n")
            assert guest.process.wait(timeout=10) == 33
        print("VGA console and PS/2 typing, shift, backspace, /clear and /quit passed", flush=True)
        argv = command("build/lfm-os.elf", args.bundle, "mode=text tokens=16", cpus=2)
        argv[argv.index("-serial") + 1] = "none"
        result = subprocess.run(argv, capture_output=True, timeout=60)
        assert result.returncode == 33, result.stderr
        print("Absent COM1 does not block graphical inference", flush=True)
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run([sys.executable, "os/run.py", "--iso", str(Path(directory) / "missing.iso")],
                                    capture_output=True, timeout=20)
            assert result.returncode != 0, "QEMU startup failure reported as success"
        result = subprocess.run([sys.executable, "os/run.py", str(args.bundle), "--mode", "ids", "--tokens", "16",
                                 "--cpus", "2", "--threads", "2"], capture_output=True, timeout=60)
        assert result.returncode == 0 and payload(result.stdout) == expected, (result.stdout, result.stderr)
        print("Public launcher distinguishes successful inference from QEMU startup failure", flush=True)
        argv = command("build/lfm-os.elf", args.bundle, "mode=chat tokens=16 threads=4", cpus=4)
        index = argv.index("isa-debug-exit,iobase=0xf4,iosize=4")
        del argv[index-1:index+1]
        with Guest(argv) as guest:
            guest.until()
            guest.serial(b"/quit\n")
            time.sleep(.2)
            for cpu in range(4):
                guest.execute("human-monitor-command", {"command-line": f"cpu {cpu}"})
                registers = guest.execute("human-monitor-command", {"command-line": "info registers"})
                assert "HLT=1" in registers, (cpu, registers)
        print("Without the QEMU exit device, /quit halts the BSP and all three workers", flush=True)

    if args.only in (None, "media", "uefi-usb"):
        cases = [(True, True)] if args.only == "uefi-usb" else [(False, False), (True, False), (False, True), (True, True)]
        for uefi, usb in cases * args.repeats:
            name = ("uefi" if uefi else "bios") + ("-usb" if usb else "-cd")
            with Guest(media_command(args.image, cpus=4, uefi=uefi, usb=usb)) as guest:
                output = guest.until()
                assert b"Active CPU cores: 4\n" in output, (name, output[-1000:])
                guest.key(france + "\n")
                output = guest.until()
                assert b"The capital of France is Paris.\n" in plain(output), (name, output)
                guest.screenshot(f"build/os-{name}.png")
                guest.serial(b"/clear\n")
                assert b"Conversation cleared" in guest.until()
                guest.serial(b"/quit\n")
                assert guest.process.wait(timeout=10) == 33
            print(f"{name}: GRUB boot, four cores, framebuffer, PS/2 chat and shutdown passed", flush=True)


if __name__ == "__main__":
    main()
