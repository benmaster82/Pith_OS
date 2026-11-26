"""Package unchanged BF16 weights and tokenizer tables as one boot module."""
import argparse
import os
import shutil
import struct
import tempfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("tokenizer", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("include", type=Path)
    args = parser.parse_args()
    if args.output.resolve() in (args.model.resolve(), args.tokenizer.resolve()):
        raise ValueError("Bundle output must differ from inputs")
    model_offset = 4096
    model_size = args.model.stat().st_size
    tokenizer_offset = (model_offset + model_size + 4095) & -4096
    tokenizer_size = args.tokenizer.stat().st_size
    total = tokenizer_offset + tokenizer_size
    header = struct.pack("<8sQQQQQ16x", b"LFMBT001", model_offset, model_size,
                         tokenizer_offset, tokenizer_size, total)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=args.output.parent, delete=False) as out:
            temporary = Path(out.name)
            out.write(header)
            for path, offset in ((args.model, model_offset), (args.tokenizer, tokenizer_offset)):
                out.seek(offset)
                with path.open("rb") as source:
                    shutil.copyfileobj(source, out, 1024 * 1024)
            out.flush()
            os.fsync(out.fileno())
        os.replace(temporary, args.output)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    args.include.write_text(f"%define BUNDLE_SIZE {total}\n%define MODEL_OFFSET {model_offset}\n"
        f"%define TOKENIZER_OFFSET {tokenizer_offset}\n%macro BUNDLE_HEADER 0\n"
        "bundle_header: db " + ",".join(map(str, header)) + "\n%endmacro\n")
    print(f"Boot module: {total:,} bytes; weights and tokenizer copied without conversion")


if __name__ == "__main__":
    main()
