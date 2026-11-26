MODEL ?= model.safetensors
PYTHON ?= python3
NASM ?= nasm
TOKENIZER ?= weights/tokenizer.json
OS_BUNDLE ?= build/lfm-boot.bundle
FONT_PYTHON ?= python3
OS_IMAGE ?= build/lfm-os.iso
.PHONY: all verify verify-threads verify-generation verify-tokenizer verify-chat benchmark benchmark-threads benchmark-generation FORCE
all: build/lfm-next

build:
	mkdir -p $@

# Re-evaluate the header when MODEL is changed, including older checkpoints.
build/layout.inc: FORCE layout.py | build
	$(PYTHON) layout.py "$(MODEL)" $@.tmp
	cmp -s $@.tmp $@ || cp $@.tmp $@
	rm -f $@.tmp

build/main.o: main.asm platform_linux.asm kernels.asm attention.asm threads.asm tokenizer.asm chat.asm layers.inc frequencies.inc build/layout.inc build/tokenizer.inc
	$(NASM) -f elf64 -Ibuild/ main.asm -o $@

build/lfm-next: build/main.o build/tokenizer.bin
	$(LD) -static -z noexecstack -o $@ $<

build/logits.o: main.asm platform_linux.asm kernels.asm attention.asm threads.asm tokenizer.asm chat.asm layers.inc frequencies.inc build/layout.inc build/tokenizer.inc
	$(NASM) -f elf64 -DDUMP_LOGITS -Ibuild/ main.asm -o $@

build/lfm-logits: build/logits.o
	$(LD) -static -z noexecstack -o $@ $<

verify: build/lfm-next build/lfm-logits
	$(PYTHON) verify.py "$(MODEL)" build/lfm-next build/lfm-logits

build/scalar.o: main.asm platform_linux.asm kernels.asm attention.asm threads.asm tokenizer.asm chat.asm layers.inc frequencies.inc build/layout.inc build/tokenizer.inc
	$(NASM) -f elf64 -DFORCE_SCALAR -Ibuild/ main.asm -o $@

build/lfm-scalar: build/scalar.o build/tokenizer.bin
	$(LD) -static -z noexecstack -o $@ $<

benchmark: build/lfm-next build/lfm-scalar
	$(PYTHON) benchmark.py "$(MODEL)" build/lfm-scalar build/lfm-next --output build/benchmark.json

verify-threads: build/lfm-logits
	$(PYTHON) verify_threads.py "$(MODEL)" build/lfm-logits

benchmark-threads: build/lfm-next
	$(PYTHON) benchmark.py "$(MODEL)" build/lfm-next build/lfm-next --threads 2 3 4 6 8 12 --repeats 5 --output build/thread-benchmark.json

verify-generation: build/lfm-next
	$(PYTHON) verify_generation.py "$(MODEL)" build/lfm-next

benchmark-generation: build/lfm-next
	$(PYTHON) benchmark_generation.py "$(MODEL)" build/lfm-next --output build/generation-benchmark.json

verify-tokenizer: build/lfm-next
	$(PYTHON) verify_tokenizer.py "$(TOKENIZER)" build/lfm-next --checkpoint "$(MODEL)"

verify-chat: build/lfm-next
	$(PYTHON) verify_chat.py "$(MODEL)" build/lfm-next

# Grouped targets keep the binary and its offsets from the same tokenizer build.
build/tokenizer.bin build/tokenizer.inc &: build_tokenizer.py $(TOKENIZER) | build
	$(PYTHON) build_tokenizer.py "$(TOKENIZER)" build/tokenizer.bin build/tokenizer.inc

.PHONY: evaluate-quality
evaluate-quality: build/lfm-next build/lfm-logits
	$(PYTHON) evaluate_quality.py "$(MODEL)"

.PHONY: os os-run os-image verify-os
os: build/lfm-os.elf $(OS_BUNDLE)

os-image: $(OS_IMAGE)

$(OS_IMAGE): build/lfm-os.elf $(OS_BUNDLE) os/image.py os/FONT-LICENSE.txt
	$(PYTHON) os/image.py build/lfm-os.elf "$(OS_BUNDLE)" "$(OS_IMAGE)"

os-run: os
	$(PYTHON) os/run.py "$(OS_BUNDLE)"

verify-os: os build/lfm-os-logits.elf build/lfm-next build/lfm-logits build/lfm-scalar
	$(PYTHON) os/verify.py "$(MODEL)" "$(OS_BUNDLE)"

.PHONY: verify-platform
verify-platform: os $(OS_IMAGE) build/lfm-os-logits.elf build/lfm-next build/lfm-logits
	$(PYTHON) os/verify_platform.py "$(MODEL)" "$(OS_BUNDLE)" "$(OS_IMAGE)"

.PHONY: benchmark-os
benchmark-os: os
	$(PYTHON) os/benchmark.py "$(OS_BUNDLE)"

$(OS_BUNDLE) build/boot_bundle.inc &: os/bundle.py $(MODEL) build/tokenizer.bin build/layout.inc | build
	$(PYTHON) os/bundle.py "$(MODEL)" build/tokenizer.bin "$(OS_BUNDLE)" build/boot_bundle.inc

build/os-boot.o: os/boot.asm | build
	$(NASM) -f elf64 os/boot.asm -o $@

build/console-font.bin: os/font.py | build
	$(FONT_PYTHON) os/font.py $@

build/os-main.o: main.asm os/platform.asm os/multiboot.asm os/smp.asm os/console.asm kernels.asm attention.asm tokenizer.asm chat.asm layers.inc frequencies.inc build/layout.inc build/tokenizer.inc build/boot_bundle.inc build/console-font.bin
	$(NASM) -f elf64 -DBARE_METAL -Ibuild/ main.asm -o $@

build/lfm-os.elf64: build/os-boot.o build/os-main.o os/linker.ld
	$(LD) -m elf_x86_64 -T os/linker.ld -z max-page-size=0x1000 -o $@ build/os-boot.o build/os-main.o

build/lfm-os.elf: build/lfm-os.elf64
	objcopy -O elf32-i386 $< $@

build/os-logits.o: main.asm os/platform.asm os/multiboot.asm os/smp.asm os/console.asm kernels.asm attention.asm tokenizer.asm chat.asm layers.inc frequencies.inc build/layout.inc build/tokenizer.inc build/boot_bundle.inc build/console-font.bin
	$(NASM) -f elf64 -DBARE_METAL -DDUMP_LOGITS -Ibuild/ main.asm -o $@

build/lfm-os-logits.elf64: build/os-boot.o build/os-logits.o os/linker.ld
	$(LD) -m elf_x86_64 -T os/linker.ld -z max-page-size=0x1000 -o $@ build/os-boot.o build/os-logits.o

build/lfm-os-logits.elf: build/lfm-os-logits.elf64
	objcopy -O elf32-i386 $< $@

# LFM3-1.7B GGUF runtime. The tokenizer tables sit beside the executable.
LFM3_MODEL ?= LFM3-1.7B-Q4_K_M.gguf
LFM3_TOKENIZER ?= lfm3-ref/tokenizer.json
LFM3_SOURCES = lfm3/main.asm lfm3/kernels.asm lfm3/frequencies.inc platform_linux.asm threads.asm \
	tokenizer.asm chat.asm util.asm expmath.asm expmath_data.inc \
	build/lfm3/layout.inc build/lfm3/lfm3-layers.inc build/lfm3/tokenizer.inc
.PHONY: lfm3
lfm3: build/lfm3/lfm3-next

build/lfm3:
	mkdir -p $@

build/lfm3/layout.inc build/lfm3/lfm3-layers.inc &: FORCE lfm3/layout.py | build/lfm3
	$(PYTHON) lfm3/layout.py "$(LFM3_MODEL)" build/lfm3/layout.inc.tmp build/lfm3/lfm3-layers.inc.tmp
	cmp -s build/lfm3/layout.inc.tmp build/lfm3/layout.inc || cp build/lfm3/layout.inc.tmp build/lfm3/layout.inc
	cmp -s build/lfm3/lfm3-layers.inc.tmp build/lfm3/lfm3-layers.inc || cp build/lfm3/lfm3-layers.inc.tmp build/lfm3/lfm3-layers.inc
	rm -f build/lfm3/layout.inc.tmp build/lfm3/lfm3-layers.inc.tmp

build/lfm3/tokenizer.bin build/lfm3/tokenizer.inc &: build_tokenizer.py $(LFM3_TOKENIZER) | build/lfm3
	$(PYTHON) build_tokenizer.py "$(LFM3_TOKENIZER)" build/lfm3/tokenizer.bin build/lfm3/tokenizer.inc --vocab 128000 --hash-bits 19

build/lfm3/main.o: $(LFM3_SOURCES)
	$(NASM) -f elf64 -Ibuild/lfm3/ lfm3/main.asm -o $@

build/lfm3/lfm3-next: build/lfm3/main.o build/lfm3/tokenizer.bin
	$(LD) -static -z noexecstack -o $@ $<

build/lfm3/logits.o: $(LFM3_SOURCES)
	$(NASM) -f elf64 -DDUMP_LOGITS -Ibuild/lfm3/ lfm3/main.asm -o $@

build/lfm3/lfm3-logits: build/lfm3/logits.o build/lfm3/tokenizer.bin
	$(LD) -static -z noexecstack -o $@ $<

# Diagnostic build: the embedding and every normalized block output, then the logits.
build/lfm3/layers.o: $(LFM3_SOURCES)
	$(NASM) -f elf64 -DDUMP_LAYERS -DDUMP_LOGITS -Ibuild/lfm3/ lfm3/main.asm -o $@

build/lfm3/lfm3-layers: build/lfm3/layers.o build/lfm3/tokenizer.bin
	$(LD) -static -z noexecstack -o $@ $<

LFM3_REF ?= lfm3-ref
.PHONY: verify-lfm3 verify-lfm3-tokenizer
verify-lfm3: build/lfm3/lfm3-next build/lfm3/lfm3-logits build/lfm3/lfm3-layers
	$(PYTHON) lfm3/verify.py "$(LFM3_MODEL)" "$(LFM3_REF)" build/lfm3/lfm3-next build/lfm3/lfm3-logits build/lfm3/lfm3-layers

verify-lfm3-tokenizer: build/lfm3/lfm3-next
	$(PYTHON) lfm3/verify_tokenizer.py "$(LFM3_TOKENIZER)" build/lfm3/lfm3-next
