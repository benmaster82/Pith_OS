OUTPUT_FORMAT(elf64-x86-64)
ENTRY(boot_start)
PHDRS {
  code PT_LOAD FLAGS(5);
  data PT_LOAD FLAGS(6);
}
SECTIONS {
  . = 0x100000;
  .multiboot : { KEEP(*(.multiboot)) } :code
  .text ALIGN(16) : { *(.text*) } :code
  .rodata ALIGN(16) : { *(.rodata*) } :code
  .data ALIGN(4096) : { *(.data*) } :data
  .bss ALIGN(4096) (NOLOAD) : {
    __bss_start = .;
    *(.bss*) *(COMMON)
    __bss_end = .;
  } :data
  __kernel_end = ALIGN(4096);
  /DISCARD/ : { *(.note*) *(.comment*) *(.eh_frame*) }
}
