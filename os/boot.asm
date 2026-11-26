; Original Multiboot 1 entry, one core, 4 GiB identity map using 2 MiB pages.
bits 32
default abs
section .multiboot align=4
    dd 0x1badb002, 3, -(0x1badb002+3)
align 8
mb2_header:
    dd 0xe85250d6,0,mb2_end-mb2_header
    dd -(0xe85250d6+mb2_end-mb2_header)
    dw 5,1                    ; optional framebuffer request
    dd 20,1024,768,32
align 8
    dw 0,0
    dd 8
mb2_end:

section .text
global boot_start
global boot_info, boot_magic, pml4, page_directories, idt_pointer
global ap_trampoline_start, ap_trampoline_end
extern __bss_start, __bss_end, os_entry, write_all, exit, ap_entry, normalize_boot
boot_start:
    cli
    cld
    mov esi,eax
    mov ebp,ebx
    lgdt [gdt_pointer]
    jmp 0x08:.segments
.segments:
    mov ax,0x10
    mov ds,ax
    mov es,ax
    mov ss,ax
    mov edi,__bss_start
    mov ecx,__bss_end
    sub ecx,edi
    xor eax,eax
    rep stosb
    mov esp,boot_stack_end
    mov [boot_info],ebp
    mov [boot_magic],esi
    cmp esi,0x2badb002
    je .valid_magic
    cmp esi,0x36d76289
    jne boot_error
.valid_magic:
    mov eax,0x80000000
    cpuid
    cmp eax,0x80000001
    jb boot_error
    mov eax,0x80000001
    cpuid
    test edx,1<<29
    jz boot_error
    mov eax,cr0
    wbinvd
    and eax,~((1<<2)|(1<<3)|(1<<29)|(1<<30)) ; clear EM/TS and enable caching
    or eax,1<<1                ; MP
    mov cr0,eax
    mov eax,cr4
    or eax,(1<<5)|(1<<9)|(1<<10) ; PAE, OSFXSR, OSXMMEXCPT
    mov cr4,eax
    fninit
    ldmxcsr [default_mxcsr]
    mov eax,1
    cpuid
    and ecx,(1<<26)|(1<<28)
    cmp ecx,(1<<26)|(1<<28)
    jne .paging
    mov eax,cr4
    or eax,1<<18
    mov cr4,eax
    xor ecx,ecx
    xor edx,edx
    mov eax,7                  ; x87, SSE, AVX state
    xsetbv
.paging:
    mov eax,pdpt
    or eax,3
    mov [pml4],eax
    xor ecx,ecx
.directories:
    mov eax,ecx
    shl eax,12
    add eax,page_directories
    or eax,3
    mov [pdpt+ecx*8],eax
    inc ecx
    cmp ecx,4
    jb .directories
    xor ecx,ecx
.pages:
    mov eax,ecx
    shl eax,21
    or eax,0x83
    cmp ecx,2032               ; top 32 MiB includes APIC/device MMIO
    jb .store
    or eax,0x18
.store:
    mov [page_directories+ecx*8],eax
    inc ecx
    cmp ecx,2048
    jb .pages
    mov eax,pml4
    mov cr3,eax
    mov ecx,0xc0000080
    rdmsr
    or eax,1<<8
    wrmsr
    mov eax,cr0
    or eax,1<<31
    mov cr0,eax
    jmp 0x18:long_entry

boot_error:
    mov dx,0xe9
    mov al,'!'
    out dx,al
    mov dx,0xf4
    mov eax,0x11
    out dx,eax
.halt:
    hlt
    jmp .halt

bits 64
long_entry:
    mov ax,0x10
    mov ds,ax
    mov es,ax
    mov ss,ax
    xor eax,eax
    mov fs,ax
    mov gs,ax
    mov rsp,boot_stack_end
    xor ebp,ebp
    ; Catch exceptions before touching model or tokenizer memory.
    mov edi,idt
    mov ecx,256
    mov rax,exception_entry
.gate:
    mov word [rdi],ax
    mov word [rdi+2],0x18
    mov byte [rdi+4],0
    mov byte [rdi+5],0x8e
    mov rdx,rax
    shr rdx,16
    mov word [rdi+6],dx
    shr rdx,16
    mov dword [rdi+8],edx
    mov dword [rdi+12],0
    add rdi,16
    loop .gate
    lidt [idt_pointer]
    call normalize_boot
    call os_entry
    xor edi,edi
    jmp exit

exception_entry:
    cli
    mov rsp,boot_stack_end
    cld
    mov edi,2
    mov esi,exception_message
    mov edx,exception_length
    call write_all
    mov edi,1
    jmp exit

; Copied to conventional RAM at 0x8000 before INIT/SIPI. No stack is used
; until ap_entry assigns a private stack from the processor's APIC ID.
align 16
ap_trampoline_start:
bits 16
    cli
    cld
    xor ax,ax
    mov ds,ax
    lgdt [0x8000+ap_gdt_pointer-ap_trampoline_start]
    mov eax,cr0
    or eax,1
    mov cr0,eax
    jmp dword 0x08:0x8000+ap_protected-ap_trampoline_start
bits 32
ap_protected:
    mov ax,0x10
    mov ds,ax
    mov es,ax
    mov ss,ax
    mov eax,cr0
    wbinvd
    and eax,~((1<<2)|(1<<3)|(1<<29)|(1<<30))
    or eax,1<<1
    mov cr0,eax
    mov eax,(1<<5)|(1<<9)|(1<<10)
    mov cr4,eax
    fninit
    ldmxcsr [default_mxcsr]
    mov eax,1
    cpuid
    and ecx,(1<<26)|(1<<28)
    cmp ecx,(1<<26)|(1<<28)
    jne .paging
    mov eax,cr4
    or eax,1<<18
    mov cr4,eax
    xor ecx,ecx
    xor edx,edx
    mov eax,7
    xsetbv
.paging:
    mov eax,pml4
    mov cr3,eax
    mov ecx,0xc0000080
    rdmsr
    or eax,1<<8
    wrmsr
    mov eax,cr0
    or eax,1<<31
    mov cr0,eax
    jmp 0x18:0x8000+ap_long-ap_trampoline_start
bits 64
ap_long:
    mov rax,ap_entry
    jmp rax
ap_gdt_pointer:
    dw gdt_end-gdt-1
    dd gdt
ap_trampoline_end:

section .rodata align=8
gdt:
    dq 0,0x00cf9a000000ffff,0x00cf92000000ffff,0x00af9a000000ffff
gdt_end:
gdt_pointer:
    dw gdt_end-gdt-1
    dd gdt
idt_pointer:
    dw 4095
    dq idt
default_mxcsr: dd 0x1f80
exception_message: db 10,"FATAL: CPU exception",10
exception_length equ $-exception_message

section .bss align=4096
boot_info: resd 1
boot_magic: resd 1
alignb 4096
pml4: resb 4096
pdpt: resb 4096
page_directories: resb 4096*4
idt: resb 4096
alignb 16
boot_stack: resb 65536
boot_stack_end:
section .note.GNU-stack noalloc noexec nowrite progbits
