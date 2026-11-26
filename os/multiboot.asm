; Convert Multiboot 2 tags into the internal Multiboot 1 layout. The original
; tag block stays reserved. Direct QEMU Multiboot 1 remains supported.
extern boot_magic
global normalize_boot
section .text
normalize_boot:
    cmp dword [boot_magic],0x36d76289
    jne .done
    mov ebx,[boot_info]
    mov eax,[rbx]
    cmp eax,16
    jb .bad
    cmp eax,0x100000
    ja .bad
    lea r12,[rbx+rax]
    mov rax,r12
    shr rax,32
    jnz .bad
    mov [boot_metadata_end],r12
    lea r13,[rbx+8]
.tag:
    lea rax,[r13+8]
    cmp rax,r12
    ja .bad
    mov ecx,[r13+4]
    cmp ecx,8
    jb .bad
    lea r14,[r13+rcx]
    cmp r14,r12
    ja .bad
    mov eax,[r13]
    test eax,eax
    jz .finish
    cmp eax,1
    je .cmdline
    cmp eax,3
    je .module
    cmp eax,6
    je .map
    cmp eax,8
    je .framebuffer
    cmp eax,14
    je .acpi_old
    cmp eax,15
    je .acpi_new
    jmp .next
.cmdline:
    lea rsi,[r13+8]
    mov [boot_normalized+16],esi
    or dword [boot_normalized],1<<2
    ; Ensure the string terminates inside its tag before os_options reads it.
.nul:
    cmp rsi,r14
    jae .bad
    cmp byte [rsi],0
    je .next
    inc rsi
    jmp .nul
.module:
    cmp ecx,17
    jb .bad
    cmp dword [boot_normalized+20],0
    jne .bad
    mov rax,[r13+8]
    mov [boot_module],rax
    mov dword [boot_normalized+20],1
    mov dword [boot_normalized+24],boot_module
    or dword [boot_normalized],1<<3
    jmp .next
.map:
    cmp ecx,16
    jb .bad
    cmp dword [r13+12],0
    jne .bad
    mov r8d,[r13+8]
    cmp r8d,24
    jb .bad
    lea rsi,[r13+16]
    mov edi,boot_map
.map_entry:
    lea rax,[rsi+r8]
    cmp rax,r14
    ja .bad
    cmp edi,boot_map+128*24
    jae .bad
    mov dword [rdi],20
    mov rax,[rsi]
    mov [rdi+4],rax
    mov rax,[rsi+8]
    mov [rdi+12],rax
    mov eax,[rsi+16]
    mov [rdi+20],eax
    add rdi,24
    add rsi,r8
    cmp rsi,r14
    jb .map_entry
    sub edi,boot_map
    mov [boot_normalized+44],edi
    mov dword [boot_normalized+48],boot_map
    or dword [boot_normalized],1<<6
    jmp .next
.framebuffer:
    cmp ecx,32
    jb .bad
    lea rsi,[r13+8]
    mov edi,boot_normalized+88
    mov ecx,22
    rep movsb
    cmp byte [boot_normalized+109],1
    jne .fb_done
    cmp dword [r13+4],38
    jb .bad
    lea rsi,[r13+32]
    mov ecx,6
    rep movsb
.fb_done:
    or dword [boot_normalized],1<<12
    jmp .next
.acpi_old:
    cmp ecx,28
    jb .bad
    cmp qword [acpi_rsdp],0
    jne .next
    jmp .rsdp
.acpi_new:
    cmp ecx,44
    jb .bad
.rsdp:
    lea rax,[r13+8]
    mov [acpi_rsdp],rax
.next:
    lea r13,[r14+7]
    and r13,-8
    jmp .tag
.finish:
    mov dword [boot_info],boot_normalized
.done:
    ret
.bad:
    ; COM1 is not initialized yet; initialize before the shared error path.
    call serial_init
    jmp fail
section .bss
boot_metadata_end: resq 1
boot_normalized: resb 116
boot_module: resb 16
boot_map: resb 128*24
section .text
