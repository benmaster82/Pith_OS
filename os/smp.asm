; xAPIC startup and persistent workers. All kernel/model addresses are identity
; mapped below 4 GiB. Unsupported firmware/APIC configurations keep one core.
%define MAX_THREADS 16
extern ap_trampoline_start, ap_trampoline_end, page_directories, idt_pointer
global ap_entry
section .text
smp_init:
    cmp dword [requested_threads],1
    jbe .report
    call acpi_find_cpus
    cmp dword [cpu_count],0
    je .report
    mov ecx,0x1b
    rdmsr
    test eax,1<<10             ; x2APIC mode needs a different MSR backend
    jnz .report
    test edx,edx
    jnz .report
    or eax,1<<11
    wrmsr
    and eax,0xfffff000
    mov [lapic_base],rax
    mov edi,eax
    shr eax,21
    or qword [page_directories+rax*8],0x18
    mov rax,cr3
    mov cr3,rax
    mov dword [rdi+0xf0],0x1ff
    mov eax,[rdi+0x20]
    shr eax,24
    mov [bsp_id],eax
    ; Check the fixed SIPI page is usable conventional RAM.
    mov ebx,[boot_info]
    mov esi,[rbx+48]
.low_memory:
    cmp rsi,[mmap_end]
    jae .report
    cmp dword [rsi+20],1
    jne .next_region
    cmp qword [rsi+4],0x8000
    ja .next_region
    mov rax,[rsi+4]
    add rax,[rsi+12]
    jc .next_region
    cmp rax,0x9000
    jae .prepare
.next_region:
    mov eax,[rsi]
    lea rsi,[rsi+rax+4]
    jmp .low_memory
.prepare:
    call calibrate_tsc
    test eax,eax
    jz .report
    mov edi,ap_slots
    mov eax,-1
    mov ecx,256/4
    rep stosd
    mov edi,worker_partition
    mov ecx,MAX_THREADS
    rep stosd
    mov edi,0x8000
    mov esi,ap_trampoline_start
    mov ecx,ap_trampoline_end
    sub ecx,ap_trampoline_start
    rep movsb
    mfence
    xor r12d,r12d
    mov r13d,1
.start:
    cmp r12d,[cpu_count]
    jae .collect
    cmp r13d,[requested_threads]
    jae .collect
    movzx ebx,byte [cpu_ids+r12]
    inc r12d
    cmp ebx,[bsp_id]
    je .start
    cmp byte [ap_slots+rbx],0xff
    jne .start
    mov [ap_slots+rbx],r13b
    mov eax,0xc500
    call send_ipi
    jc .next_cpu
    mov ecx,10
    call delay_ms
    mov eax,0x8500
    call send_ipi
    jc .next_cpu
    mov eax,0x608              ; SIPI vector 8 -> 0x8000
    call send_ipi
    jc .next_cpu
    mov ecx,1
    call delay_ms
    mov eax,0x608
    call send_ipi
    mov ecx,100
    call deadline_ms
    mov r10,rax
.online:
    cmp byte [ap_online+r13],1
    je .next_cpu
    pause
    rdtsc
    shl rdx,32
    or rax,rdx
    cmp rax,r10
    jb .online
.next_cpu:
    inc r13d                  ; never reuse a timed-out processor's stack
    jmp .start
.collect:
    mov ecx,1
    mov eax,1
.slot:
    cmp byte [ap_online+rcx],1
    jne .skip
    mov [worker_partition+rcx*4],eax
    inc eax
.skip:
    inc ecx
    cmp ecx,MAX_THREADS
    jb .slot
    mov [thread_count],eax
    mov dword [workers_released],1
.report:
    mov esi,smp_notice
    call os_puts
    mov eax,[thread_count]
    call print_id
    ret

; ACPI checksum, rsi table and ecx byte count; ZF=1 means valid.
acpi_checksum:
    xor eax,eax
.sum:
    add al,[rsi+rcx-1]
    loop .sum
    test al,al
    ret

; Validate standard SDT length/checksum, return ecx=length or zero.
acpi_sdt:
    cmp rsi,0x1000
    jb .bad
    mov eax,0xffffffe0
    cmp rsi,rax
    jae .bad
    mov ecx,[rsi+4]
    cmp ecx,36
    jb .bad
    cmp ecx,0x100000
    ja .bad
    lea rax,[rsi+rcx]
    shr rax,32
    jnz .bad
    push rcx
    call acpi_checksum
    pop rcx
    jnz .bad
    ret
.bad:
    xor ecx,ecx
    ret

acpi_find_cpus:
    mov rsi,[acpi_rsdp]
    test rsi,rsi
    jnz .rsdp
    movzx esi,word [0x40e]
    shl esi,4
    test esi,esi
    jz .bios
    lea edi,[rsi+1024]
    call .scan
    jnz .rsdp
.bios:
    mov esi,0xe0000
    mov edi,0x100000
    call .scan
    jz .done
.rsdp:
    mov r8,rsi
    mov ecx,20
    call acpi_checksum
    jnz .done
    mov ebp,4
    mov esi,[r8+16]
    cmp byte [r8+15],2
    jb .root
    mov ecx,[r8+20]
    cmp ecx,36
    jb .done
    cmp ecx,4096
    ja .done
    mov rsi,r8
    call acpi_checksum
    jnz .done
    mov rsi,[r8+24]
    mov rax,rsi
    shr rax,32
    jnz .rsdt
    test rsi,rsi
    jz .rsdt
    mov ebp,8
    jmp .root
.rsdt:
    mov esi,[r8+16]
.root:
    call acpi_sdt
    test ecx,ecx
    jz .done
    cmp ebp,8
    je .xsdt
    cmp dword [rsi],0x54445352
    jne .done
    jmp .entries
.xsdt:
    cmp dword [rsi],0x54445358
    jne .done
.entries:
    lea r11,[rsi+rcx]
    lea r10,[rsi+36]
.table:
    lea rax,[r10+rbp]
    cmp rax,r11
    ja .done
    mov esi,[r10]
    cmp ebp,8
    jne .read
    cmp dword [r10+4],0
    jne .next
.read:
    call acpi_sdt
    cmp ecx,44
    jb .next
    cmp dword [rsi],0x43495041  ; APIC
    je .madt
.next:
    add r10,rbp
    jmp .table
.madt:
    lea r11,[rsi+rcx]
    add rsi,44
.entry:
    lea rax,[rsi+2]
    cmp rax,r11
    ja .done
    movzx ecx,byte [rsi+1]
    cmp ecx,2
    jb .done
    lea r10,[rsi+rcx]
    cmp r10,r11
    ja .done
    cmp byte [rsi],0
    jne .x2
    cmp ecx,8
    jb .done
    test dword [rsi+4],1
    jz .next_entry
    mov al,[rsi+3]
    jmp .add
.x2:
    cmp byte [rsi],9
    jne .next_entry
    cmp ecx,16
    jb .done
    test dword [rsi+8],1
    jz .next_entry
    mov eax,[rsi+4]
    cmp eax,255
    ja .next_entry
.add:
    mov ecx,[cpu_count]
    cmp ecx,256
    jae .done
    mov [cpu_ids+rcx],al
    inc dword [cpu_count]
.next_entry:
    mov rsi,r10
    cmp rsi,r11
    jb .entry
.done:
    ret
.scan:
    mov rax,0x2052545020445352
.search:
    cmp [rsi],rax
    je .found
    add rsi,16
    cmp rsi,rdi
    jb .search
    xor eax,eax
    ret
.found:
    or eax,1
    ret

; Calibrate the invariant TSC against a 10 ms PIT channel-2 one-shot.
; A bounded poll preserves single-core operation if the legacy timer is absent.
calibrate_tsc:
    in al,0x61
    mov r9b,al
    and al,0xfc
    out 0x61,al
    mov al,0xb0
    out 0x43,al
    mov ax,11932
    out 0x42,al
    mov al,ah
    out 0x42,al
    rdtsc
    shl rdx,32
    or rax,rdx
    mov r8,rax
    mov al,r9b
    and al,0xfc
    or al,1
    out 0x61,al
    mov ecx,2000000
.wait:
    in al,0x61
    test al,0x20
    jnz .elapsed
    loop .wait
    mov al,r9b
    out 0x61,al
    xor eax,eax
    ret
.elapsed:
    rdtsc
    shl rdx,32
    or rax,rdx
    sub rax,r8
    xor edx,edx
    mov ecx,10
    div rcx
    mov [tsc_per_ms],rax
    mov al,r9b
    out 0x61,al
    mov eax,1
    ret

deadline_ms:
    mov r8,[tsc_per_ms]
    imul r8,rcx
    rdtsc
    shl rdx,32
    or rax,rdx
    add rax,r8
    ret
delay_ms:
    call deadline_ms
    mov r8,rax
.wait:
    pause
    rdtsc
    shl rdx,32
    or rax,rdx
    cmp rax,r8
    jb .wait
    ret

; ebx=physical destination, eax=ICR command. CF=1 on delivery timeout.
send_ipi:
    mov r9d,eax
    mov ecx,100
    call deadline_ms
    mov r8,rax
    mov rdi,[lapic_base]
.wait:
    test dword [rdi+0x300],1<<12
    jz .send
    pause
    rdtsc
    shl rdx,32
    or rax,rdx
    cmp rax,r8
    jb .wait
    stc
    ret
.send:
    mov eax,ebx
    shl eax,24
    mov [rdi+0x310],eax
    mov [rdi+0x300],r9d
    clc
    ret

ap_entry:
    mov ax,0x10
    mov ds,ax
    mov es,ax
    mov ss,ax
    xor eax,eax
    mov fs,ax
    mov gs,ax
    mov eax,1
    cpuid
    shr ebx,24
    movzx r13d,byte [ap_slots+rbx]
    cmp r13d,MAX_THREADS
    jae .park
    mov eax,r13d
    shl eax,14
    lea rsp,[worker_stacks+rax]
    lidt [idt_pointer]
    cmp qword [matvec_impl],matvec_avx2
    jne .compatible
    ; A heterogeneous processor that cannot run the selected kernel stays parked.
    mov eax,1
    cpuid
    and ecx,0x18001000
    cmp ecx,0x18001000
    jne .park
    xor ecx,ecx
    xgetbv
    and eax,6
    cmp eax,6
    jne .park
    mov eax,7
    xor ecx,ecx
    cpuid
    test ebx,1<<5
    jz .park
.compatible:
    mov byte [ap_online+r13],1
.release:
    cmp dword [workers_released],1
    je .released
    pause
    jmp .release
.released:
    mov r13d,[worker_partition+r13*4]
    cmp r13d,-1
    je .park
    xor r14d,r14d
.wait:
    cmp dword [workers_stopping],0
    jne .park
    mov eax,[job_epoch]
    cmp eax,r14d
    jne .work
    pause
    jmp .wait
.work:
    mov r14d,eax
    mov eax,r13d
    call run_job
    lock dec dword [job_remaining]
    jmp .wait
.park:
    cli
    hlt
    jmp .park

parallel_matvec:
    mov [job_input],rdi
    mov [job_weights],rsi
    mov [job_output],rdx
    mov [job_width],ecx
    mov [job_rows],r8d
    mov eax,[thread_count]
    dec eax
    mov [job_remaining],eax
    lock inc dword [job_epoch]
    xor eax,eax
    call run_job
.wait:
    cmp dword [job_remaining],0
    je .done
    pause
    jmp .wait
.done:
    ret

run_job:
    mov r9d,eax
    mov eax,[job_rows]
    xor edx,edx
    div dword [thread_count]
    mov r8d,eax
    imul eax,r9d
    inc r9d
    cmp r9d,[thread_count]
    jne .address
    add r8d,edx
.address:
    mov ecx,[job_width]
    mov rdx,[job_output]
    lea rdx,[rdx+rax*4]
    imul rax,rcx
    mov rsi,[job_weights]
    lea rsi,[rsi+rax*2]
    mov rdi,[job_input]
    jmp [matvec_impl]

section .rodata
smp_notice: db "Active CPU cores: ",0
section .data
thread_count: dd 1
requested_threads: dd 6
section .bss
acpi_rsdp: resq 1
cpu_count: resd 1
cpu_ids: resb 256
bsp_id: resd 1
lapic_base: resq 1
tsc_per_ms: resq 1
ap_slots: resb 256
ap_online: resb MAX_THREADS
worker_partition: resd MAX_THREADS
workers_released: resd 1
workers_stopping: resd 1
alignb 64
job_epoch: resd 1
alignb 64
job_remaining: resd 1
alignb 64
job_input: resq 1
job_weights: resq 1
job_output: resq 1
job_width: resd 1
job_rows: resd 1
alignb 16
worker_stacks: resb (MAX_THREADS-1)*16384
section .text
