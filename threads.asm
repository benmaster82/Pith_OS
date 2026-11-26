; Linux clone/futex worker pool. A job partitions output rows, so each dot
; product retains exactly the same reduction order at every thread count.
%define MAX_THREADS 16
%define WORKER_STACK 16384

section .text
; rdi = envp. Invalid LFM_THREADS values fail before creating any workers.
init_threads:
    push r12
    push r13
    push r14
    mov r12,rdi
.environment:
    mov rsi,[r12]
    test rsi,rsi
    jz .create
    xor ecx,ecx
.name:
    mov al,[thread_env+rcx]
    cmp al,[rsi+rcx]
    jne .next
    inc ecx
    cmp ecx,12
    jb .name
    add rsi,12
    xor eax,eax
    cmp byte [rsi],0
    je fail
.digit:
    movzx edx,byte [rsi]
    sub edx,'0'
    cmp edx,9
    ja fail
    imul eax,10
    add eax,edx
    cmp eax,MAX_THREADS
    ja fail
    inc rsi
    cmp byte [rsi],0
    jne .digit
    test eax,eax
    jz fail
    mov [requested_threads],eax
    jmp .create
.next:
    add r12,8
    jmp .environment
.create:
    cmp dword [requested_threads],0
    jne .start
    ; The default is limited to six available CPUs. Affinity may expose fewer
    ; CPUs than the host total, for example under taskset or a container.
    mov dword [requested_threads],1
    mov eax,204                 ; sched_getaffinity
    xor edi,edi
    mov esi,128
    mov edx,cpu_mask
    syscall
    test rax,rax
    js .start
    xor r8d,r8d
    xor ecx,ecx
.byte:
    movzx edx,byte [cpu_mask+r8]
.bit:
    test edx,edx
    jz .next_byte
    lea r9d,[rdx-1]
    and edx,r9d
    inc ecx
    jmp .bit
.next_byte:
    inc r8d
    cmp r8d,eax
    jb .byte
    test ecx,ecx
    jz .start
    mov eax,6
    cmp ecx,eax
    cmova ecx,eax
    mov [requested_threads],ecx
.start:
    mov r13d,1
.spawn:
    cmp r13d,[requested_threads]
    jae .done
    mov esi,r13d
    shl esi,14
    add rsi,worker_stacks
    ; VM, FS, FILES, SIGHAND, THREAD and SYSVSEM are shared.
    mov edi,0x50f00
    xor edx,edx
    xor r10d,r10d
    xor r8d,r8d
    mov eax,56
    syscall
    test rax,rax
    jz worker_main
    js .done                    ; use the workers successfully created
    inc dword [thread_count]
    inc r13d
    jmp .spawn
.done:
    pop r14
    pop r13
    pop r12
    ret

; r13d is the worker index, inherited across clone. Its new stack is private.
worker_main:
    xor r14d,r14d
.wait:
    mov ecx,4000
.spin:
    mov eax,[job_epoch]
    cmp eax,r14d
    jne .work
    pause
    dec ecx
    jnz .spin
    mov edi,job_epoch
    mov esi,128                 ; FUTEX_WAIT_PRIVATE
    mov edx,r14d
    xor r10d,r10d
    mov eax,202
    syscall
    jmp .wait                   ; handles EINTR, EAGAIN and spurious wakeups
.work:
    mov r14d,eax
    mov eax,r13d
    call [job_entry]
    lock dec dword [job_remaining]
    jnz .wait
    cmp dword [parent_waiting],0
    je .wait
    mov edi,job_remaining
    mov esi,129                 ; FUTEX_WAKE_PRIVATE
    mov edx,1
    mov eax,202
    syscall
    jmp .wait

parallel_matvec:
    mov [job_input],rdi
    mov [job_weights],rsi
    mov [job_output],rdx
    mov [job_width],ecx
    mov [job_rows],r8d
    mov qword [job_entry],run_job
; Run [job_entry] with eax = partition index on every thread.
dispatch_job:
    mov eax,[thread_count]
    dec eax
    mov [job_remaining],eax
    ; Publish after parameters and completion count. x86 stores are ordered.
    lock inc dword [job_epoch]
    mov edi,job_epoch
    mov esi,129
    mov edx,MAX_THREADS-1
    mov eax,202
    syscall
    xor eax,eax                 ; main thread computes partition zero
    call [job_entry]
    mov ecx,4000
.spin:
    cmp dword [job_remaining],0
    je .done
    pause
    dec ecx
    jnz .spin
    ; Publish the wait flag before rechecking completion. The locked exchange
    ; prevents a missed wake if the last worker finishes at this boundary.
    mov eax,1
    xchg eax,[parent_waiting]
.wait:
    mov edx,[job_remaining]
    test edx,edx
    jz .done
    mov edi,job_remaining
    mov esi,128
    xor r10d,r10d
    mov eax,202
    syscall
    jmp .wait
.done:
    mov dword [parent_waiting],0
    ret

; eax = partition index. Last partition owns the remainder.
run_job:
    mov r9d,eax
    mov eax,[job_rows]
    xor edx,edx
    div dword [thread_count]
    mov r8d,eax
    imul eax,r9d               ; first output row
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
thread_env: db "LFM_THREADS="

section .data
thread_count: dd 1
requested_threads: dd 0

section .bss
cpu_mask: resb 128
alignb 64
job_epoch: resd 1
alignb 64
job_remaining: resd 1
parent_waiting: resd 1
alignb 64
job_input: resq 1
job_weights: resq 1
job_output: resq 1
job_width: resd 1
job_rows: resd 1
job_entry: resq 1
job_kv: resq 1
job_position: resd 1
alignb 64
worker_stacks: resb (MAX_THREADS-1)*WORKER_STACK

section .text
