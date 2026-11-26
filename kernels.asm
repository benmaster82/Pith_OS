; BF16 weight / FP32 activation kernels. x86-64, Linux or bare metal.
; rdi = source, rsi = BF16 weights, rdx = destination, ecx = width.
; matvec also takes r8d = rows. Preserve r12-r15 and rbx.
matvec:
    cmp dword [thread_count],1
    jne parallel_matvec
    jmp [matvec_impl]

matvec_scalar:
.row:
    xor eax,eax
    xorps xmm0,xmm0
.col:
    movzx r9d,word [rsi+rax*2]
    shl r9d,16
    movd xmm1,r9d
    mulss xmm1,[rdi+rax*4]
    addss xmm0,xmm1
    inc eax
    cmp eax,ecx
    jb .col
    movss [rdx],xmm0
    lea rsi,[rsi+rax*2]
    add rdx,4
    dec r8d
    jnz .row
    ret

; Model widths are multiples of 32. Four independent accumulators avoid a
; serial FMA dependency. BF16 expansion is exact and requires only AVX2.
matvec_avx2:
.row:
    xor eax,eax
    vxorps ymm0,ymm0,ymm0
    vxorps ymm1,ymm1,ymm1
    vxorps ymm2,ymm2,ymm2
    vxorps ymm3,ymm3,ymm3
.col:
    vpmovzxwd ymm4,[rsi+rax*2]
    vpmovzxwd ymm5,[rsi+rax*2+16]
    vpmovzxwd ymm6,[rsi+rax*2+32]
    vpmovzxwd ymm7,[rsi+rax*2+48]
    vpslld ymm4,ymm4,16
    vpslld ymm5,ymm5,16
    vpslld ymm6,ymm6,16
    vpslld ymm7,ymm7,16
    vfmadd231ps ymm0,ymm4,[rdi+rax*4]
    vfmadd231ps ymm1,ymm5,[rdi+rax*4+32]
    vfmadd231ps ymm2,ymm6,[rdi+rax*4+64]
    vfmadd231ps ymm3,ymm7,[rdi+rax*4+96]
    add eax,32
    cmp eax,ecx
    jb .col
    vaddps ymm0,ymm0,ymm1
    vaddps ymm2,ymm2,ymm3
    vaddps ymm0,ymm0,ymm2
    vextractf128 xmm1,ymm0,1
    vaddps xmm0,xmm0,xmm1
    vhaddps xmm0,xmm0,xmm0
    vhaddps xmm0,xmm0,xmm0
    vmovss [rdx],xmm0
    lea rsi,[rsi+rax*2]
    add rdx,4
    dec r8d
    jnz .row
    vzeroupper
    ret

; Select once at startup. XGETBV is safe only after OSXSAVE is checked.
init_kernels:
%ifndef FORCE_SCALAR
    push rbx
    mov eax,1
    cpuid
    and ecx,0x18001000
    cmp ecx,0x18001000
    jne .done
    xor ecx,ecx
    xgetbv
    and eax,6
    cmp eax,6
    jne .done
    mov eax,7
    xor ecx,ecx
    cpuid
    test ebx,1<<5
    jz .done
    mov qword [matvec_impl],matvec_avx2
    mov byte [vector_math],1
.done:
    pop rbx
%endif
    ret

norm:
    xor eax,eax
    xorps xmm0,xmm0
.sum:
    movss xmm1,[rdi+rax*4]
    mulss xmm1,xmm1
    addss xmm0,xmm1
    inc eax
    cmp eax,ecx
    jb .sum
    cvtsi2ss xmm1,ecx
    divss xmm0,xmm1
    addss xmm0,[epsilon]
    sqrtss xmm0,xmm0
    movss xmm3,[one]
    divss xmm3,xmm0
    xor eax,eax
.scale:
    movzx r9d,word [rsi+rax*2]
    shl r9d,16
    movd xmm2,r9d
    movss xmm1,[rdi+rax*4]
    mulss xmm1,xmm3
    mulss xmm1,xmm2
    movss [rdx+rax*4],xmm1
    inc eax
    cmp eax,ecx
    jb .scale
    ret

%include "expmath.asm"

; Prefill blocks read each weight row once for every token of the block.
; rdi = first input, r9 = input stride in bytes, rsi = BF16 weights,
; rdx = first output, ecx = width, r8d = rows, r10d = tokens. Outputs are
; [tokens][rows]. Rows are partitioned across workers as in run_job.
matvec_batch:
    mov [job_input],rdi
    mov [job_weights],rsi
    mov [job_output],rdx
    mov [job_width],ecx
    mov [job_rows],r8d
    mov [job_in_stride],r9
    mov [job_tokens],r10d
    mov qword [job_entry],run_batch_job
    cmp dword [thread_count],1
    jne dispatch_job
    xor eax,eax
    jmp run_batch_job

; eax = partition index. Last partition owns the remainder.
run_batch_job:
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
    test r8d,r8d
    jz .empty
    mov ecx,[job_width]
    mov rdx,[job_output]
    lea rdx,[rdx+rax*4]
    imul rax,rcx
    mov rsi,[job_weights]
    lea rsi,[rsi+rax*2]
    mov rdi,[job_input]
    mov r9,[job_in_stride]
    mov r10d,[job_tokens]
    mov r11d,[job_rows]
    shl r11,2
    jmp matvec_batch_avx2
.empty:
    ret

%macro BATCH_WEIGHTS 0
    vpmovzxwd ymm12,[rsi+rax*2]
    vpmovzxwd ymm13,[rsi+rax*2+16]
    vpmovzxwd ymm14,[rsi+rax*2+32]
    vpmovzxwd ymm15,[rsi+rax*2+48]
    vpslld ymm12,ymm12,16
    vpslld ymm13,ymm13,16
    vpslld ymm14,ymm14,16
    vpslld ymm15,ymm15,16
%endmacro
%macro BATCH_ZERO 4
    vxorps %1,%1,%1
    vxorps %2,%2,%2
    vxorps %3,%3,%3
    vxorps %4,%4,%4
%endmacro
%macro BATCH_FMA 5
    vfmadd231ps %1,ymm12,[%5+rax*4]
    vfmadd231ps %2,ymm13,[%5+rax*4+32]
    vfmadd231ps %3,ymm14,[%5+rax*4+64]
    vfmadd231ps %4,ymm15,[%5+rax*4+96]
%endmacro
; The reduction tree of matvec_avx2. Stores at r13, then advances one token.
%macro BATCH_STORE 6
    vaddps %1,%1,%2
    vaddps %3,%3,%4
    vaddps %1,%1,%3
    vextractf128 %6,%1,1
    vaddps %5,%5,%6
    vhaddps %5,%5,%5
    vhaddps %5,%5,%5
    vmovss [r13],%5
    add r13,r11
%endmacro

; r11 = output stride in bytes. Up to three tokens share each converted
; weight vector; every (row, token) dot product keeps the accumulators and
; reduction order of matvec_avx2, so prefill is bit-identical to decode.
matvec_batch_avx2:
    push rbx
    push r12
    push r13
    push r14
    push r15
.row:
    mov r12,rdi
    mov r13,rdx
    mov ebx,r10d
.group:
    cmp ebx,3
    jb .small
    lea r14,[r12+r9]
    lea r15,[r14+r9]
    BATCH_ZERO ymm0,ymm1,ymm2,ymm3
    BATCH_ZERO ymm4,ymm5,ymm6,ymm7
    BATCH_ZERO ymm8,ymm9,ymm10,ymm11
    xor eax,eax
.three:
    BATCH_WEIGHTS
    BATCH_FMA ymm0,ymm1,ymm2,ymm3,r12
    BATCH_FMA ymm4,ymm5,ymm6,ymm7,r14
    BATCH_FMA ymm8,ymm9,ymm10,ymm11,r15
    add eax,32
    cmp eax,ecx
    jb .three
    BATCH_STORE ymm0,ymm1,ymm2,ymm3,xmm0,xmm1
    BATCH_STORE ymm4,ymm5,ymm6,ymm7,xmm4,xmm5
    BATCH_STORE ymm8,ymm9,ymm10,ymm11,xmm8,xmm9
    lea r12,[r15+r9]
    sub ebx,3
    jnz .group
    jmp .next_row
.small:
    cmp ebx,2
    jb .one
    lea r14,[r12+r9]
    BATCH_ZERO ymm0,ymm1,ymm2,ymm3
    BATCH_ZERO ymm4,ymm5,ymm6,ymm7
    xor eax,eax
.two:
    BATCH_WEIGHTS
    BATCH_FMA ymm0,ymm1,ymm2,ymm3,r12
    BATCH_FMA ymm4,ymm5,ymm6,ymm7,r14
    add eax,32
    cmp eax,ecx
    jb .two
    BATCH_STORE ymm0,ymm1,ymm2,ymm3,xmm0,xmm1
    BATCH_STORE ymm4,ymm5,ymm6,ymm7,xmm4,xmm5
    jmp .next_row
.one:
    BATCH_ZERO ymm0,ymm1,ymm2,ymm3
    xor eax,eax
.single:
    BATCH_WEIGHTS
    BATCH_FMA ymm0,ymm1,ymm2,ymm3,r12
    add eax,32
    cmp eax,ecx
    jb .single
    BATCH_STORE ymm0,ymm1,ymm2,ymm3,xmm0,xmm1
.next_row:
    lea rsi,[rsi+rcx*2]
    add rdx,4
    dec r8d
    jnz .row
    vzeroupper
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbx
    ret

; rdi = projected B,C,x, rsi = weights [channel,3], rdx = two history planes
conv_step:
    xor ecx,ecx
.channel:
    movss xmm0,[rdi+rcx*4]
    mulss xmm0,[rdi+rcx*4+8192]
    movss xmm1,[rdx+rcx*4]
    movss xmm2,[rdx+rcx*4+4096]
    movss [rdx+rcx*4],xmm2
    movss [rdx+rcx*4+4096],xmm0
    movzx eax,word [rsi]
    shl eax,16
    movd xmm3,eax
    mulss xmm1,xmm3
    movzx eax,word [rsi+2]
    shl eax,16
    movd xmm3,eax
    mulss xmm2,xmm3
    addss xmm1,xmm2
    movzx eax,word [rsi+4]
    shl eax,16
    movd xmm3,eax
    mulss xmm0,xmm3
    addss xmm1,xmm0
    mulss xmm1,[rdi+rcx*4+4096]
    movss [rdi+rcx*4],xmm1
    add rsi,6
    inc ecx
    cmp ecx,1024
    jb .channel
    ret
