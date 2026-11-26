; LFM3 kernels: Q8_K activations against Q4_K and Q6_K rows, FP32 norms,
; the seven-tap short convolution and windowed, gated attention. AVX2, FMA
; and F16C. The quantized dot products follow ggml's formats: every matrix
; input is quantized once per row into 256-value Q8_K blocks.

section .text
; Select once at startup. XGETBV is safe only after OSXSAVE is checked.
init_kernels:
    push rbx
    mov eax,1
    cpuid
    and ecx,0x38001000          ; FMA, OSXSAVE, AVX, F16C
    cmp ecx,0x38001000
    jne .unsupported
    xor ecx,ecx
    xgetbv
    and eax,6
    cmp eax,6
    jne .unsupported
    mov eax,7
    xor ecx,ecx
    cpuid
    test ebx,1<<5
    jz .unsupported
    mov byte [vector_math],1
    pop rbx
    ret
.unsupported:
    mov edi,2
    mov esi,cpu_error
    mov edx,cpu_error_len
    call write_all
    mov edi,1
    jmp exit

; Horizontal sum of a ymm register into the low lane of its xmm.
%macro HSUM 3
    vextractf128 %3,%1,1
    vaddps %2,%2,%3
    vmovhlps %3,%3,%2
    vaddps %2,%2,%3
    vmovshdup %3,%2
    vaddss %2,%2,%3
%endmacro

; rdi = FP32 source, rsi = Q8_K blocks, ecx = values (a multiple of 256).
; Values scale by 127 / max |x| and round to nearest; bsums hold the sums of
; each 16 quantized values.
quantize_q8k:
    shr ecx,8
    vmovups ymm7,[abs_mask8]
    vmovdqu ymm8,[pack_order8]
.block:
    vxorps ymm0,ymm0,ymm0
    xor eax,eax
.max:
    vandps ymm1,ymm7,[rdi+rax*4]
    vmaxps ymm0,ymm0,ymm1
    add eax,8
    cmp eax,256
    jb .max
    vextractf128 xmm1,ymm0,1
    vmaxps xmm0,xmm0,xmm1
    vmovhlps xmm1,xmm1,xmm0
    vmaxps xmm0,xmm0,xmm1
    vmovshdup xmm1,xmm0
    vmaxss xmm0,xmm0,xmm1
    vxorps xmm1,xmm1,xmm1
    vucomiss xmm0,xmm1
    je .zero
    vmovss xmm1,[f127]
    vdivss xmm1,xmm1,xmm0
    vbroadcastss ymm6,xmm1
    vmovss xmm2,[one]
    vdivss xmm2,xmm2,xmm1
    vmovss [rsi],xmm2
    xor eax,eax
.group:
    vmulps ymm0,ymm6,[rdi+rax*4]
    vmulps ymm1,ymm6,[rdi+rax*4+32]
    vmulps ymm2,ymm6,[rdi+rax*4+64]
    vmulps ymm3,ymm6,[rdi+rax*4+96]
    vcvtps2dq ymm0,ymm0
    vcvtps2dq ymm1,ymm1
    vcvtps2dq ymm2,ymm2
    vcvtps2dq ymm3,ymm3
    vpaddd ymm4,ymm0,ymm1
    vpaddd ymm5,ymm2,ymm3
    vphaddd ymm4,ymm4,ymm5
    vphaddd ymm4,ymm4,ymm4
    vextracti128 xmm5,ymm4,1
    vpaddd xmm4,xmm4,xmm5
    mov edx,eax
    shr edx,3                   ; two int16 sums per 32 values
    vpextrw [rsi+260+rdx],xmm4,0
    vpextrw [rsi+262+rdx],xmm4,2
    vpackssdw ymm0,ymm0,ymm1
    vpackssdw ymm2,ymm2,ymm3
    vpacksswb ymm0,ymm0,ymm2
    vpermd ymm0,ymm8,ymm0
    vmovdqu [rsi+4+rax],ymm0
    add eax,32
    cmp eax,256
    jb .group
.next:
    add rdi,1024
    add rsi,Q8K_BYTES
    dec ecx
    jnz .block
    vzeroupper
    ret
.zero:
    vpxor xmm0,xmm0,xmm0
    xor eax,eax
.clear:
    vmovdqu [rsi+rax],ymm0
    add eax,32
    cmp eax,288
    jb .clear
    mov dword [rsi+288],0
    jmp .next

; rdi = Q8_K input, rsi = Q4_K rows, rdx = FP32 output, ecx = width,
; r8d = rows. Each block: f16 d and dmin, twelve bytes of packed 6-bit scales
; and mins, then 128 bytes of nibbles. Integer products per block, then
; d * dy * sum(scale * q * a) - dmin * dy * sum(min * bsum).
matvec_q4k:
    push rbx
    push r12
    push r13
    shr ecx,8
    mov r12d,ecx
    vmovdqu ymm15,[nibble_mask8]
.row:
    vxorps ymm0,ymm0,ymm0
    vxorps xmm1,xmm1,xmm1
    mov r13,rdi
    mov ebx,r12d
.block:
    mov eax,[rsi+4]
    mov r9d,[rsi+8]
    mov r10d,[rsi+12]
    mov r11d,r10d
    shr r11d,4
    and r11d,0x0f0f0f0f
    mov ecx,r9d
    shr ecx,6
    and ecx,0x03030303
    shl ecx,4
    or r11d,ecx                 ; mins 4-7
    mov ecx,eax
    shr ecx,6
    and ecx,0x03030303
    shl ecx,4
    and r10d,0x0f0f0f0f
    or r10d,ecx                 ; scales 4-7
    and r9d,0x3f3f3f3f          ; mins 0-3
    and eax,0x3f3f3f3f          ; scales 0-3
    vmovd xmm2,eax
    vpinsrd xmm2,xmm2,r10d,1
    vpinsrd xmm2,xmm2,r9d,2
    vpinsrd xmm2,xmm2,r11d,3
    vpmovzxbw ymm2,xmm2         ; words 0-7 scales, 8-15 mins
    vmovd xmm3,[rsi]
    vcvtph2ps xmm3,xmm3
    vbroadcastss xmm4,[r13]
    vmulps xmm3,xmm3,xmm4       ; d * dy, dmin * dy
    vmovdqu ymm5,[r13+260]
    vextracti128 xmm6,ymm5,1
    vphaddw xmm5,xmm5,xmm6      ; sums of 32 values
    vextracti128 xmm6,ymm2,1
    vpmaddwd xmm5,xmm6,xmm5
    vcvtdq2ps xmm5,xmm5
    vmovshdup xmm6,xmm3
    vbroadcastss xmm6,xmm6
    vfmadd231ps xmm1,xmm6,xmm5
    vinserti128 ymm2,ymm2,xmm2,1
    vpxor ymm7,ymm7,ymm7
%assign j 0
%rep 4
    vpshufb ymm8,ymm2,[q4k_scale_low+j*32]
    vpshufb ymm9,ymm2,[q4k_scale_high+j*32]
    vmovdqu ymm10,[rsi+16+j*32]
    vpand ymm11,ymm10,ymm15
    vpsrlw ymm10,ymm10,4
    vpand ymm10,ymm10,ymm15
    vpmaddubsw ymm11,ymm11,[r13+4+j*64]
    vpmaddubsw ymm10,ymm10,[r13+4+j*64+32]
    vpmaddwd ymm11,ymm8,ymm11
    vpmaddwd ymm10,ymm9,ymm10
    vpaddd ymm7,ymm7,ymm11
    vpaddd ymm7,ymm7,ymm10
%assign j j+1
%endrep
    vcvtdq2ps ymm7,ymm7
    vbroadcastss ymm6,xmm3
    vfmadd231ps ymm0,ymm6,ymm7
    add rsi,Q4K_BYTES
    add r13,Q8K_BYTES
    dec ebx
    jnz .block
    HSUM ymm0,xmm0,xmm2
    vmovhlps xmm2,xmm2,xmm1
    vaddps xmm1,xmm1,xmm2
    vmovshdup xmm2,xmm1
    vaddss xmm1,xmm1,xmm2
    vsubss xmm0,xmm0,xmm1
    vmovss [rdx],xmm0
    add rdx,4
    dec r8d
    jnz .row
    vzeroupper
    pop r13
    pop r12
    pop rbx
    ret

; rdi = Q8_K input, rsi = rows, rdx = output, ecx = width, r8d = rows,
; r9 = kernel, r10d = bytes per 256-value weight block. Rows are partitioned
; across workers; each row's arithmetic does not depend on the partition.
qmatvec:
    cmp dword [thread_count],1
    je .direct
    mov [job_input],rdi
    mov [job_weights],rsi
    mov [job_output],rdx
    mov [job_width],ecx
    mov [job_rows],r8d
    mov [job_kernel],r9
    mov [job_block_bytes],r10d
    mov qword [job_entry],run_qjob
    jmp dispatch_job
.direct:
    jmp r9

; eax = partition index. Last partition owns the remainder.
run_qjob:
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
    mov r9d,ecx
    shr r9d,8
    imul r9d,[job_block_bytes]
    imul rax,r9
    mov rsi,[job_weights]
    add rsi,rax
    mov rdi,[job_input]
    jmp [job_kernel]
.empty:
    ret

; Unpack one Q6_K block at rsi into ymm6-ymm13: 256 values 0..63 in order.
; ymm14 = 0x03 bytes, ymm15 = 0x0f bytes; ymm1-ymm5 are clobbered.
%macro Q6_UNPACK 0
%assign n 0
%rep 2
    vmovdqu ymm1,[rsi+n*64]
    vmovdqu ymm2,[rsi+n*64+32]
    vmovdqu ymm3,[rsi+128+n*32]
    vpand ymm4,ymm3,ymm14
    vpsllw ymm4,ymm4,4
    vpand ymm5,ymm1,ymm15
%assign r 6+n*4
    vpor ymm%[r],ymm5,ymm4
    vpsrlw ymm4,ymm3,2
    vpand ymm4,ymm4,ymm14
    vpsllw ymm4,ymm4,4
    vpand ymm5,ymm2,ymm15
%assign r 7+n*4
    vpor ymm%[r],ymm5,ymm4
    vpsrlw ymm4,ymm3,4
    vpand ymm4,ymm4,ymm14
    vpsllw ymm4,ymm4,4
    vpsrlw ymm5,ymm1,4
    vpand ymm5,ymm5,ymm15
%assign r 8+n*4
    vpor ymm%[r],ymm5,ymm4
    vpsrlw ymm4,ymm3,6
    vpand ymm4,ymm4,ymm14
    vpsllw ymm4,ymm4,4
    vpsrlw ymm5,ymm2,4
    vpand ymm5,ymm5,ymm15
%assign r 9+n*4
    vpor ymm%[r],ymm5,ymm4
%assign n n+1
%endrep
%endmacro

; rdi = Q8_K input, rsi = Q6_K rows, rdx = FP32 output, ecx = width,
; r8d = rows. Each block: 128 bytes of low nibbles, 64 of high bit pairs,
; sixteen int8 scales and an f16 d. The -32 offset is folded in through the
; activation sums: sum(scale * (q - 32) * a) = sum(scale * q * a) - 32 sum(scale * bsum).
matvec_q6k:
    push rbx
    push r12
    push r13
    shr ecx,8
    mov r12d,ecx
    vmovdqu ymm14,[pair_mask8]
    vmovdqu ymm15,[nibble_mask8]
.row:
    vxorps ymm0,ymm0,ymm0
    mov r13,rdi
    mov ebx,r12d
.block:
    Q6_UNPACK
    movzx eax,word [rsi+208]
    vmovd xmm1,eax
    vcvtph2ps xmm1,xmm1
    vmulss xmm1,xmm1,[r13]
    vmovdqu xmm2,[rsi+192]
    vpmovsxbw ymm3,xmm2
    vpmaddwd ymm3,ymm3,[r13+260]
    vpslld ymm3,ymm3,5
    vpxor ymm4,ymm4,ymm4
    vpsubd ymm4,ymm4,ymm3
%assign j 0
%rep 8
    vpshufb xmm5,xmm2,[q6k_scales+j*16]
    vpmovsxbw ymm5,xmm5
%assign r 6+j
    vpmaddubsw ymm3,ymm%[r],[r13+4+j*32]
    vpmaddwd ymm3,ymm5,ymm3
    vpaddd ymm4,ymm4,ymm3
%assign j j+1
%endrep
    vcvtdq2ps ymm4,ymm4
    vbroadcastss ymm1,xmm1
    vfmadd231ps ymm0,ymm1,ymm4
    add rsi,Q6K_BYTES
    add r13,Q8K_BYTES
    dec ebx
    jnz .block
    HSUM ymm0,xmm0,xmm2
    vmovss [rdx],xmm0
    add rdx,4
    dec r8d
    jnz .row
    vzeroupper
    pop r13
    pop r12
    pop rbx
    ret

; Prefill blocks. rdi = Q8_K inputs of consecutive tokens, rsi = rows, rdx =
; outputs [tokens][rows], ecx = width, r8d = rows, r9 = batch kernel, r10d =
; tokens, r11d = bytes per weight block. The kernels then take r9 = input
; stride and r11 = output stride in bytes.
qmatvec_batch:
    mov [job_input],rdi
    mov [job_weights],rsi
    mov [job_output],rdx
    mov [job_width],ecx
    mov [job_rows],r8d
    mov [job_tokens],r10d
    mov [job_kernel],r9
    mov [job_block_bytes],r11d
    mov qword [job_entry],run_qbatch_job
    cmp dword [thread_count],1
    jne dispatch_job
    xor eax,eax
    jmp run_qbatch_job

run_qbatch_job:
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
    mov r9d,ecx
    shr r9d,8
    imul r9d,[job_block_bytes]
    imul rax,r9
    mov rsi,[job_weights]
    add rsi,rax
    mov rdi,[job_input]
    mov r9d,[job_width]
    shr r9d,8
    imul r9d,r9d,Q8K_BYTES        ; each token holds width / 256 Q8_K blocks
    mov r10d,[job_tokens]
    mov r11d,[job_rows]
    shl r11,2
    jmp [job_kernel]
.empty:
    ret

; Stack frame of the batch kernels: per token a ymm and an xmm accumulator.
%define BATCH_ACC 64
%define BATCH_LOCALS PREFILL_BLOCK*BATCH_ACC+32

; Each block's nibbles, scales and mins are unpacked once and reused by every
; token. Per (row, token) the integer sums, the FMA per block and the final
; reductions are those of matvec_q4k, so results are bit-identical.
matvec_q4k_batch:
    push rbx
    push rbp
    push r12
    push r13
    push r14
    push r15
    mov rbp,rsp
    sub rsp,BATCH_LOCALS
    and rsp,-32
    shr ecx,8
    mov [rsp+PREFILL_BLOCK*BATCH_ACC],ecx
    mov r15,rdx
.row:
    xor eax,eax
    vxorps ymm0,ymm0,ymm0
.clear:
    vmovaps [rsp+rax],ymm0
    vmovaps [rsp+rax+32],xmm0
    add eax,BATCH_ACC
    cmp eax,PREFILL_BLOCK*BATCH_ACC
    jb .clear
    mov ebx,[rsp+PREFILL_BLOCK*BATCH_ACC]
    xor r14d,r14d                ; block offset inside each token's input
.block:
    mov eax,[rsi+4]
    mov ecx,[rsi+8]
    mov edx,[rsi+12]
    mov r12d,edx
    shr r12d,4
    and r12d,0x0f0f0f0f
    mov r13d,ecx
    shr r13d,6
    and r13d,0x03030303
    shl r13d,4
    or r12d,r13d                ; mins 4-7
    mov r13d,eax
    shr r13d,6
    and r13d,0x03030303
    shl r13d,4
    and edx,0x0f0f0f0f
    or edx,r13d                 ; scales 4-7
    and ecx,0x3f3f3f3f          ; mins 0-3
    and eax,0x3f3f3f3f          ; scales 0-3
    vmovd xmm2,eax
    vpinsrd xmm2,xmm2,edx,1
    vpinsrd xmm2,xmm2,ecx,2
    vpinsrd xmm2,xmm2,r12d,3
    vpmovzxbw ymm2,xmm2
    vextracti128 xmm3,ymm2,1    ; mins
    vinserti128 ymm2,ymm2,xmm2,1
    vmovd xmm4,[rsi]
    vcvtph2ps xmm4,xmm4         ; d, dmin
%assign j 0
%rep 4
%assign r 8+2*j
    vmovdqu ymm1,[rsi+16+j*32]
    vpand ymm%[r],ymm1,[nibble_mask8]
%assign r 9+2*j
    vpsrlw ymm1,ymm1,4
    vpand ymm%[r],ymm1,[nibble_mask8]
%assign j j+1
%endrep
    lea r13,[rdi+r14]
    mov rax,rsp
    mov r12d,r10d
.token:
    vbroadcastss xmm0,[r13]
    vmulps xmm0,xmm4,xmm0       ; d * dy, dmin * dy
    vmovdqu ymm5,[r13+260]
    vextracti128 xmm6,ymm5,1
    vphaddw xmm5,xmm5,xmm6
    vpmaddwd xmm5,xmm3,xmm5
    vcvtdq2ps xmm5,xmm5
    vmovshdup xmm6,xmm0
    vbroadcastss xmm6,xmm6
    vmovaps xmm7,[rax+32]
    vfmadd231ps xmm7,xmm6,xmm5
    vmovaps [rax+32],xmm7
    vpxor ymm5,ymm5,ymm5
%assign j 0
%rep 4
%assign r 8+2*j
    vpshufb ymm6,ymm2,[q4k_scale_low+j*32]
    vpmaddubsw ymm1,ymm%[r],[r13+4+j*64]
    vpmaddwd ymm1,ymm6,ymm1
%assign r 9+2*j
    vpshufb ymm6,ymm2,[q4k_scale_high+j*32]
    vpmaddubsw ymm7,ymm%[r],[r13+4+j*64+32]
    vpmaddwd ymm7,ymm6,ymm7
    vpaddd ymm5,ymm5,ymm1
    vpaddd ymm5,ymm5,ymm7
%assign j j+1
%endrep
    vcvtdq2ps ymm5,ymm5
    vbroadcastss ymm6,xmm0
    vmovaps ymm7,[rax]
    vfmadd231ps ymm7,ymm6,ymm5
    vmovaps [rax],ymm7
    add rax,BATCH_ACC
    add r13,r9
    dec r12d
    jnz .token
    add rsi,Q4K_BYTES
    add r14,Q8K_BYTES
    dec ebx
    jnz .block
    mov rax,rsp
    mov r13,r15
    mov r12d,r10d
.store:
    vmovaps ymm0,[rax]
    vmovaps xmm1,[rax+32]
    HSUM ymm0,xmm0,xmm2
    vmovhlps xmm2,xmm2,xmm1
    vaddps xmm1,xmm1,xmm2
    vmovshdup xmm2,xmm1
    vaddss xmm1,xmm1,xmm2
    vsubss xmm0,xmm0,xmm1
    vmovss [r13],xmm0
    add rax,BATCH_ACC
    add r13,r11
    dec r12d
    jnz .store
    add r15,4
    dec r8d
    jnz .row
    vzeroupper
    mov rsp,rbp
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbp
    pop rbx
    ret

; Q6_K counterpart of matvec_q4k_batch, bit-identical to matvec_q6k.
matvec_q6k_batch:
    push rbx
    push rbp
    push r12
    push r13
    push r14
    push r15
    mov rbp,rsp
    sub rsp,BATCH_LOCALS
    and rsp,-32
    shr ecx,8
    mov [rsp+PREFILL_BLOCK*BATCH_ACC],ecx
    mov r15,rdx
.row:
    xor eax,eax
    vxorps ymm0,ymm0,ymm0
.clear:
    vmovaps [rsp+rax],ymm0
    add eax,BATCH_ACC
    cmp eax,PREFILL_BLOCK*BATCH_ACC
    jb .clear
    mov ebx,[rsp+PREFILL_BLOCK*BATCH_ACC]
    xor r14d,r14d
.block:
    vmovdqu ymm14,[pair_mask8]
    vmovdqu ymm15,[nibble_mask8]
    Q6_UNPACK
    movzx eax,word [rsi+208]
    vmovd xmm15,eax
    vcvtph2ps xmm15,xmm15       ; d
    vmovdqu xmm2,[rsi+192]
    vpmovsxbw ymm14,xmm2
    lea r13,[rdi+r14]
    mov rax,rsp
    mov r12d,r10d
.token:
    vmulss xmm0,xmm15,[r13]     ; d * dy
    vpmaddwd ymm3,ymm14,[r13+260]
    vpslld ymm3,ymm3,5
    vpxor ymm4,ymm4,ymm4
    vpsubd ymm4,ymm4,ymm3
%assign j 0
%rep 8
%assign r 6+j
    vpshufb xmm5,xmm2,[q6k_scales+j*16]
    vpmovsxbw ymm5,xmm5
    vpmaddubsw ymm3,ymm%[r],[r13+4+j*32]
    vpmaddwd ymm3,ymm5,ymm3
    vpaddd ymm4,ymm4,ymm3
%assign j j+1
%endrep
    vcvtdq2ps ymm4,ymm4
    vbroadcastss ymm0,xmm0
    vmovaps ymm1,[rax]
    vfmadd231ps ymm1,ymm0,ymm4
    vmovaps [rax],ymm1
    add rax,BATCH_ACC
    add r13,r9
    dec r12d
    jnz .token
    add rsi,Q6K_BYTES
    add r14,Q8K_BYTES
    dec ebx
    jnz .block
    mov rax,rsp
    mov r13,r15
    mov r12d,r10d
.store:
    vmovaps ymm0,[rax]
    HSUM ymm0,xmm0,xmm2
    vmovss [r13],xmm0
    add rax,BATCH_ACC
    add r13,r11
    dec r12d
    jnz .store
    add r15,4
    dec r8d
    jnz .row
    vzeroupper
    mov rsp,rbp
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbp
    pop rbx
    ret

; rsi = Q6_K row, rdi = FP32 output, ecx = width. y = (d * scale) * (q - 32),
; the order of ggml's dequantization.
dequant_q6k_row:
    shr ecx,8
    vmovdqu ymm14,[pair_mask8]
    vmovdqu ymm15,[nibble_mask8]
.block:
    Q6_UNPACK
%assign j 0
%rep 8
%assign r 6+j
    vmovdqu [q6k_values+j*32],ymm%[r]
%assign j j+1
%endrep
    movzx eax,word [rsi+208]
    vmovd xmm1,eax
    vcvtph2ps xmm1,xmm1
    xor edx,edx
.value:
    mov eax,edx
    shr eax,4
    movsx eax,byte [rsi+192+rax]
    vcvtsi2ss xmm2,xmm2,eax
    vmulss xmm2,xmm2,xmm1
    movzx eax,byte [q6k_values+rdx]
    sub eax,32
    vcvtsi2ss xmm3,xmm3,eax
    vmulss xmm2,xmm2,xmm3
    vmovss [rdi+rdx*4],xmm2
    inc edx
    cmp edx,256
    jb .value
    add rsi,Q6K_BYTES
    add rdi,1024
    dec ecx
    jnz .block
    vzeroupper
    ret

; rdi = source, rsi = FP32 weights, rdx = destination, ecx = width (a
; multiple of eight). x * (1 / sqrt(mean(x^2) + eps)) * weight; in place is safe.
norm_f32:
    vxorps ymm0,ymm0,ymm0
    xor eax,eax
.sum:
    vmovups ymm1,[rdi+rax*4]
    vfmadd231ps ymm0,ymm1,ymm1
    add eax,8
    cmp eax,ecx
    jb .sum
    HSUM ymm0,xmm0,xmm1
    vcvtsi2ss xmm1,xmm1,ecx
    vdivss xmm0,xmm0,xmm1
    vaddss xmm0,xmm0,[epsilon]
    vsqrtss xmm0,xmm0,xmm0
    vmovss xmm1,[one]
    vdivss xmm1,xmm1,xmm0
    vbroadcastss ymm1,xmm1
    xor eax,eax
.scale:
    vmulps ymm2,ymm1,[rdi+rax*4]
    vmulps ymm2,ymm2,[rsi+rax*4]
    vmovups [rdx+rax*4],ymm2
    add eax,8
    cmp eax,ecx
    jb .scale
    vzeroupper
    ret

; hidden += update over 2048 lanes.
residual:
    xor eax,eax
.add:
    vmovups ymm0,[hidden+rax*4]
    vaddps ymm0,ymm0,[update+rax*4]
    vmovups [hidden+rax*4],ymm0
    add eax,8
    cmp eax,2048
    jb .add
    vzeroupper
    ret

; rdi = projected B, C, x (3 x 2048), rsi = FP32 taps [channel][7], rdx = six
; history planes of B*x. The current B*x meets tap 6; the result C * conv
; replaces B in place.
conv_step:
    xor ecx,ecx
.channel:
    vmovss xmm0,[rdi+rcx*4]
    vmulss xmm0,xmm0,[rdi+rcx*4+16384]
    vmovss xmm1,[rdx+rcx*4]
    vmulss xmm1,xmm1,[rsi]
%assign k 1
%rep 5
    vmovss xmm2,[rdx+rcx*4+k*8192]
    vmovss [rdx+rcx*4+(k-1)*8192],xmm2
    vfmadd231ss xmm1,xmm2,[rsi+k*4]
%assign k k+1
%endrep
    vmovss [rdx+rcx*4+5*8192],xmm0
    vfmadd231ss xmm1,xmm0,[rsi+24]
    vmulss xmm1,xmm1,[rdi+rcx*4+8192]
    vmovss [rdi+rcx*4],xmm1
    add rsi,28
    inc ecx
    cmp ecx,2048
    jb .channel
    ret

; Cos and sin at position r14d for both RoPE kinds; angles are FP32 products
; of the position and the inverse frequency, as in modeling_lfm3.py.
prepare_rope:
    xor ecx,ecx
.full:
    vcvtsi2ss xmm0,xmm0,r14d
    vmulss xmm0,xmm0,[freq_full+rcx*4]
    vmovss [scratch],xmm0
    fld dword [scratch]
    fsincos
    fstp dword [rope_full+rcx*4]
    fstp dword [rope_full+256+rcx*4]
    inc ecx
    cmp ecx,32
    jb .full
    xor ecx,ecx
.window:
    vcvtsi2ss xmm0,xmm0,r14d
    vmulss xmm0,xmm0,[freq_swa+rcx*4]
    vmovss [scratch],xmm0
    fld dword [scratch]
    fsincos
    fstp dword [rope_swa+rcx*4]
    fstp dword [rope_swa+256+rcx*4]
    inc ecx
    cmp ecx,64
    jb .window
    ret

; rdi = heads of 128 lanes, esi = head count, rdx = FP32 norm weights,
; ecx = rotated pairs, r8 = cos table (sin 256 bytes later). Pairs are
; (i, i + 64); pairs past ecx have a zero frequency and stay unrotated.
head_norm_rope:
    push rbx
    push r12
    push r13
    push r14
    mov rbx,rdi
    mov r12d,esi
    mov r13,rdx
    mov r14d,ecx
.head:
    mov rdi,rbx
    mov rsi,r13
    mov rdx,rbx
    mov ecx,128
    call norm_f32
    xor ecx,ecx
.pair:
    vmovss xmm2,[r8+rcx*4]
    vmovss xmm3,[r8+256+rcx*4]
    vmovss xmm0,[rbx+rcx*4]
    vmovss xmm1,[rbx+rcx*4+256]
    vmulss xmm4,xmm0,xmm2
    vmulss xmm5,xmm1,xmm3
    vsubss xmm4,xmm4,xmm5
    vmulss xmm5,xmm1,xmm2
    vmulss xmm6,xmm0,xmm3
    vaddss xmm5,xmm5,xmm6
    vmovss [rbx+rcx*4],xmm4
    vmovss [rbx+rcx*4+256],xmm5
    inc ecx
    cmp ecx,r14d
    jb .pair
    add rbx,512
    dec r12d
    jnz .head
    pop r14
    pop r13
    pop r12
    pop rbx
    ret

; headgate = sigmoid(headgate) for the 16 heads.
sigmoid16:
%assign k 0
%rep 2
    vxorps ymm0,ymm0,ymm0
    vsubps ymm0,ymm0,[headgate+k*32]
    call exp_avx2
    vaddps ymm0,ymm0,[one8]
    vmovups ymm1,[one8]
    vdivps ymm1,ymm1,ymm0
    vmovups [headgate+k*32],ymm1
%assign k k+1
%endrep
    vzeroupper
    ret

; r12 = layer cache rows of 4096 bytes (keys of 4 heads, then values),
; r14d = position, r10d = first visible position. Heads are independent, so
; partitions compute whole heads with their own score rows and every head is
; bit-identical at any thread count.
attention:
    mov [job_kv],r12
    mov [job_position],r14d
    mov [job_window],r10d
    mov qword [job_entry],attention_job
    cmp dword [thread_count],1
    jne dispatch_job
    xor eax,eax
    jmp attention_job

; eax = partition index; heads [index*16/threads, (index+1)*16/threads).
attention_job:
    push r12
    push r13
    push r14
    mov r12,[job_kv]
    mov r14d,[job_position]
    mov r10d,[job_window]
    mov r9d,eax
    imul esi,eax,MAX_CONTEXT*4
    add rsi,scores
    shl eax,4
    xor edx,edx
    div dword [thread_count]
    mov r8d,eax
    lea eax,[r9+1]
    shl eax,4
    xor edx,edx
    div dword [thread_count]
    mov edi,eax
    call attention_heads
    pop r14
    pop r13
    pop r12
    ret

; Heads r8d up to edi, score row rsi. Each query head reads key/value head
; h / 4; its output is scaled by gate / sum(exp). r13 is clobbered.
attention_heads:
.head:
    mov eax,r8d
    shr eax,2
    shl eax,9
    mov r9d,r10d
    shl r9,12
    add r9,r12
    add r9,rax
    mov eax,r8d
    shl eax,9
    lea r11,[query+rax]
    mov ecx,r10d
    xor edx,edx
    vmovss xmm6,[negative_inf]
.score:
    vmovups ymm0,[r11]
    vmulps ymm0,ymm0,[r9]
    vmovups ymm1,[r11+32]
    vmulps ymm1,ymm1,[r9+32]
%assign k 2
%rep 14
    vmovups ymm2,[r11+k*32]
%assign r k & 1
    vfmadd231ps ymm%[r],ymm2,[r9+k*32]
%assign k k+1
%endrep
    vaddps ymm0,ymm0,ymm1
    HSUM ymm0,xmm0,xmm1
    vmulss xmm0,xmm0,[attention_scale]
    vmovss [rsi+rdx*4],xmm0
    vmaxss xmm6,xmm6,xmm0
    add r9,4096
    inc edx
    inc ecx
    cmp ecx,r14d
    jbe .score
    ; edx = visible positions. Masked final vector, as in the LFM2 softmax.
    vbroadcastss ymm4,xmm6
    vxorps ymm5,ymm5,ymm5
    xor ecx,ecx
    mov eax,edx
.softmax:
    mov edx,eax
    sub edx,ecx
    jbe .sum
    vmovd xmm6,edx
    vpbroadcastd ymm6,xmm6
    vpcmpgtd ymm6,ymm6,[lane_index8]
    vmaskmovps ymm0,ymm6,[rsi+rcx*4]
    vsubps ymm0,ymm0,ymm4
    call exp_avx2
    vandps ymm0,ymm0,ymm6
    vmaskmovps [rsi+rcx*4],ymm6,ymm0
    vaddps ymm5,ymm5,ymm0
    add ecx,8
    jmp .softmax
.sum:
    mov r13d,eax
    HSUM ymm5,xmm5,xmm1
    vmovss xmm7,[headgate+r8*4]
    vdivss xmm7,xmm7,xmm5
    vbroadcastss ymm7,xmm7
    mov eax,r8d
    shl eax,9
    lea r11,[mixed+rax]
%assign half 0
%rep 2
    mov eax,r8d
    shr eax,2
    shl eax,9
    mov r9d,r10d
    shl r9,12
    add r9,r12
    lea r9,[r9+rax+2048+half*256]
%assign k 8
%rep 8
    vxorps ymm%[k],ymm%[k],ymm%[k]
%assign k k+1
%endrep
    xor ecx,ecx
.weighted%[half]:
    vbroadcastss ymm0,[rsi+rcx*4]
%assign k 0
%rep 8
%assign r 8+k
    vfmadd231ps ymm%[r],ymm0,[r9+k*32]
%assign k k+1
%endrep
    add r9,4096
    inc ecx
    cmp ecx,r13d
    jb .weighted%[half]
%assign k 0
%rep 8
%assign r 8+k
    vmulps ymm%[r],ymm%[r],ymm7
    vmovups [r11+half*256+k*32],ymm%[r]
%assign k k+1
%endrep
%assign half half+1
%endrep
    inc r8d
    cmp r8d,edi
    jb .head
    vzeroupper
    ret

section .rodata
cpu_error: db "LFM3 needs a CPU with AVX2, FMA and F16C",10
cpu_error_len equ $-cpu_error
align 32
abs_mask8: times 8 dd 0x7fffffff
pack_order8: dd 0,4,1,5,2,6,3,7
nibble_mask8: times 32 db 0x0f
pair_mask8: times 32 db 0x03
; Q4_K: word 2j of the scales for the low nibbles of chunk j, word 2j+1 for the high.
q4k_scale_low:
%assign j 0
%rep 4
    times 16 db 4*j, 4*j+1
%assign j j+1
%endrep
q4k_scale_high:
%assign j 0
%rep 4
    times 16 db 4*j+2, 4*j+3
%assign j j+1
%endrep
; Q6_K: scale 2j for values 32j..32j+15, scale 2j+1 for the next sixteen.
q6k_scales:
%assign j 0
%rep 8
    times 8 db 2*j
    times 8 db 2*j+1
%assign j j+1
%endrep
f127: dd 127.0
attention_scale: dd 0.08838834764831845    ; head_dim ** -0.5

section .bss
alignb 32
q6k_values: resb 256
job_kernel: resq 1
job_block_bytes: resd 1
job_window: resd 1


section .text
