; Each position uses the same rotation in every attention layer and head.
prepare_rope:
    xor ecx,ecx
.pair:
    cvtsi2ss xmm0,r14d
    mulss xmm0,[freq+rcx*4]
    movss [scratch],xmm0
    fld dword [scratch]
    fsincos
    fstp dword [rope_cos+rcx*4]
    fstp dword [rope_sin+rcx*4]
    inc ecx
    cmp ecx,32
    jb .pair
    ret

; One head at a time. Rows are full-width FP32, 64 lanes per head.
; rdi buffer, esi head count, rdx norm weights. Position in r14d.
head_norm_rope:
    mov r10d,esi
    mov r11,rdx
.head:
    mov rsi,r11
    mov rdx,rdi
    mov ecx,64
    call norm
    xor ecx,ecx
.pair:
    movss xmm2,[rope_cos+rcx*4]
    movss xmm3,[rope_sin+rcx*4]
    movss xmm0,[rdi+rcx*4]
    movss xmm1,[rdi+rcx*4+128]
    movaps xmm4,xmm0
    movaps xmm5,xmm1
    mulss xmm0,xmm2
    mulss xmm5,xmm3
    subss xmm0,xmm5
    mulss xmm1,xmm2
    mulss xmm4,xmm3
    addss xmm1,xmm4
    movss [rdi+rcx*4],xmm0
    movss [rdi+rcx*4+128],xmm1
    inc ecx
    cmp ecx,32
    jb .pair
    add rdi,256
    dec r10d
    jnz .head
    ret

; r12 points to interleaved K,V cache rows, each 4096 bytes.
attention:
    cmp byte [vector_math],0
    jne attention_vector
    xor r8d,r8d
.head:
    mov eax,r8d
    shr eax,1
    shl eax,8
    lea r9,[r12+rax]
    mov eax,r8d
    shl eax,8
    lea r10,[query+rax]
    lea r11,[mixed+rax]
    xor ecx,ecx
    movss xmm6,[negative_inf]
.score:
    xor eax,eax
    xorps xmm0,xmm0
.dot:
    movups xmm1,[r10+rax*4]
    movups xmm2,[r9+rax*4]
    mulps xmm1,xmm2
    addps xmm0,xmm1
    add eax,4
    cmp eax,64
    jb .dot
    movhlps xmm1,xmm0
    addps xmm0,xmm1
    movaps xmm1,xmm0
    shufps xmm1,xmm1,0x55
    addss xmm0,xmm1
    mulss xmm0,[inv8]
    movss [scores+rcx*4],xmm0
    maxss xmm6,xmm0
    add r9,4096
    inc ecx
    cmp ecx,r14d
    jbe .score
    xor ecx,ecx
    xorps xmm7,xmm7
.softmax:
    movss xmm0,[scores+rcx*4]
    subss xmm0,xmm6
    call exp_scalar
    movss [scores+rcx*4],xmm0
    addss xmm7,xmm0
    inc ecx
    cmp ecx,r14d
    jbe .softmax
    mov eax,r8d
    shr eax,1
    shl eax,8
    lea r9,[r12+rax+2048]
    xor edx,edx
    movaps xmm5,xmm7
    shufps xmm5,xmm5,0
.lane:
    mov rsi,r9
    xor ecx,ecx
    xorps xmm0,xmm0
.weighted:
    movss xmm1,[scores+rcx*4]
    shufps xmm1,xmm1,0
    movups xmm2,[rsi+rdx*4]
    mulps xmm1,xmm2
    addps xmm0,xmm1
    add rsi,4096
    inc ecx
    cmp ecx,r14d
    jbe .weighted
    divps xmm0,xmm5
    movups [r11+rdx*4],xmm0
    add edx,4
    cmp edx,64
    jb .lane
    inc r8d
    cmp r8d,16
    jb .head
    ret

; Heads are independent, so partitions compute whole heads with their own
; score rows and each head is bit-identical at every thread count. The scalar
; path stays on the main thread because exp_scalar shares its x87 scratch.
attention_vector:
    mov [job_kv],r12
    mov [job_position],r14d
    cmp dword [thread_count],1
    jne .parallel
    xor eax,eax
    jmp attention_job
.parallel:
    mov qword [job_entry],attention_job
    jmp dispatch_job

; eax = partition index; heads [index*16/threads, (index+1)*16/threads).
attention_job:
    push r12
    mov r12,[job_kv]
    mov r10d,[job_position]
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
    call attention_avx2
    pop r12
    ret

; AVX2/FMA attention over the same cache layout. The 64 query lanes stay in
; ymm8-ymm15 while scoring; one pass over positions then accumulates all value
; lanes. The final partial softmax vector is masked on load, store and sum.
; Heads r8d up to edi, position r10d, score row rsi, KV layer r12.
attention_avx2:
.head:
    mov eax,r8d
    shr eax,1
    shl eax,8
    lea r9,[r12+rax]
    mov eax,r8d
    shl eax,8
    lea r11,[mixed+rax]
    vmovups ymm8,[query+rax]
    vmovups ymm9,[query+rax+32]
    vmovups ymm10,[query+rax+64]
    vmovups ymm11,[query+rax+96]
    vmovups ymm12,[query+rax+128]
    vmovups ymm13,[query+rax+160]
    vmovups ymm14,[query+rax+192]
    vmovups ymm15,[query+rax+224]
    xor ecx,ecx
    vmovss xmm6,[negative_inf]
.score:
    vmulps ymm0,ymm8,[r9]
    vmulps ymm1,ymm9,[r9+32]
    vfmadd231ps ymm0,ymm10,[r9+64]
    vfmadd231ps ymm1,ymm11,[r9+96]
    vfmadd231ps ymm0,ymm12,[r9+128]
    vfmadd231ps ymm1,ymm13,[r9+160]
    vfmadd231ps ymm0,ymm14,[r9+192]
    vfmadd231ps ymm1,ymm15,[r9+224]
    vaddps ymm0,ymm0,ymm1
    vextractf128 xmm1,ymm0,1
    vaddps xmm0,xmm0,xmm1
    vmovhlps xmm1,xmm0,xmm0
    vaddps xmm0,xmm0,xmm1
    vmovshdup xmm1,xmm0
    vaddss xmm0,xmm0,xmm1
    vmulss xmm0,xmm0,[inv8]
    vmovss [rsi+rcx*4],xmm0
    vmaxss xmm6,xmm6,xmm0
    add r9,4096
    inc ecx
    cmp ecx,r10d
    jbe .score
    vbroadcastss ymm4,xmm6
    vxorps ymm5,ymm5,ymm5
    xor ecx,ecx
    lea eax,[r10+1]
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
    vextractf128 xmm1,ymm5,1
    vaddps xmm5,xmm5,xmm1
    vhaddps xmm5,xmm5,xmm5
    vhaddps xmm5,xmm5,xmm5
    vbroadcastss ymm7,xmm5
    mov eax,r8d
    shr eax,1
    shl eax,8
    lea r9,[r12+rax+2048]
    vxorps ymm8,ymm8,ymm8
    vxorps ymm9,ymm9,ymm9
    vxorps ymm10,ymm10,ymm10
    vxorps ymm11,ymm11,ymm11
    vxorps ymm12,ymm12,ymm12
    vxorps ymm13,ymm13,ymm13
    vxorps ymm14,ymm14,ymm14
    vxorps ymm15,ymm15,ymm15
    xor ecx,ecx
.weighted:
    vbroadcastss ymm0,[rsi+rcx*4]
    vfmadd231ps ymm8,ymm0,[r9]
    vfmadd231ps ymm9,ymm0,[r9+32]
    vfmadd231ps ymm10,ymm0,[r9+64]
    vfmadd231ps ymm11,ymm0,[r9+96]
    vfmadd231ps ymm12,ymm0,[r9+128]
    vfmadd231ps ymm13,ymm0,[r9+160]
    vfmadd231ps ymm14,ymm0,[r9+192]
    vfmadd231ps ymm15,ymm0,[r9+224]
    add r9,4096
    inc ecx
    cmp ecx,r10d
    jbe .weighted
    vdivps ymm8,ymm8,ymm7
    vdivps ymm9,ymm9,ymm7
    vdivps ymm10,ymm10,ymm7
    vdivps ymm11,ymm11,ymm7
    vdivps ymm12,ymm12,ymm7
    vdivps ymm13,ymm13,ymm7
    vdivps ymm14,ymm14,ymm7
    vdivps ymm15,ymm15,ymm7
    vmovups [r11],ymm8
    vmovups [r11+32],ymm9
    vmovups [r11+64],ymm10
    vmovups [r11+96],ymm11
    vmovups [r11+128],ymm12
    vmovups [r11+160],ymm13
    vmovups [r11+192],ymm14
    vmovups [r11+224],ymm15
    inc r8d
    cmp r8d,edi
    jb .head
    vzeroupper
    ret
