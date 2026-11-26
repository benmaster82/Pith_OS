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
