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

; exp(xmm0), bounded for the softmax and SiLU paths.
exp_scalar:
    maxss xmm0,[minus80]
    minss xmm0,[plus80]
    movss [scratch],xmm0
    fld dword [scratch]
    fldl2e
    fmulp
    fld st0
    frndint
    fxch st1
    fsub st0,st1
    f2xm1
    fld1
    faddp
    fscale
    fstp st1
    fstp dword [scratch]
    movss xmm0,[scratch]
    ret

; exp(ymm0) for eight lanes, with the same bounds as exp_scalar. Cephes
; range reduction by round(x/ln2) and a degree-5 polynomial; 2^n is built in
; the exponent field, valid because |n| <= 116. Clobbers ymm1-ymm3 only.
exp_avx2:
    vmaxps ymm0,ymm0,[exp_min8]
    vminps ymm0,ymm0,[exp_max8]
    vmulps ymm1,ymm0,[log2e8]
    vroundps ymm1,ymm1,0
    vfnmadd231ps ymm0,ymm1,[ln2_hi8]
    vfnmadd231ps ymm0,ymm1,[ln2_lo8]
    vmovups ymm2,[exp_p0]
    vfmadd213ps ymm2,ymm0,[exp_p1]
    vfmadd213ps ymm2,ymm0,[exp_p2]
    vfmadd213ps ymm2,ymm0,[exp_p3]
    vfmadd213ps ymm2,ymm0,[exp_p4]
    vfmadd213ps ymm2,ymm0,[exp_p5]
    vmulps ymm3,ymm0,ymm0
    vfmadd213ps ymm2,ymm3,ymm0
    vaddps ymm2,ymm2,[one8]
    vcvtps2dq ymm1,ymm1
    vpaddd ymm1,ymm1,[exp_bias8]
    vpslld ymm1,ymm1,23
    vmulps ymm0,ymm2,ymm1
    ret

; gate = gate * sigmoid(gate) * up over the 4608 MLP lanes.
silu_gate:
    xor ecx,ecx
    cmp byte [vector_math],0
    je .scalar
.vector:
    vmovups ymm4,[gate+rcx*4]
    vxorps ymm0,ymm0,ymm0
    vsubps ymm0,ymm0,ymm4
    call exp_avx2
    vaddps ymm0,ymm0,[one8]
    vdivps ymm4,ymm4,ymm0
    vmulps ymm4,ymm4,[up+rcx*4]
    vmovups [gate+rcx*4],ymm4
    add ecx,8
    cmp ecx,4608
    jb .vector
    vzeroupper
    ret
.scalar:
    movss xmm1,[gate+rcx*4]
    xorps xmm0,xmm0
    subss xmm0,xmm1
    call exp_scalar
    addss xmm0,[one]
    divss xmm1,xmm0
    mulss xmm1,[up+rcx*4]
    movss [gate+rcx*4],xmm1
    inc ecx
    cmp ecx,4608
    jb .scalar
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
