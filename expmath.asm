; Bounded exponentials and the SiLU gate, shared by the model backends.

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

; gate = gate * sigmoid(gate) * up. rdi = gate, rsi = up, edx = lanes, a
; multiple of eight.
silu_gate:
    xor ecx,ecx
    cmp byte [vector_math],0
    je .scalar
.vector:
    vmovups ymm4,[rdi+rcx*4]
    vxorps ymm0,ymm0,ymm0
    vsubps ymm0,ymm0,ymm4
    call exp_avx2
    vaddps ymm0,ymm0,[one8]
    vdivps ymm4,ymm4,ymm0
    vmulps ymm4,ymm4,[rsi+rcx*4]
    vmovups [rdi+rcx*4],ymm4
    add ecx,8
    cmp ecx,edx
    jb .vector
    vzeroupper
    ret
.scalar:
    movss xmm1,[rdi+rcx*4]
    xorps xmm0,xmm0
    subss xmm0,xmm1
    call exp_scalar
    addss xmm0,[one]
    divss xmm1,xmm0
    mulss xmm1,[rsi+rcx*4]
    movss [rdi+rcx*4],xmm1
    inc ecx
    cmp ecx,edx
    jb .scalar
    ret
