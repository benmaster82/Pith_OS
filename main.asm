; Shared x86-64 inference. No libc. Greedy generation retains model caches.
bits 64
default abs
%include "layout.inc"
%include "tokenizer.inc"
%define TEXT_BYTES 8192
%define MAX_CONTEXT 32768
%ifndef PREFILL_BLOCK
%define PREFILL_BLOCK 15            ; prompt tokens per weight pass
%endif
section .text
%ifdef BARE_METAL
%include "os/platform.asm"
%else
%include "platform_linux.asm"
%endif

validate_model:
    cmp qword [r15],HEADER_SIZE
    jne fail
    mov eax,2166136261
    xor ecx,ecx
.checksum:
    xor al,[r15+rcx+8]
    imul eax,16777619
    inc ecx
    cmp ecx,HEADER_SIZE
    jb .checksum
    cmp eax,HEADER_HASH
    jne fail
    ret

; rbp = new prompt length, r14 = absolute cache position. Leaves the final
; predicted token pending in next_token. Chat commits it before the next turn.
generate_prompt:
    mov dword [generated_tokens],0
    mov dword [decode_length],0
    mov byte [decode_final],0
    xor r13d,r13d
    cmp byte [vector_math],0
    je .prefill
.block:
    mov eax,ebp
    sub eax,r13d
    mov ecx,PREFILL_BLOCK
    cmp eax,ecx
    cmova eax,ecx
    mov [block_count],eax
    add eax,r14d
    cmp eax,[context_limit]
    ja fail
    call forward_block
    add r13d,[block_count]
    cmp r13,rbp
    jb .block
    jmp .logits
.prefill:
    mov eax,[prompt_ids+r13*4]
    call consume_token
    inc r13d
    cmp r13,rbp
    jb .prefill
.logits:
    mov edi,hidden
    lea rsi,[r15+FINAL]
    mov edx,normalized
    mov ecx,1024
    call norm
    mov edi,normalized
    lea rsi,[r15+EMBED]
    mov edx,logits
    mov ecx,1024
    mov r8d,65536
    call matvec
%ifdef DUMP_LOGITS
    mov esi,logits
    mov edx,65536*4
    mov edi,1
    call write_all
    ret
%endif
    xor eax,eax
    mov ecx,1
    movss xmm0,[logits]
.argmax:
    movss xmm1,[logits+rcx*4]
    ucomiss xmm1,xmm0
    jbe .next
    movaps xmm0,xmm1
    mov eax,ecx
.next:
    inc ecx
    cmp ecx,65536
    jb .argmax
    mov [next_token],eax
    cmp byte [text_mode],0
    jne .emit_text
    call print_id
    jmp .emitted
.emit_text:
    call decode_token
.emitted:
    inc dword [generated_tokens]
    mov eax,[generated_tokens]
    cmp eax,[max_new_tokens]
    jae .finished
    mov eax,[next_token]
    cmp eax,7                   ; official eos_token_id, also emitted on stdout
    je .finished
    call consume_token
    jmp .logits
.finished:
    cmp byte [text_mode],0
    je .done
    mov byte [decode_final],1
    call decode_pending
.done:
    ret

; Consume exactly one token, updating both convolution state and KV caches.
consume_token:
    cmp r14d,[context_limit]
    jae fail
    shl eax,11
    lea rsi,[r15+rax+EMBED]
    xor ecx,ecx
.embed:
    movzx eax,word [rsi+rcx*2]
    shl eax,16
    mov [hidden+rcx*4],eax
    inc ecx
    cmp ecx,1024
    jb .embed
    call forward
    inc r14d
    ret

%include "util.asm"

%macro PROJECT 5
    mov edi,%1
    lea rsi,[r15+%2]
    mov edx,%3
    mov ecx,%4
    mov r8d,%5
    call matvec
%endmacro
%macro NORMALIZE 1
    mov edi,hidden
    lea rsi,[r15+%1]
    mov edx,normalized
    mov ecx,1024
    call norm
%endmacro
%macro CONV_LAYER 4
    NORMALIZE %1
    PROJECT normalized,%2,projection,1024,3072
    mov edi,projection
    lea rsi,[r15+%3]
    mov edx,conv_cache+conv_index*8192
    call conv_step
    PROJECT projection,%4,update,1024,1024
    %assign conv_index conv_index+1
%endmacro
%macro ATT_LAYER 7
    NORMALIZE %1
    PROJECT normalized,%2,query,1024,1024
    PROJECT normalized,%3,key,1024,512
    PROJECT normalized,%4,value,1024,512
    mov edi,query
    mov esi,16
    lea rdx,[r15+%5]
    call head_norm_rope
    mov edi,key
    mov esi,8
    lea rdx,[r15+%6]
    call head_norm_rope
    mov r12,[kv_stride]
    imul r12,att_index
    add r12,[kv_base]
    mov edi,r14d
    shl edi,12
    add rdi,r12
    mov esi,key                    ; value immediately follows key
    mov ecx,1024
    rep movsd
    call attention
    PROJECT mixed,%7,update,1024,1024
    %assign att_index att_index+1
%endmacro
%macro MLP 4
    call residual
    NORMALIZE %1
    PROJECT normalized,%2,gate,1024,4608
    PROJECT normalized,%3,up,1024,4608
    mov edi,gate
    mov esi,up
    mov edx,4608
    call silu_gate
    PROJECT gate,%4,update,4608,1024
    call residual
%endmacro
forward:
    call prepare_rope
    %assign conv_index 0
    %assign att_index 0
    %include "layers.inc"
    ret
residual:
    xor ecx,ecx
.loop:
    movss xmm0,[update+rcx*4]
    addss xmm0,[hidden+rcx*4]
    movss [hidden+rcx*4],xmm0
    inc ecx
    cmp ecx,1024
    jb .loop
    ret

; Prefill blocks: the same layers over [block_count] tokens. Matrices read
; each weight row once per block; normalization, convolution state, RoPE and
; attention advance token by token in order, so every value matches forward.
; ebx indexes the token inside a block.
%macro BPROJECT 6
    mov edi,%1
    lea rsi,[r15+%2]
    mov edx,%3
    mov ecx,%4
    mov r8d,%5
    mov r9d,%6
    mov r10d,[block_count]
    call matvec_batch
%endmacro
%macro BNORMALIZE 1
    xor ebx,ebx
%%token:
    imul edi,ebx,4096
    add edi,block_hidden
    lea rsi,[r15+%1]
    imul edx,ebx,4096
    add edx,block_normalized
    mov ecx,1024
    call norm
    inc ebx
    cmp ebx,[block_count]
    jb %%token
%endmacro
%macro BCONV_LAYER 4
    BNORMALIZE %1
    BPROJECT block_normalized,%2,block_projection,1024,3072,4096
    xor ebx,ebx
%%convolve:
    imul edi,ebx,3072*4
    add edi,block_projection
    lea rsi,[r15+%3]
    mov edx,conv_cache+conv_index*8192
    call conv_step
    inc ebx
    cmp ebx,[block_count]
    jb %%convolve
    BPROJECT block_projection,%4,block_update,1024,1024,3072*4
    %assign conv_index conv_index+1
%endmacro
%macro BATT_LAYER 7
    BNORMALIZE %1
    BPROJECT block_normalized,%2,block_query,1024,1024,4096
    BPROJECT block_normalized,%3,block_key,1024,512,4096
    BPROJECT block_normalized,%4,block_value,1024,512,4096
    mov r12,[kv_stride]
    imul r12,att_index
    add r12,[kv_base]
    xor ebx,ebx
%%rotate:
    mov r14d,[block_base]
    add r14d,ebx
    call prepare_rope
    imul edi,ebx,4096
    add edi,block_query
    mov esi,16
    lea rdx,[r15+%5]
    call head_norm_rope
    imul edi,ebx,2048
    add edi,block_key
    mov esi,8
    lea rdx,[r15+%6]
    call head_norm_rope
    mov edi,r14d
    shl edi,12
    add rdi,r12
    imul esi,ebx,2048
    add esi,block_key
    mov ecx,512
    rep movsd
    imul esi,ebx,2048
    add esi,block_value
    mov ecx,512                    ; value follows key in each cache row
    rep movsd
    inc ebx
    cmp ebx,[block_count]
    jb %%rotate
    ; Every key and value of the block is cached; each token reads only
    ; positions up to its own.
    xor ebx,ebx
%%attend:
    mov r14d,[block_base]
    add r14d,ebx
    imul esi,ebx,4096
    add esi,block_query
    mov edi,query
    mov ecx,1024
    rep movsd
    call attention
    mov esi,mixed
    imul edi,ebx,4096
    add edi,block_mixed
    mov ecx,1024
    rep movsd
    inc ebx
    cmp ebx,[block_count]
    jb %%attend
    mov r14d,[block_base]
    BPROJECT block_mixed,%7,block_update,1024,1024,4096
    %assign att_index att_index+1
%endmacro
%macro BMLP 4
    call block_residual
    BNORMALIZE %1
    BPROJECT block_normalized,%2,block_gate,1024,4608,4096
    BPROJECT block_normalized,%3,block_up,1024,4608,4096
    mov edi,block_gate
    mov esi,block_up
    imul edx,[block_count],4608
    call silu_gate
    BPROJECT block_gate,%4,block_update,4608,1024,4608*4
    call block_residual
%endmacro

; Consume prompt_ids[r13..r13+block_count) at positions r14 onward. The last
; hidden state is copied to hidden for the logits. Advances r14.
forward_block:
    push rbx
    mov [block_base],r14d
    xor ebx,ebx
.embed:
    lea eax,[r13+rbx]
    mov eax,[prompt_ids+rax*4]
    shl eax,11
    lea rsi,[r15+rax+EMBED]
    imul edi,ebx,4096
    add edi,block_hidden
    xor ecx,ecx
.lane:
    movzx eax,word [rsi+rcx*2]
    shl eax,16
    mov [rdi+rcx*4],eax
    inc ecx
    cmp ecx,1024
    jb .lane
    inc ebx
    cmp ebx,[block_count]
    jb .embed
    %assign conv_index 0
    %assign att_index 0
    %define CONV_LAYER BCONV_LAYER
    %define ATT_LAYER BATT_LAYER
    %define MLP BMLP
    %include "layers.inc"
    %undef CONV_LAYER
    %undef ATT_LAYER
    %undef MLP
    mov esi,[block_count]
    dec esi
    shl esi,12
    add esi,block_hidden
    mov edi,hidden
    mov ecx,1024
    rep movsd
    mov r14d,[block_base]
    add r14d,[block_count]
    pop rbx
    ret

block_residual:
    imul edx,[block_count],1024
    xor ecx,ecx
.loop:
    movss xmm0,[block_update+rcx*4]
    addss xmm0,[block_hidden+rcx*4]
    movss [block_hidden+rcx*4],xmm0
    inc ecx
    cmp ecx,edx
    jb .loop
    ret
%include "kernels.asm"
%include "attention.asm"
%ifndef BARE_METAL
%include "threads.asm"
%else
%include "os/smp.asm"
%endif
%include "tokenizer.asm"
%include "chat.asm"
section .rodata
epsilon: dd 1.0e-5
negative_inf: dd 0xff800000
inv8: dd 0.125
one: dd 1.0
minus80: dd -80.0
plus80: dd 80.0
%include "expmath_data.inc"
freq:
%include "frequencies.inc"
chat_option: db "--chat",0
generate_option: db "--generate",0
text_option: db "--text",0
encode_option: db "--encode",0
decode_option: db "--decode",0
chat_prefix: db "<|startoftext|><|im_start|>user",10,0
chat_user_prefix: db "<|im_start|>user",10,0
chat_suffix: db "<|im_end|>",10,"<|im_start|>assistant",10,0
message: db "Usage: lfm-next MODEL [--generate N] ID [ID ...]",10
    db "       lfm-next MODEL --chat [N]",10
    db "       lfm-next MODEL --text N 'prompt'",10
    db "       lfm-next MODEL --encode 'text' | --decode ID [ID ...]",10
    db "MODEL: official BF16 .safetensors checkpoint",10
    db "N >= 1; prompt length + N - 1 <= context; IDs 0..65535",10
    db "LFM_THREADS=1..16; LFM_CONTEXT=1..32768 (default 512, chat 2048)",10
message_len equ $-message
output_error: db "Failed to write generated output",10
output_error_len equ $-output_error
section .bss
text_mode: resb 1
vector_math: resb 1
prompt_ids: resd MAX_CONTEXT
next_token: resd 1
generated_tokens: resd 1
rope_cos: resd 32
rope_sin: resd 32
hidden: resd 1024
normalized: resd 1024
update: resd 1024
projection: resd 3072
gate: resd 4608
up: resd 4608
query: resd 1024
key: resd 512
value: resd 512
mixed: resd 1024
scores: resd MAX_CONTEXT*MAX_THREADS  ; one row per attention partition
logits: resd 65536
block_count: resd 1
block_base: resd 1
alignb 32
block_hidden: resd PREFILL_BLOCK*1024
block_normalized: resd PREFILL_BLOCK*1024
block_update: resd PREFILL_BLOCK*1024
block_projection: resd PREFILL_BLOCK*3072
block_gate: resd PREFILL_BLOCK*4608
block_up: resd PREFILL_BLOCK*4608
block_query: resd PREFILL_BLOCK*1024
block_key: resd PREFILL_BLOCK*512
block_value: resd PREFILL_BLOCK*512
block_mixed: resd PREFILL_BLOCK*1024
kv_base: resq 1
kv_stride: resq 1
model_base: resq 1
conv_cache: resb 10*8192
scratch: resd 1
number: resb 8
section .data
matvec_impl: dq matvec_scalar
max_new_tokens: dd 1
context_limit: dd 512
section .note.GNU-stack noalloc noexec nowrite progbits
