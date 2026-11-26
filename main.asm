; Shared x86-64 inference. No libc. Greedy generation retains model caches.
bits 64
default abs
%include "layout.inc"
%include "tokenizer.inc"
%define TEXT_BYTES 8192
%define MAX_CONTEXT 32768
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

option_equal:
    xor ecx,ecx
.character:
    mov al,[rsi+rcx]
    cmp al,[rdi+rcx]
    jne .different
    inc ecx
    test al,al
    jnz .character
    mov eax,1
    ret
.different:
    xor eax,eax
    ret

print_id:
    mov esi,number+7
    mov byte [rsi],10
    mov ecx,10
.digit:
    xor edx,edx
    div ecx
    add dl,'0'
    dec rsi
    mov [rsi],dl
    test eax,eax
    jnz .digit
    mov edx,number+8
    sub rdx,rsi
    mov edi,1
    jmp write_all

; Parse a nonempty decimal string, bounded by ecx. Bounds are checked after
; every digit, before another multiplication can overflow for our limits.
parse_uint:
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
    cmp eax,ecx
    ja fail
    inc rsi
    cmp byte [rsi],0
    jne .digit
    ret

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
align 32
%macro LANES 2
%1: times 8 dd %2
%endmacro
LANES exp_min8,-80.0
LANES exp_max8,80.0
LANES log2e8,1.44269504088896341
LANES ln2_hi8,0.693359375
LANES ln2_lo8,-2.12194440e-4
LANES exp_p0,1.9875691500e-4
LANES exp_p1,1.3981999507e-3
LANES exp_p2,8.3334519073e-3
LANES exp_p3,4.1665795894e-2
LANES exp_p4,1.6666665459e-1
LANES exp_p5,5.0000001201e-1
LANES one8,1.0
LANES exp_bias8,127
lane_index8: dd 0,1,2,3,4,5,6,7
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
scores: resd MAX_CONTEXT
logits: resd 65536
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
