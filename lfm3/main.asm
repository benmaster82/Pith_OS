; LFM3-1.7B GGUF inference on x86-64 Linux. No libc. Shares the platform,
; thread, tokenizer and chat code of the LFM2.5 runtime.
bits 64
default abs
%include "layout.inc"
%include "tokenizer.inc"
%define TEXT_BYTES 8192
%define MAX_CONTEXT 32768
%define TOKEN_LIMIT 127999
%define DECODE_BYTES 512
%define KV_LAYERS ATT_LAYERS
%define CONV_STATE_BYTES CONV_LAYERS*6*8192
section .text
%include "platform_linux.asm"

; The GGUF header up to the tensor data: magic, metadata and tensor table.
validate_model:
    cmp dword [r15],0x46554747
    jne fail
    mov eax,2166136261
    xor ecx,ecx
.checksum:
    xor al,[r15+rcx]
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
    mov ecx,2048
    call norm_f32
    mov edi,normalized
    mov esi,activations
    mov ecx,2048
    call quantize_q8k
    mov edi,activations
    lea rsi,[r15+EMBED]
    mov edx,logits
    mov ecx,2048
    mov r8d,128000
    call matvec_q6k
%ifdef DUMP_LOGITS
    mov esi,logits
    mov edx,128000*4
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
    cmp ecx,128000
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
    cmp eax,EOS_ID              ; <|im_end|>, also emitted on stdout
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

; Consume one token: its Q6_K embedding row, then every layer.
consume_token:
    cmp r14d,[context_limit]
    jae fail
    imul esi,eax,8*210          ; eight Q6_K blocks per 2048-wide row
    lea rsi,[r15+rsi+EMBED]
    mov edi,hidden
    mov ecx,2048
    call dequant_q6k_row
%ifdef DUMP_LAYERS
    mov edi,1
    mov esi,hidden
    mov edx,2048*4
    call write_all
%endif
    call forward
    inc r14d
    ret

%include "util.asm"

%macro NORMALIZE 3
    mov edi,%1
    lea rsi,[r15+%2]
    mov edx,%3
    mov ecx,2048
    call norm_f32
%endmacro
%macro QUANTIZE 2
    mov edi,%1
    mov esi,activations
    mov ecx,%2
    call quantize_q8k
%endmacro
; Layer, tensor, output, width, rows; the kernel follows the tensor's GGUF type.
%macro PROJECT 5
    mov edi,activations
    lea rsi,[r15+L%1_%2]
    mov edx,%3
    mov ecx,%4
    mov r8d,%5
%if L%1_%2_TYPE == 12
    call matvec_q4k
%else
    call matvec_q6k
%endif
%endmacro
; Sandwich norm: the block output is normalized before the residual.
%macro CLOSE_BLOCK 1
    NORMALIZE update,%1,update
%ifdef DUMP_LAYERS
    mov edi,1
    mov esi,update
    mov edx,2048*4
    call write_all
%endif
    call residual
%endmacro
%macro CONV_LAYER 1
    NORMALIZE hidden,L%1_OP_NORM,normalized
    QUANTIZE normalized,2048
    PROJECT %1,IN,projection,2048,6144
    mov edi,projection
    lea rsi,[r15+L%1_CONV]
    mov edx,conv_cache+conv_index*6*8192
    call conv_step
    QUANTIZE projection,2048
    PROJECT %1,OUT,update,2048,2048
    CLOSE_BLOCK L%1_POST_OP
    %assign conv_index conv_index+1
%endmacro
; Layer index, sliding window flag.
%macro ATT_LAYER 2
    NORMALIZE hidden,L%1_OP_NORM,normalized
    QUANTIZE normalized,2048
    PROJECT %1,Q,query,2048,2048
    PROJECT %1,K,key,2048,512
    PROJECT %1,V,value,2048,512
    PROJECT %1,GATE,headgate,2048,16
    call sigmoid16
%if %2
    mov ecx,64
    mov r8d,rope_swa
%else
    mov ecx,32
    mov r8d,rope_full
%endif
    mov edi,query
    mov esi,16
    lea rdx,[r15+L%1_QN]
    call head_norm_rope
    mov edi,key
    mov esi,4
    lea rdx,[r15+L%1_KN]
    call head_norm_rope
    mov r12,[kv_stride]
    imul r12,att_index
    add r12,[kv_base]
    mov edi,r14d
    shl edi,12
    add rdi,r12
    mov esi,key                 ; value immediately follows key
    mov ecx,1024
    rep movsd
    xor r10d,r10d
%if %2
    lea eax,[r14-255]           ; the last 256 positions, the current included
    test eax,eax
    cmovg r10d,eax
%endif
    call attention
    QUANTIZE mixed,2048
    PROJECT %1,OUT,update,2048,2048
    CLOSE_BLOCK L%1_POST_OP
    %assign att_index att_index+1
%endmacro
%macro MLP 1
    NORMALIZE hidden,L%1_FF_NORM,normalized
    QUANTIZE normalized,2048
    PROJECT %1,W1,gate,2048,7680
    PROJECT %1,W3,up,2048,7680
    mov edi,gate
    mov esi,up
    mov edx,7680
    call silu_gate
    QUANTIZE gate,7680
    PROJECT %1,W2,update,7680,2048
    CLOSE_BLOCK L%1_POST_FF
%endmacro
forward:
    call prepare_rope
    %assign conv_index 0
    %assign att_index 0
    %include "lfm3-layers.inc"
    ret

%include "lfm3/kernels.asm"
%include "expmath.asm"
%include "threads.asm"
%include "tokenizer.asm"
%include "chat.asm"
section .rodata
epsilon: dd 1.0e-5
negative_inf: dd 0xff800000
one: dd 1.0
minus80: dd -80.0
plus80: dd 80.0
%include "expmath_data.inc"
%include "lfm3/frequencies.inc"
chat_option: db "--chat",0
generate_option: db "--generate",0
text_option: db "--text",0
encode_option: db "--encode",0
decode_option: db "--decode",0
; The official template with enable_thinking=false: an empty reasoning block.
chat_prefix: db "<|startoftext|><|im_start|>user",10,0
chat_user_prefix: db "<|im_start|>user",10,0
chat_suffix: db "<|im_end|>",10,"<|im_start|>assistant",10,"<think></think>",0
message: db "Usage: lfm3-next MODEL [--generate N] ID [ID ...]",10
    db "       lfm3-next MODEL --chat [N]",10
    db "       lfm3-next MODEL --text N 'prompt'",10
    db "       lfm3-next MODEL --encode 'text' | --decode ID [ID ...]",10
    db "MODEL: LFM3-1.7B GGUF with Q4_K and Q6_K tensors",10
    db "N >= 1; prompt length + N - 1 <= context; IDs 0..127999",10
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
alignb 32
rope_full: resd 128                 ; 64 cos, then 64 sin
rope_swa: resd 128
hidden: resd 2048
normalized: resd 2048
update: resd 2048
projection: resd 6144
gate: resd 7680
up: resd 7680
query: resd 2048
key: resd 512
value: resd 512
headgate: resd 16
mixed: resd 2048
scores: resd MAX_CONTEXT
logits: resd 128000
activations: resb 30*Q8K_BYTES      ; Q8_K rows up to 7680 wide
kv_base: resq 1
kv_stride: resq 1
model_base: resq 1
conv_cache: resb CONV_STATE_BYTES
scratch: resd 1
number: resb 8
section .data
matvec_impl: dq 0                   ; the LFM2 worker entry; LFM3 jobs come later
max_new_tokens: dd 1
context_limit: dd 512
section .note.GNU-stack noalloc noexec nowrite progbits
