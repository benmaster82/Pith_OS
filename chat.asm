; Context allocation and a line-oriented chat REPL. Linux separates status on
; stderr from responses on stdout; bare metal uses COM1. Validate before caching.
section .text
%ifndef EOS_ID
%define EOS_ID 7                    ; official eos_token_id
%define NEWLINE_ID 708              ; newline token after im_end
%define KV_LAYERS 6                 ; attention layers with 4096-byte cache rows
%define CONV_STATE_BYTES 10*8192
%endif
init_context:
    push r12
    mov r12,rdi
.environment:
    mov rsi,[r12]
    test rsi,rsi
    jz .done
    xor ecx,ecx
.name:
    mov al,[context_env+rcx]
    cmp al,[rsi+rcx]
    jne .next
    inc ecx
    cmp ecx,12
    jb .name
    add rsi,12
    mov ecx,MAX_CONTEXT
    call parse_uint
    test eax,eax
    jz fail
    mov [context_limit],eax
    mov byte [context_explicit],1
    jmp .done
.next:
    add r12,8
    jmp .environment
.done:
    pop r12
    ret

allocate_cache:
    mov esi,[context_limit]
    shl rsi,12
    mov [kv_stride],rsi
    imul rsi,KV_LAYERS
    call allocate_pages
    cmp rax,-4095
    jae .error
    mov [kv_base],rax
    ret
.error:
    mov esi,cache_error
    mov edx,cache_error_len
    call chat_write
    mov edi,1
    jmp exit

chat_write:
    mov edi,2
    jmp write_all

chat_loop:
    mov [chat_recovery_rsp],rsp
    mov esi,chat_banner
    mov edx,chat_banner_len
    call chat_write
%ifdef ENGINE_HOOKS
    call engine_ready
%endif
chat_next_line:
    mov esi,user_label
    mov edx,user_label_len
    call chat_write
    call chat_readline
    cmp eax,-1
    je chat_done
    cmp eax,-3
    je chat_io_error
    cmp eax,-2
    je chat_bad_input
    test eax,eax
    jz chat_next_line
    mov esi,chat_line
    mov edi,quit_command
    call option_equal
    test eax,eax
    jnz chat_done
%ifdef ENGINE_HOOKS
    mov esi,chat_line
    mov edi,exit_command
    call option_equal
    test eax,eax
    jnz chat_done
%endif
    mov esi,chat_line
    mov edi,clear_command
    call option_equal
    test eax,eax
    jnz chat_clear
    cmp byte [chat_line],'/'
%ifdef ENGINE_HOOKS
    jne .message
    call extra_command          ; 1 handled, 2 handled with a fresh conversation
    cmp eax,1
    je chat_next_line
    ja chat_clear
    jmp chat_unknown_command
.message:
%else
    je chat_unknown_command
%endif

    ; All tokenizer failures in this stage can safely recover before inference.
    mov byte [chat_tokenizing],1
    mov edi,text_buffer
%ifdef ENGINE_HOOKS
    call engine_prefix
    mov esi,chat_line
    call copy_text
    call engine_suffix
%else
    mov esi,chat_user_prefix
    cmp dword [cache_position],0
    jne .prefix
    mov esi,chat_prefix
.prefix:
    call copy_text
    mov esi,chat_line
    call copy_text
    mov esi,chat_suffix
    call copy_text
%endif
    mov esi,text_buffer
    call encode_text
    mov byte [chat_tokenizing],0
    mov ebp,[token_count]
    mov eax,[cache_position]
    add eax,ebp
    add eax,[max_new_tokens]
    add eax,2                  ; reserve synthetic EOS and newline if truncated
    cmp eax,[context_limit]
    ja chat_context_full

    mov esi,assistant_label
    mov edx,assistant_label_len
    call chat_write
    mov r14d,[cache_position]
    call generate_prompt
    ; The generation loop leaves its final prediction unconsumed.
    mov eax,[next_token]
    call consume_token
    cmp dword [next_token],EOS_ID
    je .newline_token
    mov eax,EOS_ID
    call consume_token
.newline_token:
    mov eax,NEWLINE_ID          ; newline after assistant's im_end
    call consume_token
    mov [cache_position],r14d
    mov edi,1
    mov esi,chat_newline
    mov edx,1
    call write_all
    cmp dword [next_token],EOS_ID
    je chat_next_line
    mov esi,limit_notice
    mov edx,limit_notice_len
    call chat_write
    jmp chat_next_line

chat_clear:
    mov dword [cache_position],0
    mov dword [decode_length],0
    mov edi,conv_cache
    xor eax,eax
    mov ecx,CONV_STATE_BYTES/8
    rep stosq
    ; KV entries are overwritten before each read after position resets.
    mov esi,clear_notice
    mov edx,clear_notice_len
    call chat_write
    jmp chat_next_line

chat_context_full:
    mov esi,context_notice
    mov edx,context_notice_len
    call chat_write
    jmp chat_next_line

chat_unknown_command:
    mov esi,command_notice
    mov edx,command_notice_len
    call chat_write
    jmp chat_next_line

chat_tokenizer_recover:
    ; Unwind only pre-inference tokenizer calls. Cached model state is untouched.
    mov rsp,[chat_recovery_rsp]
    mov r15,[model_base]
    mov byte [chat_tokenizing],0
chat_bad_input:
    mov esi,input_notice
    mov edx,input_notice_len
    call chat_write
    jmp chat_next_line

chat_io_error:
    mov esi,input_error
    mov edx,input_error_len
    call chat_write
    mov edi,1
    jmp exit

chat_done:
    ret

; Returns length, -1 for EOF, -2 for rejected input, -3 for an I/O error.
; Overlong lines are drained so their remainder cannot become a second request.
chat_readline:
    push r12
    push r13
    xor r12d,r12d
    xor r13d,r13d
.read:
    xor edi,edi
    mov esi,input_byte
    mov edx,1
    call platform_read
    cmp rax,-4
    je .read
    test rax,rax
    js .error
    jz .eof
    mov al,[input_byte]
%ifdef BARE_METAL
    cmp al,8
    je .backspace
    cmp al,127
    je .backspace
%endif
    cmp al,10
    je .line
%ifdef PITHOS_UI
    cmp al,27
    je .clear_line              ; Escape clears the line being typed
    cmp al,9
    je .store
    cmp al,32
    jb .read                    ; other control characters are ignored
.store:
%endif
    test al,al
    jz .reject
    cmp r12d,TEXT_BYTES-1
    jae .reject
    mov [chat_line+r12],al
    inc r12d
    jmp .read
%ifdef BARE_METAL
.backspace:
    test r13d,r13d
    jnz .read
    test r12d,r12d
    jz .read
.erase_codepoint:
    dec r12d
    mov al,[chat_line+r12]
    and al,0xc0
    cmp al,0x80
    jne .erase_display
    test r12d,r12d
    jnz .erase_codepoint
.erase_display:
    mov esi,serial_erase
    mov edx,3
    call chat_write
    jmp .read
%endif
%ifdef PITHOS_UI
.clear_line:
    xor r13d,r13d
.clear_next:
    test r12d,r12d
    jz .read
.clear_codepoint:
    dec r12d
    mov al,[chat_line+r12]
    and al,0xc0
    cmp al,0x80
    jne .clear_display
    test r12d,r12d
    jnz .clear_codepoint
.clear_display:
    mov esi,serial_erase
    mov edx,3
    call chat_write
    jmp .clear_next
%endif
.reject:
    mov r13d,1
    jmp .read
.eof:
    test r12d,r12d
    jnz .line
    test r13d,r13d
    jnz .line
    mov eax,-1
    jmp .done
.line:
    test r13d,r13d
    jnz .invalid
    test r12d,r12d
    jz .terminate
    cmp byte [chat_line+r12-1],13
    jne .terminate
    dec r12d
.terminate:
    mov byte [chat_line+r12],0
    mov eax,r12d
    jmp .done
.invalid:
    mov eax,-2
    jmp .done
.error:
    mov eax,-3
.done:
    pop r13
    pop r12
    ret

section .rodata
%ifndef CHAT_USER_LABEL
%define CHAT_USER_LABEL "user> "
%define CHAT_ASSISTANT_LABEL "assistant> "
%endif
%ifndef CHAT_BANNER
%define CHAT_BANNER "Chat ready. /clear resets the conversation. /quit exits."
%endif
; Status notices may carry a dim colour for the Pithos CORE console.
%ifdef CHAT_STYLE
%define NOTICE_ON 27,"[2m",
%define NOTICE_OFF ,27,"[0m"
%else
%define NOTICE_ON
%define NOTICE_OFF
%endif
context_env: db "LFM_CONTEXT="
quit_command: db "/quit",0
%ifdef ENGINE_HOOKS
exit_command: db "/exit",0
%endif
clear_command: db "/clear",0
chat_banner: db CHAT_BANNER,10
chat_banner_len equ $-chat_banner
user_label: db CHAT_USER_LABEL
user_label_len equ $-user_label
assistant_label: db CHAT_ASSISTANT_LABEL
assistant_label_len equ $-assistant_label
chat_newline: db 10
clear_notice: db NOTICE_ON "Conversation cleared.",10 NOTICE_OFF
clear_notice_len equ $-clear_notice
%ifdef BARE_METAL
serial_erase: db 8,' ',8
context_notice: db NOTICE_ON "Context full. Use /clear or reboot with a larger context= value.",10 NOTICE_OFF
%else
context_notice: db NOTICE_ON "Context full. Use /clear or restart with a larger LFM_CONTEXT.",10 NOTICE_OFF
%endif
context_notice_len equ $-context_notice
command_notice: db NOTICE_ON "Unknown command. Use /clear or /quit.",10 NOTICE_OFF
command_notice_len equ $-command_notice
input_notice: db NOTICE_ON "Input rejected. Check UTF-8, byte and context limits, or use /clear.",10 NOTICE_OFF
input_notice_len equ $-input_notice
limit_notice: db NOTICE_ON "Response reached the token limit. The assistant turn was closed.",10 NOTICE_OFF
limit_notice_len equ $-limit_notice
input_error: db "Failed to read chat input.",10
input_error_len equ $-input_error
cache_error: db "Failed to allocate the KV cache.",10
cache_error_len equ $-cache_error

section .bss
context_explicit: resb 1
chat_mode: resb 1
chat_tokenizing: resb 1
chat_recovery_rsp: resq 1
cache_position: resd 1
chat_line: resb TEXT_BYTES
input_byte: resb 1

section .text
