; Context allocation and a line-oriented chat REPL. Linux separates status on
; stderr from responses on stdout; bare metal uses COM1. Validate before caching.
section .text
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
    imul rsi,6
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
    mov esi,chat_line
    mov edi,clear_command
    call option_equal
    test eax,eax
    jnz chat_clear
    cmp byte [chat_line],'/'
    je chat_unknown_command

    ; All tokenizer failures in this stage can safely recover before inference.
    mov byte [chat_tokenizing],1
    mov edi,text_buffer
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
    cmp dword [next_token],7
    je .newline_token
    mov eax,7
    call consume_token
.newline_token:
    mov eax,708                 ; newline after assistant's im_end
    call consume_token
    mov [cache_position],r14d
    mov edi,1
    mov esi,chat_newline
    mov edx,1
    call write_all
    cmp dword [next_token],7
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
    mov ecx,10*8192/8
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
context_env: db "LFM_CONTEXT="
quit_command: db "/quit",0
clear_command: db "/clear",0
chat_banner: db "Chat ready. /clear resets the conversation. /quit exits.",10
chat_banner_len equ $-chat_banner
user_label: db "user> "
user_label_len equ $-user_label
assistant_label: db "assistant> "
assistant_label_len equ $-assistant_label
chat_newline: db 10
clear_notice: db "Conversation cleared.",10
clear_notice_len equ $-clear_notice
%ifdef BARE_METAL
serial_erase: db 8,' ',8
context_notice: db "Context full. Use /clear or reboot with a larger context= value.",10
%else
context_notice: db "Context full. Use /clear or restart with a larger LFM_CONTEXT.",10
%endif
context_notice_len equ $-context_notice
command_notice: db "Unknown command. Use /clear or /quit.",10
command_notice_len equ $-command_notice
input_notice: db "Input rejected. Check UTF-8, byte and context limits, or use /clear.",10
input_notice_len equ $-input_notice
limit_notice: db "Response reached the token limit. The assistant turn was closed.",10
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
