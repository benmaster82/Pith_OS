; Linux entry point and platform services.
section .text
global _start
_start:
    call init_kernels
    mov rax,[rsp]
    lea rdi,[rsp+rax*8+16]
    call init_context
    mov rbp,[rsp]
    sub rbp,2
    jle fail
    lea rbx,[rsp+24]
    mov rsi,[rbx]
    cmp byte [rsi],'-'
    jne .validate_prompt
%ifdef DUMP_LOGITS
    jmp fail                    ; diagnostic binary emits one vector only
%endif
    mov edi,chat_option
    call option_equal
    test eax,eax
    jnz .chat_start
    mov edi,text_option
    call option_equal
    test eax,eax
    jnz .text_prompt
    mov edi,encode_option
    call option_equal
    test eax,eax
    jnz .encode_only
    mov edi,decode_option
    call option_equal
    test eax,eax
    jnz .decode_only
    xor ecx,ecx
.option:
    mov al,[generate_option+rcx]
    cmp al,[rsi+rcx]
    jne fail
    inc ecx
    test al,al
    jnz .option
    cmp rbp,3                   ; option, count, and at least one prompt ID
    jb fail
    mov rsi,[rbx+8]
    mov ecx,[context_limit]
    call parse_uint
    test eax,eax
    jz fail
    mov [max_new_tokens],eax
    add rbx,16
    sub rbp,2
    jmp .validate_prompt
.chat_start:
    cmp rbp,2
    ja fail
    cmp byte [context_explicit],0
    jne .chat_context
    mov dword [context_limit],2048
.chat_context:
    mov eax,[context_limit]
    mov edx,128
    cmp eax,edx
    cmova eax,edx
    mov [max_new_tokens],eax
    cmp rbp,2
    jne .chat_ready
    mov rsi,[rbx+8]
    mov ecx,[context_limit]
    call parse_uint
    test eax,eax
    jz fail
    mov [max_new_tokens],eax
.chat_ready:
    call load_tokenizer
    mov byte [text_mode],1
    mov byte [chat_mode],1
    jmp .load_model
.text_prompt:
    cmp rbp,3
    jne fail
    mov rsi,[rbx+8]
    mov ecx,[context_limit]
    call parse_uint
    test eax,eax
    jz fail
    mov [max_new_tokens],eax
    call load_tokenizer
    mov edi,text_buffer
    mov esi,chat_prefix
    call copy_text
    mov rsi,[rbx+16]
    call copy_text
    mov esi,chat_suffix
    call copy_text
    mov esi,text_buffer
    call encode_text
    mov ebp,[token_count]
    mov byte [text_mode],1
    jmp .validate_prompt
.encode_only:
    cmp rbp,2
    jne fail
    call load_tokenizer
    mov edi,text_buffer
    mov rsi,[rbx+8]
    call copy_text
    mov esi,text_buffer
    call encode_text
    xor r13d,r13d
.encode_print:
    cmp r13d,[token_count]
    jae .finished
    mov eax,[prompt_ids+r13*4]
    call print_id
    inc r13d
    jmp .encode_print
.decode_only:
    cmp rbp,2
    jb fail
    mov eax,[context_limit]
    inc eax
    cmp rbp,rax
    ja fail
    call load_tokenizer
    mov r13d,1
.decode_validate:
    mov rsi,[rbx+r13*8]
    mov ecx,65535
    call parse_uint
    mov [prompt_ids+r13*4-4],eax
    inc r13d
    cmp r13,rbp
    jb .decode_validate
    dec rbp
    xor r13d,r13d
.decode_print:
    mov eax,[prompt_ids+r13*4]
    call decode_token
    inc r13d
    cmp r13,rbp
    jb .decode_print
    mov byte [text_mode],1
    jmp .finished
.validate_prompt:
    mov eax,[max_new_tokens]
    lea rax,[rbp+rax-1]         ; final prediction needs no extra cache slot
    cmp eax,[context_limit]
    ja fail
    cmp byte [text_mode],0
    jne .load_model
    xor r13d,r13d
.parse_prompt:
    mov rsi,[rbx+r13*8]
    mov ecx,65535
    call parse_uint
    mov [prompt_ids+r13*4],eax
    inc r13d
    cmp r13,rbp
    jb .parse_prompt
.load_model:
    mov eax,2                      ; open read-only
    mov rdi,[rsp+16]
    xor esi,esi
    syscall
    test rax,rax
    js fail
    mov r8,rax
    mov rdi,rax
    mov eax,8                      ; lseek EOF
    xor esi,esi
    mov edx,2
    syscall
    cmp rax,FILE_SIZE
    jne fail
    mov esi,FILE_SIZE
    mov eax,9                      ; mmap
    xor edi,edi
    mov edx,1
    mov r10d,2
    xor r9d,r9d
    syscall
    cmp rax,-4095
    jae fail
    mov r15,rax
    mov [model_base],rax
    call validate_model
    mov rax,[rsp]
    lea rdi,[rsp+rax*8+16]       ; envp follows argv and its null terminator
    call init_threads
    call allocate_cache
    cmp byte [chat_mode],0
    jne .interactive
    xor r14d,r14d
    call generate_prompt
    xor edi,edi
    jmp exit
.interactive:
    call chat_loop
    xor edi,edi
    jmp exit
.finished:
    cmp byte [text_mode],0
    je .exit
    mov byte [decode_final],1
    call decode_pending
.exit:
    xor edi,edi
    jmp exit

write_all:
    mov eax,1
    syscall
    cmp rax,-4
    je write_all
    test rax,rax
    jle .error
    add rsi,rax
    sub rdx,rax
    jnz write_all
    ret
.error:
    mov eax,1
    mov edi,2
    mov esi,output_error
    mov edx,output_error_len
    syscall
    mov edi,1
    jmp exit
fail:
    mov eax,1
    mov edi,2
    mov esi,message
    mov edx,message_len
    syscall
    mov edi,1
exit:
    mov eax,231                 ; exit_group also releases persistent workers
    syscall

platform_read:
    xor eax,eax
    syscall
    ret

allocate_pages:
    mov eax,9
    xor edi,edi
    mov edx,3
    mov r10d,0x22
    mov r8,-1
    xor r9d,r9d
    syscall
    ret

load_tokenizer:
    push r12
    mov eax,89
    mov edi,self_exe
    mov esi,tokenizer_path
    mov edx,4095
    syscall
    test rax,rax
    jle tokenizer_fail
    cmp eax,4095
    jae tokenizer_fail
.find_slash:
    dec eax
    js tokenizer_fail
    cmp byte [tokenizer_path+rax],'/'
    jne .find_slash
    inc eax
    cmp eax,4082
    ja tokenizer_fail
    lea rdi,[tokenizer_path+rax]
    mov esi,tokenizer_name
    mov ecx,14
    rep movsb
    mov eax,2
    mov edi,tokenizer_path
    xor esi,esi
    syscall
    test rax,rax
    js tokenizer_fail
    mov r12,rax
    mov rdi,rax
    mov eax,8
    xor esi,esi
    mov edx,2
    syscall
    cmp rax,TOK_SIZE
    jne tokenizer_fail
    mov eax,9
    xor edi,edi
    mov esi,TOK_SIZE
    mov edx,1
    mov r10d,2
    mov r8,r12
    xor r9d,r9d
    syscall
    cmp rax,-4095
    jae tokenizer_fail
    mov [tokenizer_base],rax
    mov eax,3
    mov rdi,r12
    syscall
    pop r12
    jmp validate_tokenizer
