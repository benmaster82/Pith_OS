; Bare-metal backend. No Linux ABI and no guest filesystem.
%ifndef OS_BANNER
%define OS_BANNER "LFM bare-metal x86-64 | BF16"
%endif
%include "boot_bundle.inc"
extern boot_info, __kernel_end
global os_entry, write_all, exit
section .text
os_entry:
    call serial_init
    call video_init
    call keyboard_init
    mov esi,os_banner
    call os_puts
    call init_kernels
    call os_load_module
    call validate_model
    call load_tokenizer
    call os_options
%ifdef DUMP_LOGITS
    cmp byte [os_mode],2
    je fail                    ; diagnostic kernel emits one vector, never a chat
%endif
    call os_find_arena
    call allocate_cache
    call smp_init
    cmp qword [matvec_impl],matvec_avx2
    jne .scalar
    mov esi,os_avx
    jmp .cpu
.scalar:
    mov esi,os_scalar
.cpu:
    call os_puts
    cmp byte [os_mode],2
    je .chat
    mov edi,text_buffer
    mov esi,chat_prefix
    call copy_text
    mov rsi,[os_prompt]
    call copy_text
    mov esi,chat_suffix
    call copy_text
    mov esi,text_buffer
    call encode_text
    mov ebp,[token_count]
    mov eax,[max_new_tokens]
    lea rax,[rbp+rax-1]
    cmp eax,[context_limit]
    ja fail
    mov al,[os_mode]
    mov [text_mode],al          ; ids=0, text=1
    mov esi,os_output_begin
    call os_puts
    xor r14d,r14d
    call generate_prompt
    mov esi,os_output_end
    call os_puts
    xor edi,edi
    jmp exit
.chat:
    mov byte [text_mode],1
    mov byte [chat_mode],1
    call chat_loop
    xor edi,edi
    jmp exit

os_load_module:
    mov ebx,[boot_info]
    mov eax,[rbx]
    and eax,(1<<3)|(1<<6)
    cmp eax,(1<<3)|(1<<6)
    jne fail
    cmp dword [rbx+20],1
    jne fail
    mov edi,[rbx+24]
    mov r12d,[rdi]
    mov r13d,[rdi+4]
    cmp r12d,__kernel_end
    jb fail
    cmp r13d,r12d
    jbe fail
    mov eax,r13d
    sub eax,r12d
    cmp eax,BUNDLE_SIZE
    jne fail
    mov rsi,r12
    mov edi,bundle_header
    mov ecx,64
    repe cmpsb
    jne fail
    lea r15,[r12+MODEL_OFFSET]
    mov [model_base],r15
    lea rax,[r12+TOKENIZER_OFFSET]
    mov [tokenizer_base],rax
    mov [arena_floor],r13
    mov rax,[boot_metadata_end]
    call os_reserve
    ; Reserve the complete boot information, module descriptor and memory map.
    lea rax,[rbx+116]
    call os_reserve
    mov eax,[rbx+24]
    add rax,16
    call os_reserve
    mov eax,[rbx+48]
    mov edx,[rbx+44]
    add rax,rdx
    mov [mmap_end],rax
    call os_reserve
    ret

os_reserve:
    cmp rax,[arena_floor]
    jbe .done
    mov [arena_floor],rax
.done:
    ret

load_tokenizer:
    jmp validate_tokenizer

; Copy boot command line before allocating any RAM. prompt= consumes the rest.
os_options:
    mov dword [context_limit],2048
    mov dword [max_new_tokens],32
    test dword [rbx],1<<2
    jz .done
    mov esi,[rbx+16]
    test rsi,rsi
    jz .done
    mov edi,os_cmdline
    xor ecx,ecx
.copy:
    mov al,[rsi+rcx]
    mov [rdi+rcx],al
    inc ecx
    test al,al
    jz .copied
    cmp ecx,TEXT_BYTES
    jb .copy
    jmp fail
.copied:
    lea rax,[rsi+rcx]
    call os_reserve
    mov r12d,os_cmdline
.next:
    cmp byte [r12],' '
    jne .token
    inc r12
    jmp .next
.token:
    cmp byte [r12],0
    je .done
    mov rsi,r12
    mov edi,os_prompt_key
    call os_prefix
    test eax,eax
    jnz .prompt
    mov edi,os_option
    xor ecx,ecx
.word:
    mov al,[r12]
    test al,al
    jz .word_end
    cmp al,' '
    je .word_end
    cmp ecx,127
    jae fail
    mov [rdi+rcx],al
    inc r12
    inc ecx
    jmp .word
.word_end:
    mov byte [rdi+rcx],0
    mov esi,os_option
    mov edi,os_mode_key
    call os_prefix
    test eax,eax
    jnz .mode
    mov edi,os_tokens_key
    call os_prefix
    test eax,eax
    jnz .tokens
    mov edi,os_context_key
    call os_prefix
    test eax,eax
    jnz .context
    mov edi,os_threads_key
    call os_prefix
    test eax,eax
    jnz .threads
    ; QEMU prefixes the kernel filename. Only that first positional word is allowed.
    cmp byte [os_seen_option],0
    jne fail
    mov esi,os_option
.filename:
    lodsb
    cmp al,'='
    je fail
    test al,al
    jnz .filename
    mov byte [os_seen_option],1
    jmp .next
.mode:
    mov byte [os_seen_option],1
    mov esi,os_option+5
    mov edi,os_ids_value
    call option_equal
    test eax,eax
    jnz .ids
    mov edi,os_text_value
    call option_equal
    test eax,eax
    jnz .text
    mov edi,os_chat_value
    call option_equal
    test eax,eax
    jz fail
    mov byte [os_mode],2
    jmp .next
.ids:
    mov byte [os_mode],0
    jmp .next
.text:
    mov byte [os_mode],1
    jmp .next
.tokens:
    mov byte [os_seen_option],1
    mov esi,os_option+7
    mov ecx,MAX_CONTEXT
    call parse_uint
    test eax,eax
    jz fail
    mov [max_new_tokens],eax
    jmp .next
.context:
    mov byte [os_seen_option],1
    mov esi,os_option+8
    mov ecx,MAX_CONTEXT
    call parse_uint
    test eax,eax
    jz fail
    mov [context_limit],eax
    jmp .next
.threads:
    mov byte [os_seen_option],1
    mov esi,os_option+8
    mov ecx,16
    call parse_uint
    test eax,eax
    jz fail
    mov [requested_threads],eax
    jmp .next
.prompt:
    add r12,7
    mov [os_prompt],r12
.done:
    mov eax,[max_new_tokens]
    cmp eax,[context_limit]
    ja fail
    ret

; rsi=text, rdi=prefix. Returns 1 when text begins with the complete prefix.
os_prefix:
    xor ecx,ecx
.loop:
    mov al,[rdi+rcx]
    test al,al
    jz .yes
    cmp al,[rsi+rcx]
    jne .no
    inc ecx
    jmp .loop
.yes:
    mov eax,1
    ret
.no:
    xor eax,eax
    ret

; Pick the largest usable region after kernel, boot metadata and bundle.
; Physical addresses are restricted to the 4 GiB identity mapping.
os_find_arena:
    mov ebx,[boot_info]
    mov esi,[rbx+48]
    xor r9d,r9d
.entry:
    lea rax,[rsi+4]
    cmp rax,[mmap_end]
    ja fail
    mov eax,[rsi]
    cmp eax,20
    jb fail
    lea r10,[rsi+rax+4]
    cmp r10,[mmap_end]
    ja fail
    cmp dword [rsi+20],1
    jne .next
    mov rax,[rsi+4]
    mov rdx,[rsi+12]
    add rdx,rax
    jc .next
    mov rcx,0x100000000
    cmp rax,rcx
    jae .next
    cmp rdx,rcx
    cmova rdx,rcx
    cmp rax,[arena_floor]
    cmovb rax,[arena_floor]
    add rax,4095
    and rax,-4096
    and rdx,-4096
    cmp rax,rdx
    jae .next
    mov rcx,rdx
    sub rcx,rax
    cmp rcx,r9
    jbe .next
    mov r9,rcx
    mov [arena_next],rax
    mov [arena_end],rdx
.next:
    mov rsi,r10
    cmp rsi,[mmap_end]
    jb .entry
    test r9,r9
    jz fail
    ret

; rsi = size. Return zeroed page-aligned memory or a Linux-style negative error.
allocate_pages:
    mov rax,[arena_next]
    lea rcx,[rsi+4095]
    and rcx,-4096
    lea rdx,[rax+rcx]
    cmp rdx,[arena_end]
    ja .error
    mov [arena_next],rdx
    push rax
    mov rdi,rax
    xor eax,eax
    shr rcx,3
    rep stosq
    pop rax
    ret
.error:
    mov rax,-12
    ret

serial_init:
    mov dx,0x3ff
    in al,dx
    mov cl,al
    mov al,0x5a
    out dx,al
    in al,dx
    cmp al,0x5a
    jne .absent
    mov al,0xa5
    out dx,al
    in al,dx
    cmp al,0xa5
    jne .absent
    mov byte [serial_present],1
    mov al,cl
    out dx,al
    mov dx,0x3f9
    xor al,al
    out dx,al                   ; interrupts off
    mov dx,0x3fb
    mov al,0x80
    out dx,al
    mov dx,0x3f8
    mov al,1
    out dx,al                   ; divisor 1: 115200 baud
    mov dx,0x3f9
    xor al,al
    out dx,al
    mov dx,0x3fb
    mov al,3                    ; 8N1
    out dx,al
    mov dx,0x3fa
    mov al,0xc7
    out dx,al
    mov dx,0x3fc
    mov al,3
    out dx,al
    ret
.absent:
    mov al,cl
    out dx,al
    ret

; Same write_all contract as Linux. stdout and stderr share COM1.
write_all:
    push r8
    mov ecx,edx
    test ecx,ecx
    jz .done
.byte:
    lodsb
    call video_putc
    cmp byte [serial_present],0
    je .next
    push rax
    mov r8d,100000
    mov dx,0x3fd
.ready:
    in al,dx
    test al,0x20
    jnz .send
    pause
    dec r8d
    jnz .ready
    mov byte [serial_present],0
    pop rax
    jmp .next
.send:
    pop rax
    mov dx,0x3f8
    out dx,al
.next:
    dec ecx
    jnz .byte
.done:
    xor edx,edx
    pop r8
    ret

; Blocking one-byte serial read. Ctrl-D is EOF; CR and CRLF become LF.
platform_read:
    mov r8,rsi
.poll:
    cmp byte [serial_present],0
    je .keyboard
    mov dx,0x3fd
    in al,dx
    test al,1
    jnz .ready
.keyboard:
    call keyboard_poll
    test al,al
    jnz .input
    pause
    jmp .poll
.ready:
    mov dx,0x3f8
    in al,dx
.input:
    cmp al,4
    je .eof
    cmp al,10
    jne .not_lf
    cmp byte [serial_cr],0
    jne .skip_lf
.not_lf:
    mov byte [serial_cr],0
    cmp al,13
    jne .store
    mov byte [serial_cr],1
    mov al,10
.store:
    mov [r8],al
    cmp al,8
    je .read_done
    cmp al,127
    je .read_done
    mov rsi,r8
    mov edx,1
    call write_all             ; local echo for the serial terminal
.read_done:
    mov eax,1
    ret
.skip_lf:
    mov byte [serial_cr],0
    jmp .poll
.eof:
    xor eax,eax
    ret

os_puts:
    xor edx,edx
.length:
    cmp byte [rsi+rdx],0
    je .write
    inc edx
    jmp .length
.write:
    mov edi,2
    jmp write_all

fail:
    mov esi,os_error
    call os_puts
    mov edi,1
    jmp exit

exit:
    ; Distinguish guest success (33) from QEMU startup errors (normally 1).
    mov dword [workers_stopping],1
    mov eax,edi
    add eax,0x10
    mov dx,0xf4
    out dx,eax
    cli
.halt:
    hlt
    jmp .halt

section .rodata
BUNDLE_HEADER
os_banner: db OS_BANNER,10,0
os_avx: db "Kernel ready: AVX2/FMA",10,0
os_scalar: db "Kernel ready: scalar SSE2",10,0
os_error: db "FATAL: invalid boot module, model, memory map or arguments",10,0
os_output_begin: db "<lfm-output>",10,0
os_output_end: db 10,"</lfm-output>",10,0
os_default_prompt: db "What is the capital of France?",0
os_prompt_key: db "prompt=",0
os_mode_key: db "mode=",0
os_tokens_key: db "tokens=",0
os_context_key: db "context=",0
os_threads_key: db "threads=",0
os_ids_value: db "ids",0
os_text_value: db "text",0
os_chat_value: db "chat",0
section .data
os_prompt: dq os_default_prompt
os_mode: db 2
section .bss
os_cmdline: resb TEXT_BYTES
os_option: resb 128
os_seen_option: resb 1
serial_cr: resb 1
serial_present: resb 1
arena_floor: resq 1
arena_next: resq 1
arena_end: resq 1
mmap_end: resq 1
section .text
%include "os/multiboot.asm"
%include "os/console.asm"
