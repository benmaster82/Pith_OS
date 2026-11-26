; Native byte-level BPE, Unicode pre-tokenization and streaming UTF-8 decoding.

section .text
; The platform supplies tokenizer_base; verify the complete compiled tables.
validate_tokenizer:
    mov rsi,[tokenizer_base]
    mov eax,2166136261
    xor ecx,ecx
.checksum:
    xor al,[rsi+rcx]
    imul eax,16777619
    inc ecx
    cmp ecx,TOK_SIZE
    jb .checksum
    cmp eax,TOK_CHECKSUM
    jne tokenizer_fail
    ret

; Strict UTF-8 scanner. rsi points to a character, rdi is the end.
; eax = codepoint or -1, ecx = consumed prefix, edx = incomplete flag.
; Invalid continuation bytes are left for the next scan, matching lossy UTF-8.
scan_utf8:
    movzx eax,byte [rsi]
    mov ecx,1
    xor edx,edx
    cmp eax,128
    jb .done
    cmp eax,0xc2
    jb .invalid
    cmp eax,0xf4
    ja .invalid
    mov r8d,eax
    mov r9d,2
    cmp eax,0xe0
    jb .two
    inc r9d
    cmp eax,0xf0
    jb .three
    inc r9d
    and eax,7
    jmp .continuation
.three:
    and eax,15
    jmp .continuation
.two:
    and eax,31
.continuation:
    lea rdx,[rsi+rcx]
    cmp rdx,rdi
    jae .incomplete
    movzx edx,byte [rdx]
    mov r10d,edx
    and r10d,0xc0
    cmp r10d,0x80
    jne .invalid
    cmp ecx,1
    jne .combine
    cmp r8d,0xe0
    jne .surrogate
    cmp edx,0xa0
    jb .invalid
.surrogate:
    cmp r8d,0xed
    jne .overlong
    cmp edx,0xa0
    jae .invalid
.overlong:
    cmp r8d,0xf0
    jne .maximum
    cmp edx,0x90
    jb .invalid
.maximum:
    cmp r8d,0xf4
    jne .combine
    cmp edx,0x90
    jae .invalid
.combine:
    shl eax,6
    and edx,63
    or eax,edx
    inc ecx
    cmp ecx,r9d
    jb .continuation
    xor edx,edx
.done:
    ret
.invalid:
    mov eax,-1
    xor edx,edx
    ret
.incomplete:
    mov eax,-1
    mov edx,1
    ret

; Find longest added-token match at rsi, bounded by rdi.
; Returns ID in eax (-1 on no match) and byte length in ecx.
find_added:
    mov r8,[tokenizer_base]
    add r8,TOK_TRIE
    xor r9d,r9d
    xor r10d,r10d
    mov eax,-1
    xor ecx,ecx
.next:
    lea rdx,[rsi+r10]
    cmp rdx,rdi
    jae .done
    movzx edx,byte [rdx]
    shl r9d,4
    mov r9d,[r8+r9]
.child:
    test r9d,r9d
    jz .done
    mov r11d,r9d
    shl r11d,4
    cmp edx,[r8+r11+8]
    je .matched
    mov r9d,[r8+r11+4]
    jmp .child
.matched:
    inc r10d
    cmp dword [r8+r11+12],-1
    je .next
    mov eax,[r8+r11+12]
    mov ecx,r10d
    jmp .next
.done:
    ret

append_token:
    mov edx,[token_count]
    cmp edx,[context_limit]
    jae tokenizer_input_fail
    mov [prompt_ids+rdx*4],eax
    inc dword [token_count]
    ret

; Encode rsi..rdi, without implicit BOS. Added tokens split ordinary text
; before the Unicode regex is applied, including non-special added tokens.
encode_text:
    push rbx
    push rbp
    push r12
    push r13
    push r14
    push r15
    mov r12,rsi
    mov r13,rsi
    mov r14,rdi
    mov dword [token_count],0
.scan:
    cmp r13,r14
    jae .tail
    mov rsi,r13
    mov rdi,r14
    call find_added
    cmp eax,-1
    je .advance
    mov r15d,eax
    lea rbx,[r13+rcx]
    mov rsi,r12
    mov rdi,r13
    call encode_plain
    mov eax,r15d
    call append_token
    mov r13,rbx
    mov r12,rbx
    jmp .scan
.advance:
    inc r13
    jmp .scan
.tail:
    mov rsi,r12
    mov rdi,r14
    call encode_plain
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbp
    pop rbx
    ret

; Translate a plain span to Unicode codepoints, retaining original byte offsets.
encode_plain:
    push rbx
    push rbp
    push r12
    push r13
    push r14
    push r15
    mov r12,rsi
    mov r13,rdi
    mov r15,[tokenizer_base]
    xor r14d,r14d
.characters:
    cmp rsi,r13
    jae .split
    mov rax,rsi
    sub rax,r12
    mov [cp_offsets+r14*4],eax
    mov rdi,r13
    call scan_utf8
    cmp eax,-1
    je tokenizer_input_fail
    mov [cp_values+r14*4],eax
    mov dl,[r15+TOK_FLAGS+rax]
    mov [cp_flags+r14],dl
    add rsi,rcx
    inc r14d
    jmp .characters
.split:
    mov rax,r13
    sub rax,r12
    mov [cp_offsets+r14*4],eax
    mov rbp,r14
    xor ebx,ebx
.segment:
    cmp rbx,rbp
    jae .done
    ; (?i:'s|'t|'re|'ve|'m|'ll|'d)
    cmp dword [cp_values+rbx*4],39
    jne .word
    lea edx,[rbx+1]
    cmp rdx,rbp
    jae .word
    mov eax,[cp_values+rdx*4]
    cmp eax,0x17f               ; Unicode long s participates in case folding
    je .short_contraction
    or eax,32
    cmp eax,'s'
    je .short_contraction
    cmp eax,'t'
    je .short_contraction
    cmp eax,'m'
    je .short_contraction
    cmp eax,'d'
    je .short_contraction
    inc edx
    cmp rdx,rbp
    jae .word
    mov ecx,[cp_values+rdx*4]
    or ecx,32
    cmp eax,'l'
    je .double_l
    cmp eax,'r'
    je .ends_e
    cmp eax,'v'
    jne .word
.ends_e:
    cmp ecx,'e'
    jne .word
    jmp .short_contraction
.double_l:
    cmp ecx,'l'
    jne .word
.short_contraction:
    inc edx
    jmp .piece
.word:
    ; One optional non-letter/non-number prefix, excluding CR and LF.
    mov edx,ebx
    test byte [cp_flags+rdx],1
    jnz .letters
    test byte [cp_flags+rdx],3
    jnz .number
    mov eax,[cp_values+rdx*4]
    cmp eax,10
    je .number
    cmp eax,13
    je .number
    inc edx
    cmp rdx,rbp
    jae .number
    test byte [cp_flags+rdx],1
    jz .number
.letters:
    inc edx
    cmp rdx,rbp
    jae .piece
    test byte [cp_flags+rdx],1
    jnz .letters
    jmp .piece
.number:
    test byte [cp_flags+rbx],2
    jz .punctuation
    lea edx,[rbx+1]
    lea ecx,[rbx+3]
.digits:
    cmp rdx,rbp
    jae .piece
    cmp edx,ecx
    jae .piece
    test byte [cp_flags+rdx],2
    jz .piece
    inc edx
    jmp .digits
.punctuation:
    mov edx,ebx
    cmp dword [cp_values+rdx*4],32
    jne .punct_start
    inc edx
.punct_start:
    cmp rdx,rbp
    jae .whitespace
    test byte [cp_flags+rdx],7
    jnz .whitespace
.punct_loop:
    inc edx
    cmp rdx,rbp
    jae .piece
    test byte [cp_flags+rdx],7
    jz .punct_loop
.newlines:
    mov eax,[cp_values+rdx*4]
    cmp eax,10
    je .consume_newline
    cmp eax,13
    jne .piece
.consume_newline:
    inc edx
    cmp rdx,rbp
    jb .newlines
    jmp .piece
.whitespace:
    mov edx,ebx
    xor eax,eax
.spaces:
    cmp rdx,rbp
    jae .space_end
    test byte [cp_flags+rdx],4
    jz .space_end
    mov ecx,[cp_values+rdx*4]
    inc edx
    cmp ecx,10
    je .last_newline
    cmp ecx,13
    jne .spaces
.last_newline:
    mov eax,edx
    jmp .spaces
.space_end:
    test eax,eax
    jnz .newline_end
    cmp rdx,rbp
    jae .piece
    lea ecx,[rbx+1]
    cmp edx,ecx
    jbe .piece
    dec edx                    ; trailing-space lookahead leaves the final space
    jmp .piece
.newline_end:
    mov edx,eax
.piece:
    cmp edx,ebx
    jbe tokenizer_input_fail
    mov r14d,edx
    mov eax,[cp_offsets+rbx*4]
    lea rsi,[r12+rax]
    mov ecx,[cp_offsets+r14*4]
    sub ecx,eax
    call encode_piece
    mov rbx,r14
    jmp .segment
.done:
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbp
    pop rbx
    ret

; eax/edx = left/right token IDs. Returns rank in eax, merged ID in edx.
lookup_pair:
    mov r8d,edx
    shl r8,32
    or r8,rax
    imul r9d,eax,0x9e3779b1
    imul r10d,edx,0x85ebca6b
    xor r9d,r10d
    mov r10,[tokenizer_base]
    add r10,TOK_MERGES
.probe:
    and r9d,TOK_HASH_MASK
    mov r11d,r9d
    shl r11d,4
    add r11,r10
    cmp dword [r11+8],0
    je .missing
    cmp [r11],r8
    je .found
    inc r9d
    jmp .probe
.found:
    mov eax,[r11+8]
    dec eax
    mov edx,[r11+12]
    ret
.missing:
    mov eax,-1
    ret

update_pair:
    push r12
    mov r12d,eax
    mov edx,[bpe_next+r12*4]
    cmp edx,-1
    je .last
    mov edx,[bpe_ids+rdx*4]
    mov eax,[bpe_ids+r12*4]
    call lookup_pair
    mov [bpe_rank+r12*4],eax
    mov [bpe_merged+r12*4],edx
    jmp .done
.last:
    mov dword [bpe_rank+r12*4],-1
.done:
    pop r12
    ret

; Greedy BPE with linked positions. Only neighboring ranks change after a merge.
encode_piece:
    push rbx
    push rbp
    push r12
    push r13
    push r14
    push r15
    mov ebp,ecx
    mov r15,[tokenizer_base]
    xor ecx,ecx
.initialize:
    movzx eax,byte [rsi+rcx]
    mov eax,[r15+TOK_BYTE_IDS+rax*4]
    mov [bpe_ids+rcx*4],eax
    lea eax,[rcx+1]
    mov [bpe_next+rcx*4],eax
    lea eax,[rcx-1]
    mov [bpe_prev+rcx*4],eax
    inc ecx
    cmp ecx,ebp
    jb .initialize
    mov dword [bpe_next+rcx*4-4],-1
    xor r12d,r12d
.ranks:
    mov eax,r12d
    call update_pair
    inc r12d
    cmp r12d,ebp
    jb .ranks
.select:
    xor r12d,r12d
    mov r13d,-1
    mov r14d,-1
.scan:
    mov eax,[bpe_rank+r12*4]
    cmp eax,r13d
    jae .next
    mov r13d,eax
    mov r14d,r12d
.next:
    mov r12d,[bpe_next+r12*4]
    cmp r12d,-1
    jne .scan
    cmp r14d,-1
    je .emit
    mov eax,[bpe_merged+r14*4]
    mov [bpe_ids+r14*4],eax
    mov edx,[bpe_next+r14*4]
    mov ecx,[bpe_next+rdx*4]
    mov [bpe_next+r14*4],ecx
    cmp ecx,-1
    je .update
    mov [bpe_prev+rcx*4],r14d
.update:
    mov eax,r14d
    call update_pair
    mov eax,[bpe_prev+r14*4]
    cmp eax,-1
    je .select
    call update_pair
    jmp .select
.emit:
    xor r12d,r12d
.output:
    mov eax,[bpe_ids+r12*4]
    call append_token
    mov r12d,[bpe_next+r12*4]
    cmp r12d,-1
    jne .output
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbp
    pop rbx
    ret

; Copy a C string into the bounded text buffer. rdi advances.
copy_text:
    lea rdx,[text_buffer+TEXT_BYTES]
.copy:
    cmp rdi,rdx
    jae tokenizer_input_fail
    mov al,[rsi]
    test al,al
    jz .done
    mov [rdi],al
    inc rsi
    inc rdi
    jmp .copy
.done:
    ret

; eax = token ID. Skip configured special tokens and retain incomplete UTF-8.
decode_token:
    push rbx
    push rbp
    push r12
    push r13
    push r14
    push r15
    mov r15,[tokenizer_base]
    cmp byte [r15+TOK_SPECIAL+rax],0
    jne .done
    lea rbx,[r15+TOK_RECORDS+rax*8]
    mov ecx,[rbx+4]
    test ecx,ecx
    jz .done
    mov esi,[rbx]
    lea rsi,[r15+TOK_BYTES+rsi]
    mov edi,[decode_length]
    lea rdi,[decode_buffer+rdi]
    add [decode_length],ecx
    cmp dword [decode_length],128
    ja tokenizer_input_fail
    rep movsb
    call decode_pending
.done:
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbp
    pop rbx
    ret

decode_pending:
    push r12
    push r13
    xor r12d,r12d
    mov r13d,[decode_length]
.scan:
    cmp r12d,r13d
    jae .compact
    lea rsi,[decode_buffer+r12]
    lea rdi,[decode_buffer+r13]
    call scan_utf8
    cmp eax,-1
    jne .valid
    test edx,edx
    jz .replacement
    cmp byte [decode_final],0
    je .compact
.replacement:
    add r12d,ecx
    mov esi,replacement_utf8
    mov edx,3
    jmp .write
.valid:
    lea rsi,[decode_buffer+r12]
    mov edx,ecx
    add r12d,ecx
.write:
    mov edi,1
    call write_all
    jmp .scan
.compact:
    sub r13d,r12d
    mov [decode_length],r13d
    lea rsi,[decode_buffer+r12]
    mov edi,decode_buffer
    mov ecx,r13d
    rep movsb
    pop r13
    pop r12
    ret

tokenizer_fail:
    mov esi,tokenizer_error
    mov edx,tokenizer_error_len
    jmp tokenizer_error_exit
tokenizer_input_fail:
    cmp byte [chat_tokenizing],0
    jne chat_tokenizer_recover
    mov esi,tokenizer_input_error
    mov edx,tokenizer_input_error_len
tokenizer_error_exit:
    mov edi,2
    call write_all
    mov edi,1
    jmp exit

section .rodata
self_exe: db "/proc/self/exe",0
tokenizer_name: db "tokenizer.bin",0,0
%ifdef BARE_METAL
tokenizer_error: db "Invalid tokenizer in boot module",10
%else
tokenizer_error: db "Cannot load matching tokenizer.bin beside the executable",10
%endif
tokenizer_error_len equ $-tokenizer_error
tokenizer_input_error: db "Invalid UTF-8 or prompt exceeds byte/context limits",10
tokenizer_input_error_len equ $-tokenizer_input_error
replacement_utf8: db 0xef,0xbf,0xbd

section .bss
tokenizer_base: resq 1
tokenizer_path: resb 4096
text_buffer: resb TEXT_BYTES
token_count: resd 1
cp_values: resd TEXT_BYTES
cp_offsets: resd TEXT_BYTES+1
cp_flags: resb TEXT_BYTES
bpe_ids: resd TEXT_BYTES
bpe_next: resd TEXT_BYTES
bpe_prev: resd TEXT_BYTES
bpe_rank: resd TEXT_BYTES
bpe_merged: resd TEXT_BYTES
decode_buffer: resb 128
decode_length: resd 1
decode_final: resb 1

section .text
