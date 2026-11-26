; Shared argument and number helpers.

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

