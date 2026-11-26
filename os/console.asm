; VGA text and 32-bit RGB framebuffer console, with a polled PS/2 keyboard.
; Serial always retains complete UTF-8; the framebuffer font covers Latin-1.
section .text
video_init:
%ifdef DUMP_LOGITS
    ret
%endif
    mov ebx,[boot_info]
    test dword [rbx],1<<12
    jz .legacy
    cmp byte [rbx+109],2
    je .vga
    cmp byte [rbx+109],1
    jne .done
    cmp byte [rbx+108],32
    jne .done
    cmp byte [rbx+111],8
    jne .done
    cmp byte [rbx+113],8
    jne .done
    cmp byte [rbx+115],8
    jne .done
    cmp byte [rbx+110],24
    ja .done
    cmp byte [rbx+112],24
    ja .done
    cmp byte [rbx+114],24
    ja .done
    mov rax,[rbx+88]
    mov rdx,rax
    shr rdx,32
    jnz .done
    test rax,rax
    jz .done
    mov [video_base],rax
    mov ecx,[rbx+100]
    cmp ecx,640
    jb .done
    cmp ecx,4096
    ja .done
    mov edx,[rbx+104]
    cmp edx,400
    jb .done
    cmp edx,2160
    ja .done
    mov edi,[rbx+96]
    lea esi,[rcx*4]
    cmp edi,esi
    jb .done
    cmp edi,32768
    ja .done
    mov [video_pitch],edi
    imul rdx,rdi
    add rdx,rax
    mov rsi,rdx
    shr rsi,32
    jnz .done
    ; Mark framebuffer large pages uncached (not normal write-back RAM).
    shr rax,21
    dec rdx
    shr rdx,21
.uncached:
    or qword [page_directories+rax*8],0x18
    inc eax
    cmp eax,edx
    jbe .uncached
    mov rax,cr3
    mov cr3,rax
    shr ecx,3
    mov [video_cols],ecx
    mov edx,[rbx+104]
    shr edx,4
    mov [video_rows],edx
    ; Pale foreground, channel placement supplied by the bootloader.
    mov eax,0xe0
    mov cl,[rbx+110]
    shl eax,cl
    mov edi,eax
    mov eax,0xe0
    mov cl,[rbx+112]
    shl eax,cl
    or edi,eax
    mov eax,0xe0
    mov cl,[rbx+114]
    shl eax,cl
    or edi,eax
    mov [video_ink],edi
    mov byte [video_kind],2
    mov rdi,[video_base]
    mov ecx,[rbx+96]
    imul ecx,[rbx+104]
    shr ecx,2
    xor eax,eax
    rep stosd
    ret
.legacy:
    cmp dword [boot_magic],0x36d76289
    je .done                   ; EFI without framebuffer is serial-only
.vga:
    mov byte [video_kind],1
    mov qword [video_base],0xb8000
    mov dword [video_cols],80
    mov dword [video_rows],25
    mov edi,0xb8000
    mov eax,0x07200720
    mov ecx,1000
    rep stosd
    ; Disable the hardware cursor; text rendering owns its position.
    mov dx,0x3d4
    mov al,0x0a
    out dx,al
    inc dx
    mov al,0x20
    out dx,al
.done:
    ret

; AL=one UTF-8 byte. Preserve every register for write_all callers.
video_putc:
    cmp byte [video_kind],0
    je .return
    push rax
    push rbx
    push rcx
    push rdx
    push rsi
    push rdi
    push r8
    push r9
    push r10
    push r11
    movzx ebx,al
    cmp byte [video_pending],0
    je .first
    and al,0xc0
    cmp al,0x80
    jne .replacement
    shl dword [video_unicode],6
    and ebx,0x3f
    or [video_unicode],ebx
    dec byte [video_pending]
    jnz .done
    mov ebx,[video_unicode]
    cmp ebx,255
    jbe .character
    jmp .replacement
.first:
    cmp bl,128
    jb .character
    mov byte [video_pending],1
    mov eax,ebx
    and eax,0x1f
    cmp bl,0xe0
    jb .prefix
    inc byte [video_pending]
    and eax,0x0f
    cmp bl,0xf0
    jb .prefix
    inc byte [video_pending]
    and eax,7
.prefix:
    mov [video_unicode],eax
    jmp .done
.replacement:
    mov byte [video_pending],0
    mov ebx,'?'
.character:
    cmp ebx,10
    je .newline
    cmp ebx,13
    je .carriage
    cmp ebx,8
    je .backspace
    cmp ebx,9
    jne .printable
    mov ebx,' '
.printable:
    cmp ebx,32
    jb .done
    cmp byte [video_kind],1
    jne .rgb
    cmp ebx,127
    jb .vga_glyph
    mov ebx,'?'                 ; VGA firmware font is not Latin-1
.vga_glyph:
    mov eax,[video_y]
    imul eax,[video_cols]
    add eax,[video_x]
    or ebx,0x0700
    mov rdi,[video_base]
    mov [rdi+rax*2],bx
    jmp .advance
.rgb:
    shl ebx,4
    lea rsi,[console_font+rbx]
    mov eax,[video_y]
    shl eax,4
    imul eax,[video_pitch]
    mov edi,[video_x]
    shl edi,5
    add rdi,rax
    add rdi,[video_base]
    mov r8d,16
.glyph_row:
    movzx edx,byte [rsi]
    inc rsi
    mov r9,rdi
    mov ecx,8
.pixel:
    xor eax,eax
    test edx,0x80
    cmovnz eax,[video_ink]
    mov [r9],eax
    add r9,4
    shl edx,1
    loop .pixel
    add edi,[video_pitch]
    dec r8d
    jnz .glyph_row
.advance:
    inc dword [video_x]
    mov eax,[video_cols]
    cmp [video_x],eax
    jb .done
.newline:
    mov dword [video_x],0
    inc dword [video_y]
    mov eax,[video_rows]
    cmp [video_y],eax
    jb .done
    dec dword [video_y]
    call video_scroll
    jmp .done
.carriage:
    mov dword [video_x],0
    jmp .done
.backspace:
    cmp dword [video_x],0
    je .previous_row
    dec dword [video_x]
    jmp .done
.previous_row:
    cmp dword [video_y],0
    je .done
    dec dword [video_y]
    mov eax,[video_cols]
    dec eax
    mov [video_x],eax
.done:
    pop r11
    pop r10
    pop r9
    pop r8
    pop rdi
    pop rsi
    pop rdx
    pop rcx
    pop rbx
    pop rax
.return:
    ret

video_scroll:
    mov rdi,[video_base]
    cmp byte [video_kind],1
    jne .rgb
    lea rsi,[rdi+160]
    mov ecx,80*24/2
    rep movsd
    mov eax,0x07200720
    mov ecx,40
    rep stosd
    ret
.rgb:
    mov eax,[video_pitch]
    shl eax,4
    lea rsi,[rdi+rax]
    mov ecx,[video_rows]
    dec ecx
    imul ecx,eax
    shr ecx,2
    rep movsd
    mov ecx,eax
    shr ecx,2
    xor eax,eax
    rep stosd
    ret

; Bounded i8042 operations. No controller means serial input still works.
kbd_wait_write:
    mov ecx,100000
.wait:
    in al,0x64
    test al,2
    jz .ready
    loop .wait
    stc
    ret
.ready:
    clc
    ret
kbd_wait_read:
    mov ecx,100000
.wait:
    in al,0x64
    test al,1
    jnz .ready
    loop .wait
    stc
    ret
.ready:
    in al,0x60
    clc
    ret
keyboard_init:
    mov ecx,256
.flush:
    in al,0x64
    test al,1
    jz .config
    in al,0x60
    loop .flush
    ret
.config:
    call kbd_wait_write
    jc .done
    mov al,0x20
    out 0x64,al
    call kbd_wait_read
    jc .done
    or al,0x40
    and al,0xec                ; translation on, IRQs off, first clock enabled
    mov bl,al
    call kbd_wait_write
    jc .done
    mov al,0x60
    out 0x64,al
    call kbd_wait_write
    jc .done
    mov al,bl
    out 0x60,al
    call kbd_wait_write
    jc .done
    mov al,0xae
    out 0x64,al
    call kbd_wait_write
    jc .done
    mov al,0xf4
    out 0x60,al
    call kbd_wait_read          ; consume ACK if present
    mov byte [keyboard_present],1
.done:
    ret

; Nonblocking ASCII input; zero means no printable key was read.
keyboard_poll:
    cmp byte [keyboard_present],0
    je .none
    in al,0x64
    test al,1
    jz .none
    test al,0x20
    jnz .mouse
    in al,0x60
    cmp al,0xe0
    je .extended
    cmp al,0x2a
    je .left_shift
    cmp al,0x36
    je .right_shift
    cmp al,0xaa
    je .left_release
    cmp al,0xb6
    je .right_release
    cmp al,0x1d
    je .ctrl
    cmp al,0x9d
    je .ctrl_release
    cmp al,0x3a
    je .caps
    test al,0x80
    jnz .clear
    cmp byte [keyboard_extended],0
    je .translate
    cmp al,0x1c
    jne .clear
.translate:
    mov byte [keyboard_extended],0
    movzx eax,al
    mov al,[keyboard_plain+rax]
    cmp al,'a'
    jb .symbol
    cmp al,'z'
    ja .symbol
    cmp byte [keyboard_ctrl],0
    jne .control_letter
    mov cl,[keyboard_shift]
    test cl,cl
    setnz cl
    xor cl,[keyboard_caps]
    jz .return
    sub al,32
    ret
.symbol:
    cmp byte [keyboard_shift],0
    je .return
    ; Find the shifted counterpart of punctuation, digits and whitespace.
    mov cl,al
    xor eax,eax
.find:
    cmp [keyboard_plain+rax],cl
    je .shifted
    inc eax
    cmp eax,128
    jb .find
    jmp .none
.shifted:
    mov al,[keyboard_shifted+rax]
.return:
    ret
.control_letter:
    and al,0x1f
    ret
.mouse:
    in al,0x60
    jmp .none
.extended:
    mov byte [keyboard_extended],1
    jmp .none
.left_shift:
    or byte [keyboard_shift],1
    jmp .clear
.right_shift:
    or byte [keyboard_shift],2
    jmp .clear
.left_release:
    and byte [keyboard_shift],~1
    jmp .clear
.right_release:
    and byte [keyboard_shift],~2
    jmp .clear
.ctrl:
    mov byte [keyboard_ctrl],1
    jmp .clear
.ctrl_release:
    mov byte [keyboard_ctrl],0
    jmp .clear
.caps:
    xor byte [keyboard_caps],1
.clear:
    mov byte [keyboard_extended],0
.none:
    xor eax,eax
    ret

section .rodata
align 16
console_font: incbin "console-font.bin"
keyboard_plain:
    db 0,27,"1234567890-=",8,9,"qwertyuiop[]",13,0,"asdfghjkl;",39,96,0,92
    db "zxcvbnm,./",0,"*",0," ",0
    times 128-($-keyboard_plain) db 0
keyboard_shifted:
    db 0,27,"!@#$%^&*()_+",8,9,"QWERTYUIOP{}",13,0,"ASDFGHJKL:",34,126,0,124
    db "ZXCVBNM<>?",0,"*",0," ",0
    times 128-($-keyboard_shifted) db 0
section .bss
video_kind: resb 1
video_base: resq 1
video_pitch: resd 1
video_cols: resd 1
video_rows: resd 1
video_x: resd 1
video_y: resd 1
video_ink: resd 1
video_pending: resb 1
video_unicode: resd 1
keyboard_present: resb 1
keyboard_extended: resb 1
keyboard_shift: resb 1
keyboard_ctrl: resb 1
keyboard_caps: resb 1
section .text
