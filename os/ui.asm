; Pithos CORE console. A framebuffer interface with a header, a scrolling
; transcript and a status bar; Unicode glyphs and the logo from ui_font.py,
; keyboard layouts from keymaps.py. VGA text and COM1 remain available.
; Text is drawn into a shadow buffer in RAM and copied to the framebuffer,
; which is mapped uncached, so scrolling never reads video memory.
%define UI_HEADER 40                ; header band height in pixels
%define UI_STATUS 32                ; status band height
%define UI_LINE 20                  ; transcript line height
%define UI_MARGIN 24
%define UI_MAX_WIDTH 1920
%define UI_MAX_HEIGHT 1200
; The zinc palette of the pithos web interface, RGB.
%define UI_PAGE 0x111113
%define UI_BAR 0x18181b
%define UI_BORDER 0x2c2c31
%define UI_TEXT 0xececf0
%define UI_BRIGHT 0xffffff
%define UI_MUTED 0x9d9da8
%define UI_FAINT 0x6b6b76
%define UI_ACCENT 0xfbbf24
%define UI_ERROR 0xf87171
%define UI_LINE_GLYPHS 512           ; glyph widths kept for backspace
%define UI_WORD 64                   ; longest word moved by line wrapping
%include "os/keymaps.inc"
section .text
video_init:
%ifdef DUMP_LOGITS
    ret
%endif
    mov dword [ui_fg],UI_MUTED       ; boot messages are secondary
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
    mov [ui_base],rax
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
    mov [ui_pitch],edi
    mov r8d,edx
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
    mov eax,UI_MAX_WIDTH
    cmp ecx,eax
    cmova ecx,eax
    mov [ui_width],ecx
    mov eax,UI_MAX_HEIGHT
    cmp r8d,eax
    cmova r8d,eax
    mov [ui_height],r8d
    mov al,[rbx+110]
    mov [ui_red_shift],al
    mov al,[rbx+112]
    mov [ui_green_shift],al
    mov al,[rbx+114]
    mov [ui_blue_shift],al
    mov byte [video_kind],2
    ; The whole framebuffer, beyond the drawn area too, takes the page colour.
    mov eax,UI_PAGE
    call ui_pack
    mov rdi,[ui_base]
    mov ecx,[rbx+96]
    imul ecx,[rbx+104]
    shr ecx,2
    rep stosd
    call ui_layout
    ret
.legacy:
    cmp dword [boot_magic],0x36d76289
    je .done                   ; EFI without framebuffer is serial-only
.vga:
    mov byte [video_kind],1
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

; eax = 0xRRGGBB. Returns the framebuffer pixel. Clobbers ecx, edx, r8d.
ui_pack:
    mov edx,eax
    movzx r8d,dl
    mov cl,[ui_blue_shift]
    shl r8d,cl
    movzx eax,dh
    mov cl,[ui_green_shift]
    shl eax,cl
    or r8d,eax
    shr edx,16
    movzx eax,dl
    mov cl,[ui_red_shift]
    shl eax,cl
    or eax,r8d
    ret

; c = (f * a + b * (255 - a) + 128) * 257 >> 16, exact rounding of / 255.
%macro BLEND_CHANNEL 1
    mov r8d,r9d
    shr r8d,%1
    and r8d,255
    imul r8d,edx
    mov r11d,r10d
    shr r11d,%1
    and r11d,255
    mov ecx,255
    sub ecx,edx
    imul r11d,ecx
    add r8d,r11d
    add r8d,128
    imul r8d,257
    shr r8d,16
    shl r8d,%1
    or eax,r8d
%endmacro

; edx = coverage 0..255, r9d = foreground RGB, r10d = background RGB.
; Returns the blended framebuffer pixel. Clobbers ecx, edx, r8d, r11d.
ui_blend:
    xor eax,eax
    BLEND_CHANNEL 16
    BLEND_CHANNEL 8
    BLEND_CHANNEL 0
    jmp ui_pack

; ecx = codepoint. Returns eax = glyph kind (0 when absent), rbx = its data.
ui_lookup:
    xor eax,eax
    cmp ecx,0x10ffff
    ja .done
    mov edx,ecx
    shr edx,8
    mov edx,[ui_font+16+rdx*4]
    test edx,edx
    jz .done
    movzx r8d,cl
    mov edx,[ui_font+rdx+r8*4]
    test edx,edx
    jz .done
    mov eax,edx
    shr eax,24
    and edx,0xffffff
    lea rbx,[ui_font+rdx]
.done:
    ret

; Kinds 2 and 4 are wide (two cells).
%macro GLYPH_WIDTH 0
    mov edx,8
    cmp eax,2
    je %%wide
    cmp eax,4
    jne %%done
%%wide:
    mov edx,16
%%done:
%endmacro

; ecx = codepoint, esi = x, edi = y, r9d = foreground, r10d = background.
; Draws into the shadow buffer and returns eax = width, 0 when absent.
ui_glyph:
    push rbx
    push rbp
    push r12
    push r13
    push r14
    push r15
    mov r12d,esi
    mov r13d,edi
    call ui_lookup
    test eax,eax
    jz .done
    mov r14d,eax
    GLYPH_WIDTH
    mov [ui_glyph_width],edx
    mov eax,r9d
    call ui_pack
    mov [ui_fg_pixel],eax
    mov eax,r10d
    call ui_pack
    mov r15d,eax                    ; background pixel
    mov eax,r13d
    imul eax,[ui_width]
    add eax,r12d
    lea rbp,[ui_shadow+rax*4]
    xor r12d,r12d                   ; row
.row:
    xor r13d,r13d                   ; column
.pixel:
    cmp r14d,3
    je .coverage
    cmp r14d,4
    je .colour
    ; One bit per pixel: kind 1 one byte per row, kind 2 two bytes.
    mov eax,r12d
    cmp r14d,2
    jne .bits
    shl eax,1
    mov edx,r13d
    shr edx,3
    add eax,edx
.bits:
    movzx eax,byte [rbx+rax]
    mov ecx,r13d
    and ecx,7
    shl eax,cl
    test al,0x80
    mov eax,r15d
    jz .store
    mov eax,[ui_fg_pixel]
    jmp .store
.coverage:
    mov eax,r12d
    shl eax,3
    add eax,r13d
    movzx edx,byte [rbx+rax]
    test edx,edx
    mov eax,r15d
    jz .store
    mov eax,[ui_fg_pixel]
    cmp edx,255
    je .store
    call ui_blend
    jmp .store
.colour:
    mov eax,r12d
    shl eax,4
    add eax,r13d
    mov ecx,[rbx+rax*4]             ; R, G, B, A bytes
    movzx edx,cl
    shl edx,16
    mov eax,ecx
    and eax,0xff00
    or edx,eax
    mov eax,ecx
    shr eax,16
    and eax,0xff
    or edx,eax
    push r9
    mov r9d,edx
    mov edx,ecx
    shr edx,24
    call ui_blend
    pop r9
.store:
    mov [rbp+r13*4],eax
    inc r13d
    cmp r13d,[ui_glyph_width]
    jb .pixel
    mov eax,[ui_width]
    lea rbp,[rbp+rax*4]
    inc r12d
    cmp r12d,16
    jb .row
    mov eax,[ui_glyph_width]
.done:
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbp
    pop rbx
    ret

; esi = x, edi = y, edx = width, ecx = height: copy the shadow rectangle to
; the framebuffer. Clobbers rax, rcx, rdx, rsi, rdi, r8, r9.
ui_blit:
    mov r8d,edx
    mov r9d,ecx
    mov eax,edi
    imul eax,[ui_width]
    add eax,esi
    lea rax,[ui_shadow+rax*4]
    mov ecx,edi
    imul ecx,[ui_pitch]
    lea rdx,[rcx+rsi*4]
    add rdx,[ui_base]
.row:
    mov rsi,rax
    mov rdi,rdx
    mov ecx,r8d
    rep movsd
    mov ecx,[ui_width]
    lea rax,[rax+rcx*4]
    mov ecx,[ui_pitch]
    add rdx,rcx
    dec r9d
    jnz .row
    ret

; esi = x, edi = y, edx = width, ecx = height, eax = RGB: fill the shadow
; rectangle and copy it. Clobbers rax, rcx, rdx, rsi, rdi, r8-r11.
ui_fill:
    push rbx
    push r12
    push r13
    mov r12d,edx
    mov r13d,ecx
    push rsi
    push rdi
    call ui_pack
    pop rdi
    pop rsi
    mov r10d,eax
    mov eax,edi
    imul eax,[ui_width]
    add eax,esi
    lea rbx,[ui_shadow+rax*4]
    mov r11d,r13d
.row:
    push rdi
    mov rdi,rbx
    mov eax,r10d
    mov ecx,r12d
    rep stosd
    pop rdi
    mov eax,[ui_width]
    lea rbx,[rbx+rax*4]
    dec r11d
    jnz .row
    mov edx,r12d
    mov ecx,r13d
    call ui_blit
    pop r13
    pop r12
    pop rbx
    ret

; rsi = UTF-8 text. Returns ecx = next codepoint (0 at the end), advances rsi.
; Our own strings and checked model text only; malformed bytes become U+FFFD.
ui_next:
    movzx ecx,byte [rsi]
    test ecx,ecx
    jz .done
    inc rsi
    cmp ecx,0x80
    jb .done
    mov edx,1
    and ecx,0x1f
    cmp byte [rsi-1],0xe0
    jb .continue
    inc edx
    and ecx,0x0f
    cmp byte [rsi-1],0xf0
    jb .continue
    inc edx
    and ecx,0x07
.continue:
    movzx eax,byte [rsi]
    mov r8d,eax
    and r8d,0xc0
    cmp r8d,0x80
    jne .invalid
    shl ecx,6
    and eax,0x3f
    or ecx,eax
    inc rsi
    dec edx
    jnz .continue
.done:
    ret
.invalid:
    mov ecx,0xfffd
    ret

; rsi = UTF-8 text, r12d = x, r13d = y, r9d/r10d = colours, r14d = right
; limit. Draws into the shadow buffer and advances r12d.
ui_text:
    push rbx
.next:
    call ui_next
    test ecx,ecx
    jz .done
    push rcx
    call ui_lookup
    pop rcx
    test eax,eax
    jz .next
    GLYPH_WIDTH
    lea eax,[r12+rdx]
    cmp eax,r14d
    ja .done
    push rsi
    mov esi,r12d
    mov edi,r13d
    call ui_glyph
    pop rsi
    add r12d,eax
    jmp .next
.done:
    pop rbx
    ret

; rsi = UTF-8 text. Returns eax = its width in pixels.
ui_text_width:
    push rbx
    xor r9d,r9d
.next:
    call ui_next
    test ecx,ecx
    jz .done
    call ui_lookup
    test eax,eax
    jz .next
    GLYPH_WIDTH
    add r9d,edx
    jmp .next
.done:
    mov eax,r9d
    pop rbx
    ret

; Draw the header, the empty transcript and the status band.
ui_layout:
    mov eax,[ui_height]
    sub eax,UI_STATUS+9
    mov [ui_bottom],eax
    mov dword [ui_top],UI_HEADER+13
    mov eax,[ui_width]
    sub eax,UI_MARGIN
    mov [ui_right],eax
    mov dword [ui_x],UI_MARGIN
    mov dword [ui_y],UI_HEADER+13
    xor esi,esi
    xor edi,edi
    mov edx,[ui_width]
    mov ecx,[ui_height]
    mov eax,UI_PAGE
    call ui_fill
    call ui_header
    mov esi,ui_starting
    jmp ui_status

ui_header:
    push r12
    push r13
    push r14
    xor esi,esi
    xor edi,edi
    mov edx,[ui_width]
    mov ecx,UI_HEADER
    mov eax,UI_BAR
    call ui_fill
    xor esi,esi
    mov edi,UI_HEADER
    mov edx,[ui_width]
    mov ecx,1
    mov eax,UI_BORDER
    call ui_fill
    ; The mark, 24 pixels, antialiased against the header band.
    mov r9d,UI_TEXT
    mov r10d,UI_BAR
    mov esi,[ui_font+12]            ; logo offset
    lea rsi,[ui_font+rsi+8]
    lea rdi,[ui_shadow+16*4]         ; x 16, y 8
    mov eax,[ui_width]
    imul eax,8*4
    add rdi,rax
    xor r12d,r12d
.logo_row:
    xor r13d,r13d
.logo_pixel:
    movzx edx,byte [rsi]
    inc rsi
    push rsi
    push rdi
    call ui_blend
    pop rdi
    pop rsi
    mov [rdi+r13*4],eax
    inc r13d
    cmp r13d,24
    jb .logo_pixel
    mov eax,[ui_width]
    lea rdi,[rdi+rax*4]
    inc r12d
    cmp r12d,24
    jb .logo_row
    mov esi,ui_title
    mov r12d,50
    mov r13d,12
    mov r9d,UI_TEXT
    mov r10d,UI_BAR
    mov r14d,[ui_width]
    call ui_text
    mov esi,ui_subtitle
    call ui_text_width
    mov r12d,[ui_width]
    sub r12d,16
    sub r12d,eax
    mov esi,ui_subtitle
    mov r9d,UI_MUTED
    mov r10d,UI_BAR
    mov r14d,[ui_width]
    call ui_text
    xor esi,esi
    xor edi,edi
    mov edx,[ui_width]
    mov ecx,UI_HEADER
    call ui_blit
    pop r14
    pop r13
    pop r12
    ret

; rsi = UTF-8 status text, drawn in the status band with the help hint.
ui_status:
    cmp byte [video_kind],2
    jne .done
    push rbx
    push r12
    push r13
    push r14
    push rsi
    xor esi,esi
    mov edi,[ui_height]
    sub edi,UI_STATUS+1
    mov edx,[ui_width]
    mov ecx,1
    mov eax,UI_BORDER
    call ui_fill
    xor esi,esi
    mov edi,[ui_height]
    sub edi,UI_STATUS
    mov edx,[ui_width]
    mov ecx,UI_STATUS
    mov eax,UI_BAR
    call ui_fill
    mov esi,ui_hint
    call ui_text_width
    mov ebx,[ui_width]
    sub ebx,16
    sub ebx,eax
    mov r12d,ebx
    mov r13d,[ui_height]
    sub r13d,UI_STATUS-8
    mov esi,ui_hint
    mov r9d,UI_FAINT
    mov r10d,UI_BAR
    mov r14d,[ui_width]
    call ui_text
    pop rsi
    mov r12d,16
    mov r9d,UI_MUTED
    lea r14d,[rbx-16]
    call ui_text
    xor esi,esi
    mov edi,[ui_height]
    sub edi,UI_STATUS
    mov edx,[ui_width]
    mov ecx,UI_STATUS
    call ui_blit
    pop r14
    pop r13
    pop r12
    pop rbx
.done:
    ret

; AL = one byte of UTF-8 text with optional SGR colour sequences. Preserve
; every register for write_all callers.
video_putc:
    cmp byte [video_kind],0
    je .return
    push rax
    push rbx
    push rcx
    push rdx
    push rsi
    push rdi
    push rbp
    push r8
    push r9
    push r10
    push r11
    push r12
    push r13
    push r14
    push r15
    movzx eax,al
    call ui_byte
    pop r15
    pop r14
    pop r13
    pop r12
    pop r11
    pop r10
    pop r9
    pop r8
    pop rbp
    pop rdi
    pop rsi
    pop rdx
    pop rcx
    pop rbx
    pop rax
.return:
    ret

; eax = byte. ESC [ parameters m selects a colour; other sequences are
; consumed silently. UTF-8 is decoded here; malformed input becomes U+FFFD.
ui_byte:
    movzx ecx,byte [ui_escape]
    test ecx,ecx
    jnz .escape
    cmp eax,27
    je .start_escape
    mov ecx,[ui_pending]
    test ecx,ecx
    jnz .continuation
    cmp eax,0x80
    jb .codepoint
    mov [ui_unicode],eax
    mov dword [ui_pending],1
    and dword [ui_unicode],0x1f
    cmp eax,0xc2
    jb .invalid
    cmp eax,0xe0
    jb .wait
    mov dword [ui_pending],2
    and dword [ui_unicode],0x0f
    cmp eax,0xf0
    jb .wait
    mov dword [ui_pending],3
    and dword [ui_unicode],0x07
    cmp eax,0xf4
    ja .invalid
.wait:
    ret
.continuation:
    mov edx,eax
    and edx,0xc0
    cmp edx,0x80
    jne .broken
    shl dword [ui_unicode],6
    and eax,0x3f
    or [ui_unicode],eax
    dec dword [ui_pending]
    jnz .wait
    mov eax,[ui_unicode]
.codepoint:
    mov ecx,eax
    cmp byte [video_kind],1
    je vga_codepoint
    jmp ui_codepoint
.broken:
    ; A new lead byte interrupts the sequence: show U+FFFD, then retry it.
    mov dword [ui_pending],0
    push rax
    mov ecx,0xfffd
    call .route
    pop rax
    jmp ui_byte
.invalid:
    mov dword [ui_pending],0
    mov ecx,0xfffd
.route:
    cmp byte [video_kind],1
    je vga_codepoint
    jmp ui_codepoint
.start_escape:
    mov byte [ui_escape],1
    mov dword [ui_param_count],0
    mov dword [ui_params],0
    ret
.escape:
    cmp ecx,1
    jne .parameters
    cmp eax,'['
    mov byte [ui_escape],2
    je .wait
    mov byte [ui_escape],0
    ret
.parameters:
    cmp eax,'0'
    jb .separator
    cmp eax,'9'
    ja .separator
    mov ecx,[ui_param_count]
    cmp ecx,7
    ja .wait
    mov edx,[ui_params+rcx*4]
    imul edx,10
    lea edx,[rdx+rax-'0']
    mov [ui_params+rcx*4],edx
    ret
.separator:
    cmp eax,';'
    jne .final
    mov ecx,[ui_param_count]
    cmp ecx,7
    jae .wait
    inc ecx
    mov [ui_param_count],ecx
    mov dword [ui_params+rcx*4],0
    ret
.final:
    mov byte [ui_escape],0
    cmp eax,'m'
    jne .wait
    xor ecx,ecx
.sgr:
    mov eax,[ui_params+rcx*4]
    mov edx,UI_TEXT
    test eax,eax
    jz .colour
    cmp eax,22
    je .colour
    cmp eax,37
    je .colour
    cmp eax,39
    je .colour
    mov edx,UI_BRIGHT
    cmp eax,1
    je .colour
    mov edx,UI_MUTED
    cmp eax,2
    je .colour
    mov edx,UI_ERROR
    cmp eax,31
    je .colour
    mov edx,UI_ACCENT
    cmp eax,33
    je .colour
    mov edx,UI_FAINT
    cmp eax,90
    jne .next_sgr
.colour:
    mov [ui_fg],edx
.next_sgr:
    inc ecx
    cmp ecx,[ui_param_count]
    jbe .sgr
    ret

; Zero-width codepoints: variation selectors, joiners, tags, skin tones and
; combining marks are not drawn.
ui_zero_width:
    xor eax,eax
    mov rdx,ui_invisible
.range:
    cmp ecx,[rdx]
    jb .next
    cmp ecx,[rdx+4]
    jbe .yes
.next:
    add rdx,8
    cmp rdx,ui_invisible_end
    jb .range
    ret
.yes:
    inc eax
    ret

; ecx = codepoint for the transcript. Markdown "**" toggles bright text on
; screen; COM1 keeps the raw text. A lone '*' is drawn as it is.
ui_codepoint:
    cmp ecx,'*'
    jne .other
    cmp byte [ui_star],0
    je .hold
    mov byte [ui_star],0
    xor byte [ui_bold],1
    ret
.hold:
    mov byte [ui_star],1
    ret
.other:
    cmp byte [ui_star],0
    je ui_put
    mov byte [ui_star],0
    cmp ecx,8
    je .done                        ; backspace removes the held '*'
    push rcx
    mov ecx,'*'
    call ui_put
    pop rcx
    jmp ui_put
.done:
    ret

; ecx = codepoint drawn in the transcript, wrapping lines at word boundaries.
ui_put:
    cmp ecx,10
    jne .not_newline
    mov byte [ui_bold],0
    jmp ui_newline
.not_newline:
    cmp ecx,8
    je .backspace
    cmp ecx,9
    je .tab
    cmp ecx,32
    jb .done
    cmp ecx,127
    je .done
    push rcx
    call ui_lookup
    pop rcx
    test eax,eax
    jnz .draw
    call ui_zero_width
    test eax,eax
    jnz .done
    mov ecx,0xfffd
    push rcx
    call ui_lookup
    pop rcx
.draw:
    GLYPH_WIDTH
    mov eax,[ui_x]
    add eax,edx
    cmp eax,[ui_right]
    jbe .fits
    push rcx
    push rdx
    call ui_wrap
    pop rdx
    pop rcx
.fits:
    mov r9d,[ui_fg]
    cmp byte [ui_bold],0
    je .colour
    mov r9d,UI_BRIGHT
.colour:
    call ui_draw
    cmp ecx,' '
    jne .word
    mov dword [ui_word_count],0
    ret
.word:
    mov eax,[ui_word_count]
    cmp eax,UI_WORD
    jae .long
    mov [ui_word+rax*8],ecx
    mov [ui_word+rax*8+4],r9d
.long:
    inc dword [ui_word_count]
.done:
    ret
.tab:
    mov ecx,' '
    call ui_put
    mov eax,[ui_x]
    sub eax,UI_MARGIN
    test eax,31
    jnz .tab
    ret
.backspace:
    mov eax,[ui_line_count]
    test eax,eax
    jz .done
    dec eax
    mov [ui_line_count],eax
    cmp dword [ui_word_count],0
    je .erase
    dec dword [ui_word_count]
.erase:
    movzx edx,byte [ui_line_widths+rax]
    sub [ui_x],edx
    mov esi,[ui_x]
    mov edi,[ui_y]
    mov ecx,UI_LINE
    mov eax,UI_PAGE
    jmp ui_fill

; ecx = codepoint, r9d = colour: draw at the cursor, copy it to the screen
; and advance. Preserves ecx and r9d.
ui_draw:
    push rcx
    push r9
    mov esi,[ui_x]
    mov edi,[ui_y]
    add edi,2
    mov r10d,UI_PAGE
    call ui_glyph
    mov ebx,eax
    mov esi,[ui_x]
    mov edi,[ui_y]
    add edi,2
    mov edx,ebx
    mov ecx,16
    call ui_blit
    add [ui_x],ebx
    mov eax,[ui_line_count]
    cmp eax,UI_LINE_GLYPHS
    jae .done
    mov [ui_line_widths+rax],bl
    inc dword [ui_line_count]
.done:
    pop r9
    pop rcx
    ret

; The glyph does not fit: move the word in progress to a new line when it is
; shorter than the line, otherwise break the line here.
ui_wrap:
    mov eax,[ui_word_count]
    test eax,eax
    jz ui_newline
    cmp eax,UI_WORD
    jae ui_newline
    cmp eax,[ui_line_count]
    jae ui_newline
    ; Erase the word's glyphs at the end of the line.
    mov ecx,[ui_line_count]
    sub ecx,eax
    xor edx,edx
.width:
    movzx r8d,byte [ui_line_widths+rcx]
    add edx,r8d
    inc ecx
    cmp ecx,[ui_line_count]
    jb .width
    sub [ui_x],edx
    mov esi,[ui_x]
    mov edi,[ui_y]
    mov ecx,UI_LINE
    mov eax,UI_PAGE
    call ui_fill
    ; Copy the word, start a new line and draw it again in its colours.
    mov ecx,[ui_word_count]
    mov [ui_moved_count],ecx
    mov esi,ui_word
    mov edi,ui_moved
    shl ecx,1
    rep movsd
    call ui_newline
    xor eax,eax
.redraw:
    cmp eax,[ui_moved_count]
    jae .done
    push rax
    mov ecx,[ui_moved+rax*8]
    mov r9d,[ui_moved+rax*8+4]
    call ui_draw
    pop rax
    mov edx,[ui_moved+rax*8]
    mov [ui_word+rax*8],edx
    mov edx,[ui_moved+rax*8+4]
    mov [ui_word+rax*8+4],edx
    inc eax
    mov [ui_word_count],eax
    jmp .redraw
.done:
    ret

ui_newline:
    mov dword [ui_word_count],0
    mov dword [ui_x],UI_MARGIN
    mov dword [ui_line_count],0
    mov eax,[ui_y]
    add eax,UI_LINE
    lea edx,[rax+UI_LINE]
    cmp edx,[ui_bottom]
    ja .scroll
    mov [ui_y],eax
    ret
.scroll:
    ; Move the transcript up one line in the shadow, clear the last line and
    ; copy the transcript to the framebuffer.
    mov eax,[ui_top]
    imul eax,[ui_width]
    lea rdi,[ui_shadow+rax*4]
    mov eax,[ui_width]
    imul eax,UI_LINE
    lea rsi,[rdi+rax*4]
    mov ecx,[ui_y]
    sub ecx,[ui_top]
    imul ecx,[ui_width]
    rep movsd
    xor esi,esi
    mov edi,[ui_y]
    mov edx,[ui_width]
    mov ecx,UI_LINE
    mov eax,UI_PAGE
    call ui_fill
    xor esi,esi
    mov edi,[ui_top]
    mov edx,[ui_width]
    mov ecx,[ui_y]
    add ecx,UI_LINE
    sub ecx,edi
    jmp ui_blit

; ecx = codepoint for the 80x25 VGA text fallback. Colours map to attributes;
; codepoints outside ASCII show as '?'.
vga_codepoint:
    mov eax,[ui_fg]
    mov dl,0x07
    cmp eax,UI_MUTED
    jne .accent
    mov dl,0x08
.accent:
    cmp eax,UI_ACCENT
    jne .error
    mov dl,0x0e
.error:
    cmp eax,UI_ERROR
    jne .bright
    mov dl,0x0c
.bright:
    cmp eax,UI_BRIGHT
    jne .chosen
    mov dl,0x0f
.chosen:
    cmp ecx,10
    je .newline
    cmp ecx,8
    je .backspace
    cmp ecx,32
    jb .done
    call ui_zero_width
    test eax,eax
    jnz .done
    cmp ecx,127
    jb .ascii
    mov ecx,'?'
.ascii:
    mov eax,[ui_vga_y]
    imul eax,80
    add eax,[ui_vga_x]
    mov [0xb8000+rax*2],cl
    mov [0xb8000+rax*2+1],dl
    inc dword [ui_vga_x]
    cmp dword [ui_vga_x],80
    jb .done
.newline:
    mov dword [ui_vga_x],0
    inc dword [ui_vga_y]
    cmp dword [ui_vga_y],25
    jb .done
    mov dword [ui_vga_y],24
    mov esi,0xb8000+160
    mov edi,0xb8000
    mov ecx,24*80/2
    rep movsd
    mov edi,0xb8000+24*160
    mov eax,0x07200720
    mov ecx,40
    rep stosd
.done:
    ret
.backspace:
    cmp dword [ui_vga_x],0
    je .done
    dec dword [ui_vga_x]
    mov eax,[ui_vga_y]
    imul eax,80
    add eax,[ui_vga_x]
    mov word [0xb8000+rax*2],0x0720
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

; Nonblocking keyboard input as UTF-8 bytes; zero means no byte is ready.
; Set 1 scancodes go through the selected layout: Shift, AltGr, Caps Lock,
; Ctrl with letters, and dead keys composed with the next key.
keyboard_poll:
    mov ecx,[keyboard_queued]
    cmp ecx,[keyboard_taken]
    jne .queued
    cmp byte [keyboard_present],0
    je .none
    in al,0x64
    test al,1
    jz .none
    test al,0x20
    jnz .mouse
    in al,0x60
    movzx eax,al
    cmp eax,0xe0
    je .extended
    movzx edx,byte [keyboard_extended]
    mov byte [keyboard_extended],0
    mov ecx,eax
    and ecx,0x7f
    cmp ecx,0x2a
    je .left_shift
    cmp ecx,0x36
    je .right_shift
    cmp ecx,0x1d
    je .ctrl
    cmp ecx,0x38
    je .alt
    test eax,0x80
    jnz .none
    cmp eax,0x3a
    je .caps
    test edx,edx
    jz .plain
    cmp eax,0x1c                ; keypad Enter
    jne .none
.plain:
    mov edx,27
    cmp eax,0x01
    je .byte
    mov edx,8
    cmp eax,0x0e
    je .byte
    mov edx,9
    cmp eax,0x0f
    je .byte
    mov edx,13
    cmp eax,0x1c
    je .byte
    ; Level: Shift adds 1, AltGr adds 2; Caps Lock flips Shift for letters.
    movzx ecx,byte [keyboard_layout]
    imul ecx,KEYMAP_BYTES
    lea rsi,[keymaps+rcx]
    xor ecx,ecx
    cmp byte [keyboard_shift],0
    setne cl
    cmp byte [keyboard_caps],0
    je .altgr
    cmp byte [rsi+1024+rax],0
    je .altgr
    xor ecx,1
.altgr:
    cmp byte [keyboard_altgr],0
    je .level
    or ecx,2
.level:
    lea edx,[rax*4+rcx]
    movzx edx,word [rsi+rdx*2]
    test edx,edx
    jz .none
    cmp byte [keyboard_ctrl],0
    je .dead
    or edx,0x20
    cmp edx,'a'
    jb .none
    cmp edx,'z'
    ja .none
    and edx,0x1f
    jmp .byte
.dead:
    cmp edx,0xf000
    jb .compose
    and edx,0xff
    inc edx
    mov [keyboard_dead],dl
    jmp .none
.compose:
    movzx eax,byte [keyboard_dead]
    test eax,eax
    jz .emit
    mov byte [keyboard_dead],0
    dec eax
    mov rsi,dead_pairs
    mov ecx,DEAD_PAIRS
.pair:
    cmp [rsi],ax
    jne .other
    cmp [rsi+2],dx
    jne .other
    movzx edx,word [rsi+4]
    jmp .emit
.other:
    add rsi,6
    loop .pair
    ; No composition: the spacing accent, then the key itself.
    push rdx
    movzx edx,word [dead_spacing+rax*2]
    call keyboard_queue
    pop rdx
.emit:
    call keyboard_queue
    jmp keyboard_poll
.byte:
    mov eax,edx
    ret
.queued:
    mov ecx,[keyboard_taken]
    and ecx,15
    movzx eax,byte [keyboard_bytes+rcx]
    inc dword [keyboard_taken]
    ret
.mouse:
    in al,0x60
    jmp .none
.extended:
    mov byte [keyboard_extended],1
    jmp .none
.left_shift:
    mov dl,1
    jmp .shift
.right_shift:
    mov dl,2
.shift:
    test eax,0x80
    jnz .shift_release
    or [keyboard_shift],dl
    jmp .none
.shift_release:
    not dl
    and [keyboard_shift],dl
    jmp .none
.ctrl:
    test eax,0x80
    setz [keyboard_ctrl]
    jmp .none
.alt:
    test edx,edx                ; left Alt is ignored, right Alt is AltGr
    jz .none
    test eax,0x80
    setz [keyboard_altgr]
    jmp .none
.caps:
    xor byte [keyboard_caps],1
.none:
    xor eax,eax
    ret

; edx = codepoint: append its UTF-8 bytes to the keyboard queue.
keyboard_queue:
    mov ecx,[keyboard_queued]
    cmp edx,0x80
    jb .one
    cmp edx,0x800
    jb .two
    mov eax,edx
    shr eax,12
    or eax,0xe0
    call .put
    mov eax,edx
    shr eax,6
    and eax,0x3f
    or eax,0x80
    call .put
    jmp .last
.two:
    mov eax,edx
    shr eax,6
    or eax,0xc0
    call .put
.last:
    mov eax,edx
    and eax,0x3f
    or eax,0x80
    jmp .put
.one:
    mov eax,edx
.put:
    mov ecx,[keyboard_queued]
    and ecx,15
    mov [keyboard_bytes+rcx],al
    inc dword [keyboard_queued]
    ret

; rsi = layout name. Selects it and returns eax = 1, or 0 when unknown.
keyboard_select:
    xor ecx,ecx
.layout:
    lea rdi,[keymap_names+rcx*5]
    xor edx,edx
.char:
    mov al,[rdi+rdx]
    cmp al,[rsi+rdx]
    jne .next
    inc edx
    test al,al
    jnz .char
    mov [keyboard_layout],cl
    mov eax,1
    ret
.next:
    inc ecx
    cmp ecx,KEYMAP_COUNT
    jb .layout
    xor eax,eax
    ret

section .rodata
ui_title: db "Pithos CORE",0
ui_subtitle: db OS_SUBTITLE,0
ui_starting: db "Starting",0xe2,0x80,0xa6,0
ui_hint: db "/help",0
align 4
ui_invisible:
    dd 0x0300,0x036f, 0x200b,0x200f, 0x2028,0x202e, 0x2060,0x206f, 0x20d0,0x20ff
    dd 0xfe00,0xfe0f, 0xfeff,0xfeff, 0x1f3fb,0x1f3ff, 0xe0000,0xe007f
ui_invisible_end:
align 16
ui_font: incbin "pithos-font.bin"
section .bss
video_kind: resb 1
keyboard_present: resb 1
keyboard_extended: resb 1
keyboard_shift: resb 1
keyboard_ctrl: resb 1
keyboard_altgr: resb 1
keyboard_caps: resb 1
keyboard_dead: resb 1
keyboard_layout: resb 1
ui_escape: resb 1
ui_star: resb 1
ui_bold: resb 1
ui_red_shift: resb 1
ui_green_shift: resb 1
ui_blue_shift: resb 1
alignb 8
ui_base: resq 1
ui_pitch: resd 1
ui_width: resd 1
ui_height: resd 1
ui_top: resd 1
ui_bottom: resd 1
ui_right: resd 1
ui_x: resd 1
ui_y: resd 1
ui_fg: resd 1
ui_fg_pixel: resd 1
ui_glyph_width: resd 1
ui_pending: resd 1
ui_unicode: resd 1
ui_param_count: resd 1
ui_params: resd 8
ui_vga_x: resd 1
ui_vga_y: resd 1
ui_line_count: resd 1
ui_line_widths: resb UI_LINE_GLYPHS
ui_word_count: resd 1
ui_moved_count: resd 1
alignb 8
ui_word: resq UI_WORD               ; codepoint and colour per glyph
ui_moved: resq UI_WORD
keyboard_queued: resd 1
keyboard_taken: resd 1
keyboard_bytes: resb 16
alignb 64
ui_shadow: resd UI_MAX_WIDTH*UI_MAX_HEIGHT
section .text
