; LFM3 engine options shared by Linux and Pithos CORE: sampling, reasoning,
; a system prompt, statistics and the chat commands. Greedy decoding stays
; the default and uses the verified argmax path, so default outputs do not
; change. Functions preserve rbx, rbp and r12-r15.
%define RECENT 64                   ; tokens considered by the repetition penalty
%define CANDIDATES 128              ; top-k ceiling
%define SYSTEM_BYTES 1024
%ifdef CHAT_STYLE
%define DIM 27,"[2m"
%define PLAIN 27,"[0m"
%else
%define DIM ""
%define PLAIN ""
%endif
section .text

; Returns eax = the next token from logits.
choose_token:
    call apply_repeat_penalty
    movss xmm0,[sample_temperature]
    xorps xmm1,xmm1
    ucomiss xmm0,xmm1
    ja sample_token
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
    ret

; eax = token: remember it for the repetition penalty.
remember_token:
    mov ecx,[recent_count]
    and ecx,RECENT-1
    mov [recent_ids+rcx*4],eax
    inc dword [recent_count]
    ret

; llama.cpp's penalty: each distinct recent token's logit is divided by the
; penalty when positive and multiplied when negative.
apply_repeat_penalty:
    movss xmm2,[sample_repeat]
    ucomiss xmm2,[one]
    je .done
    mov r8d,[recent_count]
    mov eax,RECENT
    cmp r8d,eax
    cmova r8d,eax
    xor ecx,ecx
.token:
    cmp ecx,r8d
    jae .done
    mov eax,[recent_ids+rcx*4]
    xor edx,edx
.seen:
    cmp edx,ecx
    jae .apply
    cmp [recent_ids+rdx*4],eax
    je .skip
    inc edx
    jmp .seen
.apply:
    movss xmm0,[logits+rax*4]
    xorps xmm1,xmm1
    ucomiss xmm0,xmm1
    jbe .negative
    divss xmm0,xmm2
    jmp .store
.negative:
    mulss xmm0,xmm2
.store:
    movss [logits+rax*4],xmm0
.skip:
    inc ecx
    jmp .token
.done:
    ret

; Top-k candidates sorted by logit, softmax with temperature, min-p and top-p
; filters, then a draw from the seeded xorshift64* generator.
sample_token:
    push rbx
    mov r8d,[sample_top_k]
    mov eax,CANDIDATES
    test r8d,r8d
    cmovz r8d,eax
    cmp r8d,eax
    cmova r8d,eax
    xor r9d,r9d                     ; candidates kept
    xor ecx,ecx
.scan:
    movss xmm0,[logits+rcx*4]
    cmp r9d,r8d
    jb .insert
    ucomiss xmm0,[candidate_logits+r8*4-4]
    jbe .scanned
    lea edx,[r8-1]
    jmp .shift
.insert:
    mov edx,r9d
    inc r9d
.shift:
    test edx,edx
    jz .place
    movss xmm1,[candidate_logits+rdx*4-4]
    ucomiss xmm0,xmm1
    jbe .place
    movss [candidate_logits+rdx*4],xmm1
    mov eax,[candidate_ids+rdx*4-4]
    mov [candidate_ids+rdx*4],eax
    dec edx
    jmp .shift
.place:
    movss [candidate_logits+rdx*4],xmm0
    mov [candidate_ids+rdx*4],ecx
.scanned:
    inc ecx
    cmp ecx,128000
    jb .scan
    ; p_j = exp((l_j - l_0) / T); p_0 = 1.
    xor ebx,ebx
    xorps xmm4,xmm4                 ; total
.probability:
    movss xmm0,[candidate_logits+rbx*4]
    subss xmm0,[candidate_logits]
    divss xmm0,[sample_temperature]
    call exp_scalar
    movss [candidate_probs+rbx*4],xmm0
    ; min-p: stop at the first candidate below min_p * p_0.
    test ebx,ebx
    jz .keep
    ucomiss xmm0,[sample_min_p]
    jb .filtered
.keep:
    addss xmm4,xmm0
    inc ebx
    cmp ebx,r9d
    jb .probability
.filtered:
    ; top-p: the smallest prefix whose mass reaches top_p of the kept mass.
    movss xmm5,xmm4
    mulss xmm5,[sample_top_p]
    xorps xmm6,xmm6
    xor ecx,ecx
.nucleus:
    addss xmm6,[candidate_probs+rcx*4]
    inc ecx
    cmp ecx,ebx
    jae .draw
    ucomiss xmm6,xmm5
    jb .nucleus
.draw:
    mov ebx,ecx                     ; candidates in the nucleus, mass xmm6
    call random_unit
    mulss xmm0,xmm6
    xorps xmm1,xmm1
    xor ecx,ecx
.walk:
    addss xmm1,[candidate_probs+rcx*4]
    ucomiss xmm1,xmm0
    ja .chosen
    inc ecx
    cmp ecx,ebx
    jb .walk
    dec ecx
.chosen:
    mov eax,[candidate_ids+rcx*4]
    pop rbx
    ret

; xmm0 = uniform in [0, 1) from xorshift64*.
random_unit:
    mov rax,[random_state]
    mov rdx,rax
    shr rdx,12
    xor rax,rdx
    mov rdx,rax
    shl rdx,25
    xor rax,rdx
    mov rdx,rax
    shr rdx,27
    xor rax,rdx
    mov [random_state],rax
    mov rdx,0x2545f4914f6cdd1d
    imul rax,rdx
    shr rax,40                      ; 24 random bits
    cvtsi2ss xmm0,eax
    mulss xmm0,[two_minus24]
    ret

seed_random:
    mov eax,[sample_seed]
    mov rdx,0x9e3779b97f4a7c15
    xor rax,rdx
    mov [random_state],rax
    ret

; Microseconds from a monotonic clock, or 0 when none is calibrated.
engine_ticks:
%ifdef BARE_METAL
    mov rcx,[tsc_per_ms]
    test rcx,rcx
    jz .none
    rdtsc
    shl rdx,32
    or rax,rdx
    mov edx,1000
    mul rdx
    div rcx
    ret
.none:
    xor eax,eax
    ret
%else
    mov eax,228                     ; clock_gettime(CLOCK_MONOTONIC)
    mov edi,1
    mov esi,clock_buffer
    syscall
    imul rax,[clock_buffer],1000000
    mov rcx,rax
    mov rax,[clock_buffer+8]
    xor edx,edx
    mov r8d,1000
    div r8
    add rax,rcx
    ret
%endif

; Returns eax = 1 when Escape was pressed during generation. Other keys
; typed meanwhile are discarded.
engine_interrupted:
%ifdef BARE_METAL
.poll:
    call keyboard_poll
    test al,al
    jz .serial
    cmp al,27
    je .yes
    jmp .poll
.serial:
    cmp byte [serial_present],0
    je .no
    mov dx,0x3fd
    in al,dx
    test al,1
    jz .no
    mov dx,0x3f8
    in al,dx
    cmp al,27
    je .yes
    jmp .serial
.yes:
    mov eax,1
    ret
.no:
%endif
    xor eax,eax
    ret

; Once per chat session: the generator, the clock and the status line.
engine_ready:
    call seed_random
%ifdef BARE_METAL
    cmp qword [tsc_per_ms],0
    jne .status
    call calibrate_tsc
.status:
%endif
    mov dword [stat_generated],0
    jmp engine_show_status

; rbp = prompt tokens of this turn, r14 = cache position.
engine_turn_start:
    mov byte [turn_over],0
    mov byte [generating],1
    call engine_ticks
    mov [stat_start],rax
    mov [stat_first],rax
    mov [stat_prompt],ebp
    mov dword [stat_generated],0
    mov byte [stat_stopped],0
    xor ecx,ecx
.recent:
    cmp ecx,ebp
    jae .think
    mov eax,[prompt_ids+rcx*4]
    push rcx
    call remember_token
    pop rcx
    inc ecx
    jmp .recent
.think:
    mov al,[think_mode]
    mov [in_think],al
    mov dword [think_tokens],0
    test al,al
    jz .done
    cmp byte [text_mode],0
    je .done
    cmp byte [think_show],0
    je .done
%ifdef CHAT_STYLE
    mov esi,dim_on
    mov edx,dim_on_len
    mov edi,1
    call write_all
%endif
.done:
    ret

; eax = generated token. Returns eax = 1 when the user stopped the answer.
engine_progress:
    push rax
    call remember_token
    inc dword [stat_generated]
    cmp dword [stat_generated],1
    jne .later
    call engine_ticks
    mov [stat_first],rax
.later:
    call engine_ticks
    mov [stat_end],rax
    test dword [stat_generated],7
    jnz .interrupt
    call engine_show_status
.interrupt:
    pop rax
    cmp byte [turn_over],0
    je .escape
    mov byte [turn_over],0
    mov eax,1
    ret
.escape:
    cmp byte [chat_mode],0
    je .continue
    call engine_interrupted
    test eax,eax
    jz .continue
    mov byte [stat_stopped],1
    mov esi,stopped_notice
    mov edx,stopped_notice_len
    mov edi,1
    call write_all
    mov eax,1
    ret
.continue:
    xor eax,eax
    ret

engine_turn_end:
    call engine_ticks
    mov [stat_end],rax
    call engine_show_status         ; still at the turn's cache position
    mov byte [generating],0
    ret

; eax = token in text mode. Reasoning is shown dim, or collapsed into a count.
engine_emit:
    cmp eax,THINK_OPEN
    je .open
    cmp eax,THINK_CLOSE
    je .close
    cmp byte [in_think],0
    je decode_token
    inc dword [think_tokens]
    cmp byte [think_show],0
    jne decode_token
    ret
.open:
    mov byte [in_think],1
    mov dword [think_tokens],0
%ifdef CHAT_STYLE
    cmp byte [think_show],0
    je .done
    mov esi,dim_on
    mov edx,dim_on_len
    jmp .write
%else
    ret
%endif
.close:
    cmp byte [in_think],0
    jne .closed
    ; A closing tag outside reasoning: the model has answered and would
    ; repeat the answer. End the turn here as if it had written EOS.
    mov dword [next_token],EOS_ID
    mov byte [turn_over],1
    ret
.closed:
    mov byte [in_think],0
    cmp byte [think_show],0
    je .collapsed
    mov esi,dim_off
    mov edx,dim_off_len
    jmp .write
.collapsed:
    mov edi,message_buffer
    mov esi,reasoned_text
    call put_text
    mov eax,[think_tokens]
    call put_uint
    mov esi,reasoned_tail
    call put_text
    mov esi,message_buffer
    mov rdx,rdi
    sub rdx,rsi
.write:
    mov edi,1
    jmp write_all
.done:
    ret

; Text builders: rdi = cursor, return rdi advanced.
put_text:
    mov al,[rsi]
    test al,al
    jz .done
    mov [rdi],al
    inc rsi
    inc rdi
    jmp put_text
.done:
    ret

; eax = unsigned value.
put_uint:
    mov ecx,10
    lea r8,[number_buffer+12]
    mov byte [r8],0
.digit:
    xor edx,edx
    div ecx
    add dl,'0'
    dec r8
    mov [r8],dl
    test eax,eax
    jnz .digit
    mov rsi,r8
    jmp put_text

; eax = value in hundredths; ecx = decimals shown (1 or 2), rounded.
put_fixed:
    cmp ecx,1
    je .one
    xor edx,edx
    mov r9d,100
    div r9d
    push rdx
    call put_uint
    pop rax
    mov byte [rdi],'.'
    inc rdi
    jmp .two
.one:
    add eax,5
    xor edx,edx
    mov r9d,100
    div r9d                         ; eax units, edx hundredths + 5
    push rdx
    call put_uint
    pop rax
    mov byte [rdi],'.'
    xor edx,edx
    mov ecx,10
    div ecx
    add al,'0'
    mov [rdi+1],al
    add rdi,2
    ret
.two:
    xor edx,edx
    mov ecx,10
    div ecx
    add al,'0'
    mov [rdi],al
    add dl,'0'
    mov [rdi+1],dl
    add rdi,2
    ret

; xmm0 = float: rdi += its value with two decimals.
put_float:
    mulss xmm0,[hundred]
    cvtss2si eax,xmm0
    mov ecx,2
    jmp put_fixed

; rax = count, rcx = microseconds: rdi += count per second with one decimal.
put_rate:
    test rcx,rcx
    jz .unknown
    imul rax,rax,100000000
    xor edx,edx
    div rcx
    mov ecx,1
    jmp put_fixed
.unknown:
    mov esi,unknown_text
    jmp put_text

; Fill status_buffer with the speeds, the context and the settings, and show
; it on the Pithos CORE status line.
engine_status_text:
    mov edi,status_buffer
    cmp dword [stat_generated],0
    je .settings
    mov eax,[stat_generated]
    dec eax
    mov rcx,[stat_end]
    sub rcx,[stat_first]
    call put_rate
    mov esi,tokens_per_second
    call put_text
    mov eax,[stat_prompt]
    mov rcx,[stat_first]
    sub rcx,[stat_start]
    call put_rate
    mov esi,prompt_rate_text
    call put_text
    mov rax,[stat_first]
    sub rax,[stat_start]
    xor edx,edx
    mov ecx,10000
    div rcx
    mov ecx,2
    call put_fixed
    mov esi,first_token_text
    call put_text
.settings:
    mov esi,context_text
    call put_text
    mov eax,[cache_position]
    cmp byte [generating],0
    je .position
    mov eax,r14d                    ; inside a turn, r14 is the cache position
.position:
    call put_uint
    mov byte [rdi],'/'
    inc rdi
    mov eax,[context_limit]
    call put_uint
    mov esi,separator
    call put_text
    call put_sampling_short
    mov esi,think_off_text
    cmp byte [think_mode],0
    je .think
    mov esi,think_on_text
.think:
    call put_text
    mov esi,separator
    call put_text
    mov eax,[thread_count]
    call put_uint
    mov esi,cores_text
    call put_text
%ifdef PITHOS_UI
    mov esi,separator
    call put_text
    movzx eax,byte [keyboard_layout]
    lea rsi,[keymap_names+rax*5]
    call put_text
%endif
    mov byte [rdi],0
    ret

engine_show_status:
    call engine_status_text
%ifdef PITHOS_UI
    mov esi,status_buffer
    call ui_status
%endif
    ret

; The status line names only the decoding and its temperature; /set lists all.
put_sampling_short:
    movss xmm0,[sample_temperature]
    xorps xmm1,xmm1
    ucomiss xmm0,xmm1
    ja .sampled
    mov esi,greedy_text
    call put_text
    jmp .done
.sampled:
    mov esi,temperature_text
    call put_text
    movss xmm0,[sample_temperature]
    call put_float
.done:
    mov esi,separator
    jmp put_text

put_sampling:
    movss xmm0,[sample_temperature]
    xorps xmm1,xmm1
    ucomiss xmm0,xmm1
    ja .sampled
    mov esi,greedy_text
    call put_text
    jmp .repeat
.sampled:
    mov esi,temperature_text
    call put_text
    movss xmm0,[sample_temperature]
    call put_float
    mov esi,top_k_text
    call put_text
    mov eax,[sample_top_k]
    call put_uint
    mov esi,top_p_text
    call put_text
    movss xmm0,[sample_top_p]
    call put_float
    mov esi,min_p_text
    call put_text
    movss xmm0,[sample_min_p]
    call put_float
.repeat:
    movss xmm0,[sample_repeat]
    ucomiss xmm0,[one]
    je .done
    mov esi,repeat_text
    call put_text
    movss xmm0,[sample_repeat]
    call put_float
.done:
    mov esi,separator
    jmp put_text

; rsi = text. Returns rsi at the first non-space.
skip_spaces:
    cmp byte [rsi],' '
    jne .done
    inc rsi
    jmp skip_spaces
.done:
    ret

; rsi = text, rdi = word. When the text starts with the word followed by a
; space or its end, returns eax = 1 and rsi after the word.
take_word:
    xor ecx,ecx
.char:
    mov al,[rdi+rcx]
    test al,al
    jz .end
    cmp al,[rsi+rcx]
    jne .no
    inc ecx
    jmp .char
.end:
    mov al,[rsi+rcx]
    test al,al
    jz .yes
    cmp al,' '
    jne .no
.yes:
    add rsi,rcx
    call skip_spaces
    mov eax,1
    ret
.no:
    xor eax,eax
    ret

; rsi = decimal text, digits with an optional fraction. Returns xmm0 and
; CF clear, or CF set when the word is malformed.
parse_decimal:
    xor eax,eax
    xor ecx,ecx                     ; digits seen
    mov r8d,1                       ; fraction scale
    xor r9d,r9d                     ; in the fraction
.digit:
    movzx edx,byte [rsi]
    cmp edx,'.'
    je .point
    sub edx,'0'
    cmp edx,9
    ja .end
    cmp ecx,8
    jae .bad
    imul eax,10
    add eax,edx
    inc ecx
    test r9d,r9d
    jz .next
    imul r8d,10
.next:
    inc rsi
    jmp .digit
.point:
    test r9d,r9d
    jnz .bad
    mov r9d,1
    inc rsi
    jmp .digit
.end:
    test ecx,ecx
    jz .bad
    cmp byte [rsi],0
    jne .bad
    cvtsi2ss xmm0,eax
    cvtsi2ss xmm1,r8d
    divss xmm0,xmm1
    clc
    ret
.bad:
    stc
    ret

; chat_line holds a command. Returns eax = 0 for an unknown command, 1 when
; handled, 2 when handled and the conversation must restart.
extra_command:
    push rbx
    call command_dispatch
    pop rbx
    ret

command_dispatch:
    mov esi,chat_line
    mov edi,help_command
    call take_word
    test eax,eax
    jnz .help
    mov esi,chat_line
    mov edi,set_command
    call take_word
    test eax,eax
    jnz .set
    mov esi,chat_line
    mov edi,think_command
    call take_word
    test eax,eax
    jnz .think
    mov esi,chat_line
    mov edi,system_command
    call take_word
    test eax,eax
    jnz .system
    mov esi,chat_line
    mov edi,stats_command
    call take_word
    test eax,eax
    jnz .stats
    mov esi,chat_line
    mov edi,reset_command
    call take_word
    test eax,eax
    jnz .reset
%ifdef PITHOS_UI
    mov esi,chat_line
    mov edi,keymap_command
    call take_word
    test eax,eax
    jnz .keymap
%endif
    xor eax,eax
    ret
.help:
    mov esi,help_text
    mov edx,help_text_len
    jmp .write
.stats:
    call engine_status_text
    mov esi,status_buffer
    jmp .line
.set:
    cmp byte [rsi],0
    je .show
    mov rbx,rsi
    mov edi,temperature_word
    call take_word
    test eax,eax
    jnz .temperature
    mov rsi,rbx
    mov edi,top_k_word
    call take_word
    test eax,eax
    jnz .top_k
    mov rsi,rbx
    mov edi,top_p_word
    call take_word
    test eax,eax
    jnz .top_p
    mov rsi,rbx
    mov edi,min_p_word
    call take_word
    test eax,eax
    jnz .min_p
    mov rsi,rbx
    mov edi,repeat_word
    call take_word
    test eax,eax
    jnz .repeat
    mov rsi,rbx
    mov edi,seed_word
    call take_word
    test eax,eax
    jnz .seed
    mov rsi,rbx
    mov edi,tokens_word
    call take_word
    test eax,eax
    jnz .tokens
    jmp .invalid
.temperature:
    call parse_decimal
    jc .invalid
    ucomiss xmm0,[five]
    ja .invalid
    movss [sample_temperature],xmm0
    jmp .show
.top_k:
    call parse_decimal
    jc .invalid
    cvttss2si eax,xmm0
    cmp eax,CANDIDATES
    ja .invalid
    mov [sample_top_k],eax
    jmp .show
.top_p:
    call parse_decimal
    jc .invalid
    xorps xmm1,xmm1
    ucomiss xmm0,xmm1
    jbe .invalid
    ucomiss xmm0,[one]
    ja .invalid
    movss [sample_top_p],xmm0
    jmp .show
.min_p:
    call parse_decimal
    jc .invalid
    ucomiss xmm0,[one]
    ja .invalid
    movss [sample_min_p],xmm0
    jmp .show
.repeat:
    call parse_decimal
    jc .invalid
    ucomiss xmm0,[one]
    jb .invalid
    ucomiss xmm0,[two]
    ja .invalid
    movss [sample_repeat],xmm0
    jmp .show
.seed:
    mov ecx,0x7fffffff
    call parse_uint_soft
    jc .invalid
    mov [sample_seed],eax
    call seed_random
    jmp .show
.tokens:
    mov ecx,[context_limit]
    call parse_uint_soft
    jc .invalid
    test eax,eax
    jz .invalid
    mov [max_new_tokens],eax
.show:
    call engine_show_status
    mov edi,status_buffer
    call put_sampling
    mov esi,tokens_limit_text
    call put_text
    mov eax,[max_new_tokens]
    call put_uint
    mov byte [rdi],0
    mov esi,status_buffer
    jmp .line
.think:
    mov rbx,rsi
    mov edi,on_word
    call take_word
    test eax,eax
    jnz .think_on
    mov rsi,rbx
    mov edi,off_word
    call take_word
    test eax,eax
    jnz .think_off
    mov rsi,rbx
    mov edi,show_word
    call take_word
    test eax,eax
    jnz .think_show
    mov rsi,rbx
    mov edi,hide_word
    call take_word
    test eax,eax
    jnz .think_hide
    cmp byte [rbx],0
    jne .invalid
    jmp .think_state
.think_on:
    mov byte [think_mode],1
    jmp .think_state
.think_off:
    mov byte [think_mode],0
    jmp .think_state
.think_show:
    mov byte [think_show],1
    jmp .think_state
.think_hide:
    mov byte [think_show],0
.think_state:
    call engine_show_status
    mov esi,think_off_text
    cmp byte [think_mode],0
    je .think_line
    mov esi,think_on_text
.think_line:
    mov edi,status_buffer
    call put_text
    mov esi,think_hidden_text
    cmp byte [think_show],0
    je .think_visibility
    mov esi,think_shown_text
.think_visibility:
    call put_text
    mov byte [rdi],0
    mov esi,status_buffer
    jmp .line
.system:
    mov edi,system_prompt
    mov ecx,SYSTEM_BYTES-1
.copy:
    mov al,[rsi]
    test al,al
    jz .copied
    dec ecx
    jz .invalid
    mov [rdi],al
    inc rsi
    inc rdi
    jmp .copy
.copied:
    mov byte [rdi],0
    mov esi,system_notice
    mov edx,system_notice_len
    call chat_write
    mov eax,2
    ret
.reset:
    mov dword [sample_temperature],0
    mov dword [sample_top_k],50
    mov eax,[one]
    mov [sample_top_p],eax
    mov [sample_repeat],eax
    mov dword [sample_min_p],0
    mov dword [sample_seed],42
    mov byte [think_mode],0
    mov byte [think_show],1
    call seed_random
    jmp .show
%ifdef PITHOS_UI
.keymap:
    cmp byte [rsi],0
    je .keymaps
    call keyboard_select
    test eax,eax
    jz .invalid
    call engine_show_status
.keymaps:
    mov edi,status_buffer
    mov esi,keymap_list_text
    call put_text
    xor ebx,ebx
.keymap_name:
    lea rsi,[keymap_names+rbx*5]
    call put_text
    mov byte [rdi],' '
    inc rdi
    inc ebx
    cmp ebx,KEYMAP_COUNT
    jb .keymap_name
    mov esi,keymap_current_text
    call put_text
    movzx eax,byte [keyboard_layout]
    lea rsi,[keymap_names+rax*5]
    call put_text
    mov byte [rdi],0
    mov esi,status_buffer
    jmp .line
%endif
.invalid:
    mov esi,invalid_notice
    mov edx,invalid_notice_len
.write:
    call chat_write
    mov eax,1
    ret
; rsi = zero-terminated line for stderr, dim on the console.
.line:
    mov rbx,rsi
%ifdef CHAT_STYLE
    mov esi,dim_on
    mov edx,dim_on_len
    call chat_write
%endif
    mov rsi,rbx
    xor edx,edx
.length:
    cmp byte [rsi+rdx],0
    je .print
    inc edx
    jmp .length
.print:
    call chat_write
    mov esi,line_end
    mov edx,line_end_len
    jmp .write

; rsi = digits, ecx = largest value. eax = value with CF clear, or CF set.
parse_uint_soft:
    xor eax,eax
    cmp byte [rsi],0
    je .bad
.digit:
    movzx edx,byte [rsi]
    sub edx,'0'
    cmp edx,9
    ja .bad
    imul eax,10
    add eax,edx
    cmp eax,ecx
    ja .bad
    inc rsi
    cmp byte [rsi],0
    jne .digit
    clc
    ret
.bad:
    stc
    ret

; rdi = text cursor: the conversation start (with the system prompt) or the
; next user turn.
engine_prefix:
    cmp dword [cache_position],0
    jne .user
    mov esi,template_start
    call copy_text
    cmp byte [system_prompt],0
    je .user
    mov esi,template_system
    call copy_text
    mov esi,system_prompt
    call copy_text
    mov esi,template_end
    call copy_text
.user:
    mov esi,chat_user_prefix
    jmp copy_text

; rdi = text cursor: close the user turn and open the assistant's, with an
; empty reasoning block unless reasoning is on.
engine_suffix:
    mov esi,template_assistant
    call copy_text
    mov esi,template_no_think
    cmp byte [think_mode],0
    je .copy
    mov esi,template_think
.copy:
    jmp copy_text

section .rodata
template_start: db "<|startoftext|>",0
template_system: db "<|im_start|>system",10,0
template_end: db "<|im_end|>",10,0
template_assistant: db "<|im_end|>",10,"<|im_start|>assistant",10,0
template_no_think: db "<think></think>",0
template_think: db "<think>",0
help_command: db "/help",0
set_command: db "/set",0
think_command: db "/think",0
system_command: db "/system",0
stats_command: db "/stats",0
reset_command: db "/reset",0
keymap_command: db "/keymap",0
temperature_word: db "temp",0
top_k_word: db "top-k",0
top_p_word: db "top-p",0
min_p_word: db "min-p",0
repeat_word: db "repeat",0
seed_word: db "seed",0
tokens_word: db "tokens",0
on_word: db "on",0
off_word: db "off",0
show_word: db "show",0
hide_word: db "hide",0
help_text:
    db DIM
    db "Commands",10
    db "  /set                 show sampling settings",10
    db "  /set temp 0.7        temperature; 0 is greedy (default)",10
    db "  /set top-k 50        top-p 0.95  min-p 0.05  repeat 1.1  seed 42  tokens 256",10
    db "  /think on|off        let the model reason before answering",10
    db "  /think show|hide     show the reasoning or only its length",10
    db "  /system TEXT         set the system prompt and restart (empty clears it)",10
    db "  /stats               speed, context and settings",10
%ifdef PITHOS_UI
    db "  /keymap it           keyboard layout: us uk it de fr es pt br ch se fi no dk",10
    db "  Esc                  stop the answer being written",10
%endif
    db "  /reset               default settings     /clear  new conversation     /quit",10
    db PLAIN
help_text_len equ $-help_text
dim_on: db DIM
dim_on_len equ $-dim_on
dim_off: db PLAIN,10
dim_off_len equ $-dim_off
line_end: db PLAIN,10
line_end_len equ $-line_end
stopped_notice: db 10,DIM,"Stopped.",PLAIN
stopped_notice_len equ $-stopped_notice
system_notice: db DIM,"System prompt set; conversation restarted.",PLAIN,10
system_notice_len equ $-system_notice
invalid_notice: db DIM,"Invalid value. /help lists the commands and ranges.",PLAIN,10
invalid_notice_len equ $-invalid_notice
reasoned_text: db DIM,0xe2,0x96,0xb8," reasoned for ",0
reasoned_tail: db " tokens",PLAIN,10,0
tokens_per_second: db " tok/s ",0xc2,0xb7," ",0
prompt_rate_text: db " tok/s prompt ",0xc2,0xb7," ",0
first_token_text: db " s first token ",0xc2,0xb7," ",0
context_text: db "ctx ",0
separator: db " ",0xc2,0xb7," ",0
greedy_text: db "greedy",0
temperature_text: db "temp ",0
top_k_text: db " top-k ",0
top_p_text: db " top-p ",0
min_p_text: db " min-p ",0
repeat_text: db " repeat ",0
tokens_limit_text: db "tokens ",0
think_on_text: db "think on",0
think_off_text: db "think off",0
think_shown_text: db ", reasoning shown",0
think_hidden_text: db ", reasoning collapsed",0
cores_text: db " cores",0
unknown_text: db 0xe2,0x80,0x93,0
keymap_list_text: db "Layouts: ",0
keymap_current_text: db 0xc2,0xb7," current ",0
align 4
hundred: dd 100.0
five: dd 5.0
two: dd 2.0
two_minus24: dd 0x33800000
section .data
sample_temperature: dd 0.0
sample_top_k: dd 50
sample_top_p: dd 1.0
sample_min_p: dd 0.0
sample_repeat: dd 1.0
sample_seed: dd 42
think_show: db 1
section .bss
think_mode: resb 1
in_think: resb 1
stat_stopped: resb 1
generating: resb 1
turn_over: resb 1
alignb 8
random_state: resq 1
clock_buffer: resq 2
stat_start: resq 1
stat_first: resq 1
stat_end: resq 1
stat_prompt: resd 1
stat_generated: resd 1
think_tokens: resd 1
recent_count: resd 1
recent_ids: resd RECENT
candidate_logits: resd CANDIDATES
candidate_ids: resd CANDIDATES
candidate_probs: resd CANDIDATES
system_prompt: resb SYSTEM_BYTES
status_buffer: resb 512
message_buffer: resb 128
number_buffer: resb 16
section .text
