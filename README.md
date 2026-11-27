# Pithos CORE

Inferenza di modelli LiquidAI in NASM x86-64, su Linux e come sistema operativo autonomo.

Pithos CORE esegue LiquidAI LFM3-1.7B, il modello principale, dal GGUF Q4_K_M, e LFM2.5-350M, il modello compatto, dai pesi BF16 ufficiali. Ogni modello ha un eseguibile Linux e un kernel autonomo con avvio BIOS/UEFI da CD o chiavetta USB, fino a 16 CPU, console grafica Unicode e porta seriale. Tokenizer BPE, prodotti AVX2/FMA, worker paralleli, prefill a blocchi, campionamento e comandi della chat sono condivisi. Il runtime non richiede libc o Python. Progetto indipendente, ispirato a llm.asm, non un fork.

## LFM3-1.7B: il modello principale

Il runtime quantizza ogni ingresso di matrice in Q8_K e calcola prodotti interi con le righe Q4_K e Q6_K del GGUF, come ggml. Gestisce sandwich norm, convoluzioni a sette coefficienti, attenzione con finestra di 256 posizioni, RoPE parziale e gate per testa. Le risposte coincidono con un riferimento FP32 costruito sul codice ufficiale `modeling_lfm3.py`; i dettagli sono in [lfm3/README.md](lfm3/README.md).

### Build e utilizzo

Servono NASM, GNU ld, GNU make 4.3 o successivo e Python 3 con il pacchetto tokenizers. Il GGUF e i file del tokenizer ufficiale si scaricano da Hugging Face.

~~~sh
hf download LiquidAI/LFM3-1.7B-GGUF LFM3-1.7B-Q4_K_M.gguf --local-dir weights
hf download LiquidAI/LFM3-1.7B tokenizer.json tokenizer_config.json chat_template.jinja --local-dir lfm3-ref
make lfm3 LFM3_MODEL=weights/LFM3-1.7B-Q4_K_M.gguf LFM3_TOKENIZER=lfm3-ref/tokenizer.json
./build/lfm3/lfm3-next weights/LFM3-1.7B-Q4_K_M.gguf --chat 256
./build/lfm3/lfm3-next weights/LFM3-1.7B-Q4_K_M.gguf --text 128 "Scrivi due frasi su Torino."
~~~

Le opzioni sono `--chat [N]`, `--text N "prompt"`, `--generate N ID ...`, `--encode` e `--decode`; `LFM_THREADS` sceglie da 1 a 16 thread e `LFM_CONTEXT` da 1 a 32.768 posizioni. Le tabelle `build/lfm3/tokenizer.bin` devono restare accanto all'eseguibile.

### Pithos CORE in QEMU, da CD o da USB

~~~sh
make lfm3-os-image LFM3_MODEL=weights/LFM3-1.7B-Q4_K_M.gguf LFM3_TOKENIZER=lfm3-ref/tokenizer.json OS_KEYMAP=it
python3 os/run.py --iso build/lfm3/pithos-core.iso --cpus 4 --display gtk
python3 os/run.py --iso build/lfm3/pithos-core.iso --uefi --usb --cpus 4 --display gtk
~~~

`build/lfm3/pithos-core.iso`, circa 1,13 GB, e un'immagine ibrida GRUB per BIOS e UEFI, scrivibile come immagine raw su una chiavetta. Contiene kernel, GGUF invariato e tabelle del tokenizer; dopo l'avvio il kernel lavora interamente in RAM. L'immagine apre la chat con 256 token per risposta, contesto 4.096 e fino a sei CPU; `OS_KEYMAP` sceglie il layout di tastiera predefinito. Servono 2 GiB di RAM sotto i 4 GiB e una CPU con AVX2, FMA e F16C; Secure Boot va disattivato.

### Console e comandi

La console mostra il marchio di Pithos, il modello, la conversazione con a capo per parola e una barra di stato con velocita di generazione e del prompt, tempo del primo token, contesto, decodifica, ragionamento, processori e layout. Cinese, giapponese, coreano ed emoji a colori sono visibili sullo schermo; la tastiera segue tredici layout (us, uk, it, de, fr, es, pt, br, ch, se, fi, no, dk) con AltGr, Caps Lock e tasti morti.

| Comando | Effetto |
| --- | --- |
| `/set temp 0.7` | campionamento con temperatura; 0 e la decodifica greedy predefinita |
| `/set top-k 50`, `top-p 0.95`, `min-p 0.05`, `repeat 1.1`, `seed 42`, `tokens 256` | filtri, penalita di ripetizione, seme e token per risposta |
| `/think on`, `/think off`, `/think show`, `/think hide` | ragionamento prima della risposta, visibile o riassunto |
| `/system TESTO` | system prompt; riavvia la conversazione |
| `/stats`, `/set`, `/reset`, `/help`, `/clear`, `/quit` | statistiche, impostazioni, valori predefiniti, aiuto, nuova conversazione, uscita |
| `/keymap it`, Esc | layout di tastiera; Esc cancella la riga o interrompe la risposta (solo nel kernel) |

### Prestazioni

Misure del 27 novembre 2026 su Intel Core i5-10500H, WSL2 con quattro CPU visibili, prompt di 465 token e 64 token generati.

| Thread | Primo token | Prompt | Generazione |
| --- | --- | --- | --- |
| 1 | 19,67 s | 24 token/s | 9,79 token/s |
| 4 | 7,95 s | 59 token/s | 17,77 token/s |

Sulla stessa macchina il motore C pithos, con quattro thread su Windows, ha elaborato lo stesso prompt a 56-62 token/s e generato a 19-21 token/s.

### Verifica

~~~sh
make verify-lfm3 verify-lfm3-tokenizer verify-lfm3-os LFM3_MODEL=weights/LFM3-1.7B-Q4_K_M.gguf LFM3_TOKENIZER=lfm3-ref/tokenizer.json LFM3_REF=lfm3-ref
~~~

Il tokenizer coincide con quello ufficiale su 62 testi e su tutti i 128.000 ID. Ogni blocco del modello riproduce il riferimento FP32 con gli stessi ingressi Q8_K, l'argmax finale coincide su prompt fino a 721 token e le risposte greedy sono l'argmax del riferimento salvo quasi pareggi. Logit e risposte sono identici bit per bit fra 1 e 16 thread, fra prefill a blocchi e token per token, e fra Linux e kernel con 1, 2, 3, 4 e 6 processori.

## LFM2.5-350M: il modello compatto

Inferenza per i pesi BF16 ufficiali di LiquidAI/LFM2.5-350M, con attivazioni e accumulazioni FP32, un percorso scalare per le CPU senza AVX2 e verifiche bit per bit contro Transformers. Linux carica i pesi con mmap e usa thread persistenti; il kernel riceve pesi e tokenizer dal bootloader e avvia i propri worker, con la stessa console di Pithos CORE.

### Build e utilizzo

Servono NASM, GNU ld, GNU make 4.3 o successivo e Python 3 con il pacchetto tokenizers per la build. La versione verificata di tokenizers e 0.23.3. Su Linux x86-64, con il client Hugging Face installato, dalla directory dei sorgenti si esegue

~~~sh
hf download LiquidAI/LFM2.5-350M --revision 9921cd4d82c0e3986887aa4b376f9446729ef8a3 --include model.safetensors config.json tokenizer.json tokenizer_config.json chat_template.jinja --local-dir weights
make MODEL=weights/model.safetensors
./build/lfm-next weights/model.safetensors 1 314 271
./build/lfm-next weights/model.safetensors --text 64 "Scrivi un saluto in italiano."
./build/lfm-next weights/model.safetensors --chat
~~~

Il generatore Python controlla forme e tipi dei tensori e produce gli offset del checkpoint. Un secondo generatore compila il tokenizer ufficiale in `build/tokenizer.bin`, con vocabolario, merge BPE, token aggiunti e categorie Unicode ricavate dal medesimo motore del riferimento. Le tabelle occupano 4.224.959 byte e devono essere distribuite accanto all'eseguibile. Il runtime le trova tramite `/proc/self/exe` e ne verifica dimensione e checksum. Python serve durante la build e le verifiche. Per scegliere l'interprete passare `PYTHON=/percorso/venv/bin/python` a make.

La modalita `--text N "prompt"` applica il template ufficiale per un messaggio utente e il prefisso assistant, genera fino a N token e scrive la risposta UTF-8 in streaming. I token marcati speciali vengono omessi dal testo. La decodifica conserva le sequenze UTF-8 spezzate tra token e sostituisce quelle malformate con U+FFFD. La risposta termina senza aggiungere caratteri al testo generato. Ogni invocazione avvia una conversazione di un turno. Il testo con il template deve occupare al massimo 8.191 byte e rispettare il limite della cache.

Per usare la build presente da PowerShell nella directory del repository

~~~powershell
wsl -d Ubuntu -- ./build/lfm-next /opt/lfm-model/model.safetensors --text 64 "Scrivi un saluto in italiano."
~~~

La modalita `--chat [N]` mantiene il modello, i worker e la cronologia in memoria tra i turni. N limita i token generati per risposta e vale 128 per impostazione predefinita, ridotto alla dimensione del contesto se necessario. Inserire una domanda per riga, `/clear` per azzerare la conversazione e `/quit` oppure EOF per uscire. Le risposte sono emesse su stdout, mentre etichette e messaggi di stato usano stderr.

~~~powershell
wsl -d Ubuntu -- ./build/lfm-next /opt/lfm-model/model.safetensors --chat
wsl -d Ubuntu -- env LFM_CONTEXT=4096 LFM_THREADS=6 ./build/lfm-next /opt/lfm-model/model.safetensors --chat 128
~~~

La variabile `LFM_CONTEXT` accetta valori da 1 a 32.768. Il valore predefinito e 2.048 in chat e 512 nelle altre modalita. La cache KV FP32 viene allocata dinamicamente e riserva 24 KiB per posizione, quindi 48 MiB a 2.048 posizioni e 768 MiB a 32.768. Pesi e buffer di lavoro occupano memoria aggiuntiva. I limiti del testo UTF-8 restano 8.191 byte per richiesta, template incluso.

Prima di accettare un turno, la chat verifica lo spazio necessario per il nuovo prompt, N token di risposta e due posizioni per chiudere il turno. Quando lo spazio e insufficiente, mostra un messaggio e attende un'altra riga. `/clear` azzera la posizione e lo stato delle convoluzioni. I nuovi token sovrascrivono le righe KV prima di leggerle.

La cronologia conserva gli ID realmente generati. Ogni risposta viene terminata con EOS e una nuova riga secondo il template. Se la generazione raggiunge N prima di EOS, il programma segnala il limite e aggiunge alla cronologia i token di chiusura. Input malformati o eccessivamente lunghi vengono rifiutati prima di modificare lo stato del modello.

Il numero di thread predefinito e il minore tra sei e il numero di CPU disponibili nell'affinita del processo. La variabile `LFM_THREADS` permette di scegliere da 1 a 16 thread, incluso il thread principale. Se il sistema limita la creazione dei worker, il programma usa quelli creati con successo.

~~~sh
LFM_THREADS=4 ./build/lfm-next weights/model.safetensors 1 314 271
~~~

L'opzione `--generate N`, dopo il checkpoint e prima degli ID, genera fino a N token. Il valore predefinito e uno, compatibile con la CLI iniziale. La generazione termina quando raggiunge N oppure emette il token EOS 7, incluso nell'output. N deve essere positivo e la somma tra lunghezza del prompt e N meno uno deve rientrare nel contesto configurato. L'ultimo token previsto viene emesso senza richiedere una nuova posizione in cache. Tutti gli argomenti sono verificati prima dell'inferenza.

Gli ID nell'esempio seguente rappresentano la domanda `What is the capital of France?` con il chat template ufficiale.

~~~sh
./build/lfm-next weights/model.safetensors --generate 32 1 6 6423 708 3493 856 779 5706 803 4481 540 7 708 6 64015 708
~~~

Produce gli ID `1098 5706 803 4481 856 5242 523 7`, uno per riga. La chiamata equivalente con `--text 32 "What is the capital of France?"` stampa direttamente `The capital of France is Paris.`

Le modalita diagnostiche `--encode "testo"` e `--decode ID ...` espongono il tokenizer. Encode restituisce gli ID senza BOS implicito e decode omette i token speciali. Entrambe rispettano il numero di token consentito dal contesto configurato e usano le sole tabelle, quindi il percorso del checkpoint puo essere sostituito da `_`.

### Avvio come kernel in QEMU

Il kernel esegue inferenza e chat senza Linux nel sistema ospite. Multiboot 1 supporta l'avvio diretto di QEMU; Multiboot 2 supporta GRUB in modalita BIOS e UEFI. Il codice configura GDT, paging, long mode, stato SSE/AVX e gestori di eccezione. La cache viene allocata dalla mappa RAM del bootloader; pesi e tokenizer rimangono in memoria.

Oltre agli strumenti della build Linux servono GNU objcopy, QEMU, Pillow e il font DejaVu. La generazione del font usa `FONT_PYTHON=python3`, distinto dall'interprete `PYTHON` eventualmente impostato sul venv del modello. Il percorso verificato usa KVM su Linux/WSL2 con accesso a `/dev/kvm`. Per installare anche gli strumenti dell'immagine BIOS/UEFI su Ubuntu:

~~~sh
sudo apt-get install --no-install-recommends qemu-system-x86 qemu-system-gui python3-pil fonts-dejavu-core grub-pc-bin grub-efi-amd64-bin xorriso mtools ovmf
make os MODEL=weights/model.safetensors
python3 os/run.py build/lfm-boot.bundle
~~~

La build produce `build/lfm-os.elf` e `build/lfm-boot.bundle`. Il secondo contiene una copia invariata del checkpoint BF16 e delle tabelle tokenizer: 713.213.887 byte con i file verificati. Il kernel e un ELF32 per il caricatore Multiboot di QEMU, ma passa a 64 bit prima dell'inferenza. L'ELF64 intermedio resta disponibile in `build/lfm-os.elf64`.

Nell'ambiente gia predisposto, da PowerShell nella directory del repository:

~~~powershell
wsl -d Ubuntu -- make os MODEL=/opt/lfm-model/model.safetensors OS_BUNDLE=/opt/lfm-model/lfm-boot.bundle PYTHON=/opt/lfm-venv/bin/python
wsl -d Ubuntu -- /opt/lfm-venv/bin/python os/run.py /opt/lfm-model/lfm-boot.bundle
~~~

Il launcher assegna sei CPU alla VM e richiede sei worker totali, incluso il processore principale. `--cpus` cambia le CPU virtuali disponibili; `--threads` cambia quelle richieste dal kernel, entrambe da 1 a 16. Le tabelle ACPI MADT individuano i processori, che vengono avviati tramite xAPIC e sequenza INIT/SIPI. Il kernel usa solo le CPU avviate e compatibili con il percorso SIMD scelto; ACPI o xAPIC non supportati lasciano attivo un solo processore. Ogni worker calcola righe complete delle matrici, mantenendo l'ordine delle riduzioni.

L'avvio diretto usa 2.048 posizioni e un massimo di 32 token per risposta. La chat supporta `/clear`, `/quit`, Ctrl-D, CRLF e backspace, e i comandi del motore descritti per il modello principale, tranne il ragionamento. `--context` e `--tokens` cambiano i limiti. La RAM predefinita e 2 GiB; `--memory` modifica la memoria della VM. La cache conserva il costo di 24 KiB per posizione. Se la memoria non basta, il kernel stampa un errore e termina.

`--display gtk` apre la console video, la stessa console Pithos CORE del modello principale: Unicode con emoji a colori, tredici layout di tastiera, barra di stato e comandi della chat. L'avvio diretto usa VGA testuale; GRUB richiede un framebuffer RGB a 32 bit, normalmente 1024x768. COM1 conserva il testo UTF-8 completo con le stesse sequenze di colore. In assenza di una porta seriale, l'inferenza e la console video continuano a funzionare.

Per una singola risposta o gli ID generati:

~~~sh
python3 os/run.py build/lfm-boot.bundle --mode text --tokens 32 --prompt "What is the capital of France?"
python3 os/run.py build/lfm-boot.bundle --mode ids --tokens 8 --prompt "What is the capital of France?"
~~~

Le modalita singole delimitano il risultato con `<lfm-output>` e `</lfm-output>` sulla seriale. Il launcher interpreta il dispositivo QEMU `isa-debug-exit`, restituendo zero se il kernel termina correttamente. Con QEMU diretto, il codice host 33 indica successo del guest e 35 indica errore; il codice 1 di un errore di avvio di QEMU resta un errore. Senza KVM si puo specificare `--accel tcg`, ma l'emulazione software dell'inferenza e molto piu lenta.

### Immagine BIOS/UEFI e USB

~~~sh
make os-image MODEL=weights/model.safetensors
python3 os/run.py --iso build/lfm-os.iso --display gtk
python3 os/run.py --iso build/lfm-os.iso --uefi --display gtk
python3 os/run.py --iso build/lfm-os.iso --uefi --usb --display gtk
~~~

`build/lfm-os.iso` e un'immagine ibrida GRUB, circa 726 MB, contenente kernel, tokenizer e checkpoint BF16 invariato. Puo essere scritta come immagine raw su una chiavetta per avviare una macchina x86-64 compatibile BIOS o UEFI. La procedura di build crea soltanto il file ISO e non scrive su dispositivi fisici. GRUB legge i dati dal supporto e li consegna al kernel; dopo il boot il kernel lavora interamente in RAM e non accede al filesystem.

L'immagine avvia la chat con 64 token per risposta, contesto 2.048 e un massimo di sei CPU attive. I parametri sono incorporati in `grub.cfg`; il launcher consente di cambiare CPU e RAM virtuali, ma rifiuta parametri del kernel che non potrebbero modificare l'immagine. Per un'immagine con limiti differenti:

~~~sh
python3 os/image.py build/lfm-os.elf build/lfm-boot.bundle build/lfm-os.iso --options "mode=chat tokens=128 context=4096 threads=4"
~~~

L'ambiente predisposto conserva anche una copia nel filesystem Linux, che evita il costo degli accessi al disco attraverso il mount Windows:

~~~powershell
wsl -d Ubuntu -- make os-image MODEL=/opt/lfm-model/model.safetensors OS_BUNDLE=/opt/lfm-model/lfm-boot.bundle OS_IMAGE=/opt/lfm-model/lfm-os.iso PYTHON=/opt/lfm-venv/bin/python
wsl -d Ubuntu -- /opt/lfm-venv/bin/python os/run.py --iso /opt/lfm-model/lfm-os.iso --uefi --usb --display gtk
~~~

Il percorso da immagine usa QEMU q35 con AHCI/DMA, evitando il caricamento dei pesi tramite IDE PIO. L'emulazione USB usa un controller USB2 EHCI: xHCI ha prodotto blocchi intermittenti del firmware durante le prove in WSL/KVM, anche con MSI disattivato. Il caricamento dei circa 713 MB tramite EHCI puo richiedere oltre un minuto; l'avvio da CD virtuale o diretto e piu rapido per lo sviluppo. OVMF usa xAPIC, MMIO PCI sotto 4 GiB e una copia temporanea delle variabili firmware. La dimensione della finestra MMIO si imposta tramite la [configurazione ufficiale OVMF](https://github.com/tianocore/edk2/blob/master/OvmfPkg/RUNTIME_CONFIG.md). L'immagine non e firmata: sui PC con Secure Boot attivo serve disabilitarlo per avviarla.

Il boot e verificato in QEMU, non ancora su un PC fisico. La compatibilita attuale richiede RAM utilizzabile sotto 4 GiB, framebuffer RGB a 32 bit sotto 4 GiB oppure VGA/seriale, e tastiera PS/2 o seriale. Non e incluso un driver USB HID: l'avvio da supporto USB e gestito dal firmware/GRUB e non implica il funzionamento di ogni tastiera USB dopo il boot. Le CPU con x2APIC gia attivo usano il fallback a un core. Console e worker usano attesa attiva; il risparmio energetico in inattivita e ancora da implementare. Su hardware fisico `/quit` arresta i processori, ma non effettua uno spegnimento ACPI.

### Verifiche del kernel

~~~sh
make verify-os MODEL=weights/model.safetensors
make verify-platform MODEL=weights/model.safetensors
~~~

`verify-os` confronta token, testo UTF-8 e tutti i 65.536 logit con Linux a un thread. Copre chat con cronologia, reset, errori di input, limite della cache, CPU senza AVX/XSAVE, modulo mancante o corrotto e RAM insufficiente. `verify-platform` verifica generazione con 1, 2, 3, 4, 6 e 16 CPU, logit con 3 e 6 CPU, fallback senza ACPI, VGA, tastiera PS/2 e assenza di COM1. Avvia inoltre l'immagine con BIOS/UEFI da CD e USB emulati, verifica una risposta in ogni caso e salva le schermate in `build/os-*.png`. Richiede KVM e costruisce anche l'immagine e il kernel diagnostico `lfm-os-logits.elf`.

Verifica completata il 26 novembre 2026 con QEMU 10.2.1 e KVM in WSL2: cinque sequenze greedy e tre risposte testuali coincidono con Linux; tutti i logit sono identici bit per bit per due prompt. Il bundle conserva gli hash SHA-256 di checkpoint e tokenizer. Sono passati anche i test della chat seriale e degli errori di boot. Con contesto 32.768 sono verificate allocazione e prima predizione con 2 GiB di RAM, e il rifiuto della cache con 1 GiB; questo non costituisce una verifica numerica di sequenze lunghe 32.768 token. Le suite Linux di logit, generazione, chat, tokenizer e thread sono passate dopo la separazione delle piattaforme.

Le verifiche della piattaforma sono passate con 1, 2, 3, 4, 6 e 16 CPU, logit bit-identici a 3 e 6 CPU e fallback senza ACPI. BIOS/UEFI da CD e USB emulati hanno completato una risposta da tastiera PS/2. Il percorso UEFI/USB con EHCI e stato verificato anche dopo un avvio a freddo di WSL e su avvii successivi. Senza il dispositivo QEMU di uscita, tutti e quattro i processori controllati raggiungono HLT dopo `/quit`. Le suite Linux di logit fino a 512 token e generazione fino a 64 token sono state rieseguite con successo.

Il benchmark del kernel si esegue senza altre VM o benchmark in parallelo:

~~~sh
make benchmark-os MODEL=weights/model.safetensors
~~~

Usa sei vCPU, RAM di 2 GiB, contesto 512, lo stesso prompt da 25 token del benchmark Linux e 64 token generati. Confronta 1, 2, 4 e 6 CPU attive, controllando l'identita degli ID, con un riscaldamento e tre campioni per configurazione in ordine alternato. Distingue il caricamento/avvio della VM dal prefill e dalla generazione successiva. Campioni e mediane sono in `build/os-benchmark.json`.

Misure del 26 novembre 2026 su Intel Core i7-10750H, QEMU 10.2.1/KVM in WSL2, pesi BF16 nel filesystem Linux. Il prefill della tabella esclude l'avvio della VM; il totale lo include. La generazione successiva misura i 63 intervalli tra il primo e il sessantaquattresimo token.

| CPU attive | Prefill e primo token | Generazione successiva | Totale con avvio VM |
| --- | --- | --- | --- |
| 1 | 990 ms | 20,67 token/s | 4,61 s |
| 2 | 782 ms | 29,05 token/s | 4,00 s |
| 4 | 648 ms | 29,55 token/s | 3,41 s |
| 6 | 623 ms | 32,13 token/s | 3,23 s |

Sei CPU hanno dato il risultato migliore fra le configurazioni misurate, circa il 55% in piu nella generazione rispetto a una CPU. Il risultato riguarda questa macchina e questo carico; gli avvii BIOS/UEFI da immagine non sono inclusi nel benchmark, che usa il caricamento Multiboot diretto di QEMU.

Linux/WSL resta il sistema host che esegue QEMU e gli strumenti di preparazione. I test non equivalgono a una certificazione di compatibilita hardware o a un confronto di velocita con altri runtime.

### Qualita

Il progetto usa esclusivamente i pesi BF16 ufficiali, con attivazioni e accumulazioni FP32. Questa configurazione ha dato la qualita migliore tra quelle provate e conserva la precisione dei pesi originali. Il checkpoint occupa 708.984.464 byte; cache e buffer richiedono memoria aggiuntiva.

`quality_corpus.json` contiene 12 testi originali, sei italiani e sei inglesi, per 1.739 token valutati, e otto richieste semplici. E un piccolo corpus diagnostico, non un benchmark rappresentativo delle capacita generali.

~~~sh
make evaluate-quality MODEL=weights/model.safetensors
~~~

La valutazione usa Transformers in FP32 con i pesi BF16 ufficiali, un BOS per documento, nessun troncamento o EOS aggiunto e medie pesate per il numero di token. Un documento per lingua viene verificato confrontando gli ultimi logit con il runtime assembly. Le risposte alle otto richieste vengono generate dal runtime. Metodo, hash del corpus, risultati per documento e risposte complete sono in `build/quality-report.json`.

La perplexity misurata e 27,44 in italiano, 32,86 in inglese e 29,55 complessivamente. Cinque delle otto risposte rispettano il risultato e il formato richiesti. Il confronto ignora maiuscole, spazi e punti finali; una spiegazione aggiunta quando viene richiesta una sola parola puo quindi risultare non conforme.

### Verifica

Con PyTorch e Transformers che supporta LFM2 installati, eseguire:

~~~sh
make verify MODEL=weights/model.safetensors
~~~

Il test confronta la predizione greedy e tutti i 65.536 logit per sequenze da 1, 2, 5, 16, 64 e 512 token. La domanda formattata con il chat template ufficiale viene ripetuta per esercitare anche il limite della cache. Il riferimento usa gli stessi pesi in FP32 con attenzione eager. Le tolleranze sono assoluta 0,001 e relativa 0,0001. Il test verifica anche il rifiuto degli argomenti fuori intervallo o malformati.

L'ambiente di verifica su questa macchina risiede in Ubuntu WSL2, in `/opt/lfm-venv`. Da PowerShell nella directory del repository si esegue con

~~~powershell
wsl -d Ubuntu -- make verify MODEL=weights/model.safetensors PYTHON=/opt/lfm-venv/bin/python
~~~

Con AVX2/FMA il prompt viene elaborato a blocchi di 15 token, mantenendo i pesi BF16. Il percorso scalare elabora ancora il prompt token per token.

Verifica BF16 completata il 26 novembre 2026 su Ubuntu WSL2 con NASM 3.01, PyTorch 2.15.0+cpu e Transformers 5.24.1. Le sei predizioni coincidono con il riferimento, fino a 512 token. Lo scarto massimo assoluto sui logit e 0,00018620. La domanda `What is the capital of France?` produce come primo token `The` (ID 1098). L'eseguibile ELF e statico; pesi e tabelle del tokenizer sono in file separati.

SHA-256 del checkpoint verificato

~~~text
1c9c77a4471a7f590f85240f74ed1fc26df7fbde88c3006724e2f93ca993ea4e
~~~

### Implementazione

- main.asm verifica il modello, esegue gli strati e calcola i logit con la matrice di embedding condivisa.
- platform_linux.asm gestisce CLI, caricamento dei file, memoria e I/O tramite syscall Linux.
- os/boot.asm e os/platform.asm gestiscono avvio a 64 bit, modulo BF16, allocazione della RAM e I/O del kernel autonomo.
- os/multiboot.asm normalizza le informazioni di avvio Multiboot 2; os/smp.asm rileva e avvia le CPU e distribuisce le righe delle matrici.
- os/console.asm gestisce VGA, framebuffer RGB, decodifica UTF-8 per il font Latin-1 e tastiera PS/2.
- kernels.asm implementa prodotti matrice-vettore, RMSNorm, esponenziale e convoluzione causale con due campioni di storia per canale.
- attention.asm normalizza query e key per testa, applica RoPE e calcola la grouped-query attention sulla cache.
- layers.inc contiene l'ordine dei dieci blocchi convoluzionali e dei sei blocchi di attenzione.
- threads.asm implementa il gruppo di worker tramite le syscall clone e futex. Ogni worker calcola righe complete della matrice. Una breve attesa attiva precede il blocco su futex e il processo termina tutti i worker tramite exit_group.
- tokenizer.asm implementa la suddivisione Unicode del testo, il riconoscimento dei token aggiunti, i merge BPE e la decodifica UTF-8 in streaming. Il tokenizer opera sul thread principale tra i lavori di inferenza e usa tabelle in sola lettura.
- chat.asm gestisce la lettura delle righe, i comandi della chat, il recupero dagli errori di input e l'allocazione della cache tramite il backend della piattaforma.

I prodotti matrice-vettore AVX2 espandono i BF16 esattamente in FP32 e usano quattro accumulatori FMA indipendenti. La selezione verifica anche il supporto del sistema operativo ai registri YMM tramite XGETBV. Le rotazioni RoPE vengono calcolate una volta per posizione e riutilizzate dalle 144 teste query/key dei sei strati di attenzione. L'attenzione elabora quattro componenti per istruzione SSE e RMSNorm calcola una sola divisione per vettore.

Con lo stesso percorso AVX2/FMA, l'esponenziale della SiLU negli MLP e della softmax nell'attenzione elabora otto valori per istruzione. La riduzione dell'argomento segue Cephes, con un polinomio di quinto grado e gli stessi limiti di ±80 della versione x87. Su 1.048.576 input fra -90 e 90 l'errore massimo e 1,002 ulp, contro 0,5 ulp della x87. La softmax maschera l'ultimo blocco incompleto, quindi le posizioni oltre quella corrente non vengono lette ne sommate. La build scalare e le CPU senza AVX2/FMA mantengono l'esponenziale x87, con risultati identici bit per bit a prima. Tutte le verifiche di Linux e del kernel sono passate anche con questa modifica: logit entro le tolleranze di Transformers, con scarto massimo 0,000115, e bit-identici fra thread, fra core e fra Linux e kernel. Su una macchina con quattro CPU visibili in WSL2 la generazione e cresciuta di circa il 10-14%, con un margine di rumore fra ripetizioni vicino al 10%.

Anche l'attenzione ha un percorso AVX2/FMA. Le 64 componenti della query restano nei registri durante i prodotti con le key, e una sola passata sulle posizioni accumula tutte le componenti dei valori. Le 16 teste sono distribuite fra i worker, e ogni partizione usa una propria riga di punteggi. Ogni testa e calcolata dallo stesso codice qualunque sia il thread, quindi i logit restano identici bit per bit fra 1 e 16 thread o core. Linux e kernel condividono il meccanismo dei job: i worker eseguono la funzione indicata da `job_entry`, che puo essere il matvec o l'attenzione. Il percorso scalare dell'attenzione resta sul thread principale. Sulla stessa macchina, con 1.900 posizioni in cache, la generazione e salita da 15,8 a 21,4 token/s con un thread e da circa 23 a 28 token/s con quattro.

Il prompt viene elaborato a blocchi di 15 token, impostabili a compile time con `-DPREFILL_BLOCK=N`. Nelle matrici ogni riga di pesi viene letta una sola volta per blocco, e ogni vettore BF16 convertito serve fino a tre token. Ogni coppia (riga, token) usa gli stessi quattro accumulatori e lo stesso albero di riduzione del matvec a un token, quindi il prefill a blocchi e identico bit per bit all'elaborazione token per token. Normalizzazioni, stato delle convoluzioni, RoPE e attenzione avanzano un token alla volta nell'ordine originale. Le key e i valori del blocco vengono scritti in cache prima dell'attenzione, e ogni token legge solo le posizioni fino alla propria. Logit, generazione e trascrizioni della chat coincidono byte per byte con la versione precedente, per prompt da 1 a 511 token e con 1, 3 e 4 thread. Sulla stessa macchina, misurando l'intero processo fino al primo token, i tempi sono questi:

| Prompt | Thread | Token per token | A blocchi | Accelerazione |
| --- | --- | --- | --- | --- |
| 25 token | 1 | 896 ms | 371 ms | 2,4x |
| 512 token | 1 | 18,66 s | 7,53 s | 2,5x |
| 25 token | 4 | 631 ms | 179 ms | 3,5x |
| 512 token | 4 | 16,55 s | 3,51 s | 4,7x |

Il prefill a blocchi e limitato dal calcolo FMA e non piu dalla banda di memoria, quindi scala con i thread; blocchi piu grandi di 15 token non hanno migliorato i tempi.

Le scelte architetturali seguono il rapporto tecnico LFM2 e la configurazione del checkpoint. Le matrici dense dei blocchi convoluzionali e degli MLP restano il principale obiettivo del kernel vettoriale. [Rapporto tecnico Liquid AI](https://arxiv.org/html/2511.23404v1#S2)

Per confrontare sullo stesso checkpoint il percorso scalare e quello selezionato automaticamente, eseguire

~~~sh
make benchmark MODEL=weights/model.safetensors
~~~

Il benchmark misura il tempo dell'intero processo per predire un token, incluso il prompt. Esegue un riscaldamento e tre ripetizioni con ordine alternato, verifica il token prodotto e salva tutti i campioni in `build/benchmark.json`. Il file deve restare identico per entrambi gli eseguibili. Con WSL, la posizione dei pesi nel filesystem influenza sensibilmente la misura. La copia Linux usata per i confronti locali si trova in `/opt/lfm-model/model.safetensors`.

Misure del 26 novembre 2026 su Intel Core i7-10750H, Ubuntu WSL2, singolo thread e pesi BF16 nel filesystem Linux. Mediane di cinque ripetizioni dopo il riscaldamento, confrontando `lfm-scalar` e `lfm-next`.

| Prompt | Matvec scalare | Matvec AVX2/FMA | Accelerazione |
| --- | --- | --- | --- |
| 1 token | 351,68 ms | 65,26 ms | 5,39x |
| 16 token | 4.455,52 ms | 637,90 ms | 6,98x |

Per ripetere questa configurazione dalla shell Linux nella directory del repository

~~~sh
make build/lfm-scalar build/lfm-next MODEL=/opt/lfm-model/model.safetensors
python3 benchmark.py /opt/lfm-model/model.safetensors build/lfm-scalar build/lfm-next --repeats 5 --output build/benchmark.json
~~~

La latenza include avvio, mmap, elaborazione del prompt e predizione finale. Le misure di questa sezione precedono il prefill a blocchi descritto in Implementazione. I risultati di altri runtime richiedono un confronto con gli stessi pesi, precisione e hardware.

Il confronto successivo del parallelismo, sulla stessa CPU e con cinque ripetizioni dopo il riscaldamento, ha prodotto queste mediane per il prompt da 16 token.

| Thread | Latenza | Accelerazione rispetto a 1 thread |
| --- | --- | --- |
| 1 | 668,33 ms | 1,00x |
| 2 | 481,73 ms | 1,39x |
| 4 | 470,67 ms | 1,42x |
| 6 | 472,06 ms | 1,42x |
| 8 | 469,80 ms | 1,42x |

I risultati a 4, 6 e 8 thread sono vicini. Il valore predefinito di sei limita l'uso di thread su questa CPU con sei core fisici. Il limite riguarda i prodotti matrice-vettore, mentre le altre operazioni restano sul thread principale.

~~~sh
make benchmark-threads MODEL=/opt/lfm-model/model.safetensors
make verify-threads MODEL=/opt/lfm-model/model.safetensors
~~~

La verifica dei thread confronta i logit bit per bit con il percorso a un thread, include partizioni con resto e contesti fino a 512 token, esegue otto richieste con quattro processi simultanei e controlla i valori non validi di LFM_THREADS.

La chat e il contesto dinamico si verificano con

~~~sh
make verify-chat MODEL=/opt/lfm-model/model.safetensors PYTHON=/opt/lfm-venv/bin/python
~~~

Il confronto con Transformers copre quattro turni, reset, risposte troncate e chiusura dei turni, input rifiutati, CRLF ed EOF dopo una riga incompleta. Sono verificati anche l'esaurimento esatto della cache, un prompt di 513 token con contesto di 1.024 posizioni e i limiti della configurazione. A 32.768 posizioni viene verificata l'allocazione e una predizione iniziale. La verifica completa dei logit resta sui contesti fino a 512 posizioni.

La verifica del tokenizer si esegue con

~~~sh
make verify-tokenizer MODEL=/opt/lfm-model/model.safetensors PYTHON=/opt/lfm-venv/bin/python
~~~

Sono passati 774 casi di codifica e 1.130 casi di decodifica, inclusi tutti gli ID del vocabolario in gruppi da 256, i token aggiunti, testo multilingue, emoji e UTF-8 malformato o incompleto. I test controllano anche i limiti di ingresso e la ricerca delle tabelle quando cambia la directory di lavoro. Le tabelle mancanti o corrotte producono un errore. L'inferenza testuale in inglese, italiano e cinese coincide con quella avviata dagli ID del tokenizer ufficiale.

La generazione continuativa si verifica e si misura con

~~~sh
make verify-generation MODEL=/opt/lfm-model/model.safetensors PYTHON=/opt/lfm-venv/bin/python
make benchmark-generation MODEL=/opt/lfm-model/model.safetensors
~~~

La verifica confronta con Transformers una risposta conclusa da EOS, una risposta limitata a tre token, una sequenza di 64 token e due token generati dopo un prompt di 511 posizioni. Controlla anche la compatibilita della CLI iniziale, gli argomenti non validi e l'identita della sequenza lunga tra uno e sei thread.

Benchmark del 26 novembre 2026 sulla stessa CPU, pesi BF16 nel filesystem Linux, prompt di 25 token e generazione di 64 token. Mediane di tre ripetizioni dopo il riscaldamento.

| Thread | Tempo al primo token | Generazione successiva | Tempo totale |
| --- | --- | --- | --- |
| 1 | 979 ms | 21,45 token/s | 3,92 s |
| 6 | 700 ms | 28,41 token/s | 2,92 s |

Il benchmark osserva i token su stdout. Il tempo al primo token include l'avvio del processo e il prefill. La velocita successiva e calcolata sui 63 intervalli dopo il primo token. I campioni completi sono salvati in `build/generation-benchmark.json`.

[Checkpoint e configurazione ufficiali](https://huggingface.co/LiquidAI/LFM2.5-350M/tree/main)
