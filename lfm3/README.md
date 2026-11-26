# LFM3-1.7B: runtime e riferimento

Questa cartella aggiunge il supporto a LiquidAI LFM3-1.7B in formato GGUF Q4_K_M: un eseguibile Linux in assembly, `lfm3-next`, il riferimento numerico e le verifiche. Il runtime condivide con quello di LFM2.5 piattaforma, thread, tokenizer e chat; il kernel autonomo non lo supporta ancora.

## Build e utilizzo

Servono gli stessi strumenti della build principale, il GGUF e i file ufficiali del tokenizer. Su CPU senza AVX2, FMA e F16C il programma si ferma con un messaggio.

~~~sh
make lfm3 LFM3_MODEL=LFM3-1.7B-Q4_K_M.gguf LFM3_TOKENIZER=/opt/lfm3-ref/tokenizer.json
./build/lfm3/lfm3-next LFM3-1.7B-Q4_K_M.gguf --text 128 "Scrivi due frasi su Torino."
./build/lfm3/lfm3-next LFM3-1.7B-Q4_K_M.gguf --chat 256
~~~

Le opzioni sono quelle di `lfm-next`: `--generate`, `--text`, `--chat`, `--encode`, `--decode`, `LFM_CONTEXT` e `LFM_THREADS`. Gli ID vanno da 0 a 127.999. `build/lfm3/tokenizer.bin`, di 12.519.600 byte, deve restare accanto all'eseguibile. Il prompt segue il template ufficiale con `enable_thinking=False`, che termina con un blocco di ragionamento vuoto `<think></think>`; anche i turni precedenti della chat lo conservano, come nel template.

I prodotti matrice-vettore e l'attenzione sono distribuiti fra i worker, e il prompt viene elaborato a blocchi di 15 token. Logit, generazione e trascrizioni della chat sono identici bit per bit fra 1 e 16 thread e fra prefill a blocchi e prefill token per token.

Misure del 27 novembre 2026 su Intel Core i5-10500H, WSL2 con quattro CPU visibili, prompt di 465 token e 64 token generati, mediane di tre ripetizioni. Il tempo del primo token include l'avvio del processo.

| Thread | Primo token | Prompt | Generazione |
| --- | --- | --- | --- |
| 1, token per token | 38,04 s | 12 token/s | 9,91 token/s |
| 1 | 19,67 s | 24 token/s | 9,79 token/s |
| 2 | 11,82 s | 39 token/s | 16,37 token/s |
| 4 | 7,95 s | 59 token/s | 17,77 token/s |

Sulla stessa macchina pithos su Windows, con `-t 4`, ha elaborato lo stesso prompt a 55,7-61,9 token/s, avvio escluso, e generato a 19,0-21,4 token/s; con sei thread 67-68 e 22 token/s. La generazione e limitata dalla banda di memoria: ogni token legge circa 1 GB di pesi.

## Implementazione

- `layout.py` legge il GGUF in fase di build, verifica metadati, forme e tipi di tutti i 287 tensori e genera offset, tipi e la sequenza degli strati presa dal checkpoint. L'eseguibile controlla all'avvio un hash FNV-1a dell'intestazione GGUF, 8.247.328 byte.
- Ogni ingresso di matrice viene quantizzato una volta in Q8_K: blocchi da 256 valori con scala 127 / max |x| e arrotondamento al pari. I prodotti con le righe Q4_K e Q6_K sono interi, con `vpmaddubsw` e `vpmaddwd`, seguiti da una FMA per blocco, come nei kernel AVX2 di pithos e ggml. L'embedding condiviso Q6_K fornisce sia la riga di ingresso, dequantizzata nell'ordine di ggml, sia i 128.000 logit.
- RMSNorm usa pesi F32, la convoluzione conserva sei campioni di storia per canale, e la sandwich norm normalizza l'uscita di ogni blocco prima del residuo.
- L'attenzione calcola norma e RoPE per testa, con le frequenze FP32 generate da `modeling_lfm3.py` in `frequencies.inc`; negli strati ad attenzione piena ruotano le prime 32 coppie. Ogni testa legge la testa KV h / 4, gli strati a finestra le ultime 256 posizioni, e l'uscita viene moltiplicata per il gate sigmoid della testa. La cache KV FP32 occupa 44 KiB per posizione.
- I worker ricevono righe intere delle matrici, o teste intere dell'attenzione con una propria riga di punteggi, tramite il meccanismo `job_entry` condiviso con LFM2.5. Nel prefill a blocchi ogni blocco di pesi Q4_K o Q6_K viene spacchettato una volta e riusato per tutti i token; per ogni coppia (riga, token) somme intere, FMA per blocco e riduzioni finali sono quelle del kernel a un token. Norme, convoluzione, RoPE e attenzione avanzano un token alla volta, e le key e i valori del blocco vengono scritti in cache prima dell'attenzione. La build diagnostica `lfm3-layers` elabora il prompt token per token per conservare l'ordine dei dump.
- `build_tokenizer.py` accetta `--vocab` e `--hash-bits` e, per i tokenizer con `ignore_merges`, aggiunge una tabella da sequenza di byte a ID; `tokenizer.asm` cerca prima il pezzo intero. Con i valori predefiniti le tabelle di LFM2.5 restano identiche byte per byte, e anche codice e dati caricati degli eseguibili LFM2.5 e del kernel coincidono con la versione precedente.

## Verifica

~~~sh
make verify-lfm3-tokenizer LFM3_TOKENIZER=/opt/lfm3-ref/tokenizer.json
make verify-lfm3 LFM3_MODEL=LFM3-1.7B-Q4_K_M.gguf LFM3_TOKENIZER=/opt/lfm3-ref/tokenizer.json LFM3_REF=/opt/lfm3-ref
~~~

Il tokenizer coincide con quello ufficiale su 62 testi, fra cui italiano, cinese, giapponese, coreano, arabo, hindi, thai, emoji, codice e token speciali, e nella decodifica di tutti i 128.000 ID in gruppi da 2.000.

Per il modello un confronto bit per bit non e possibile. L'arrotondamento Q8_K e discontinuo: un valore al confine fra due livelli si sposta di un passo intero per una differenza di un ulp a monte, e questi casi si amplificano nei 24 strati, fino a circa un logit sui logit finali. Riguarda qualunque motore con attivazioni Q8_K, compresi ggml e pithos. `verify.py` controlla quindi tre aspetti.

- Replay dei blocchi: la build diagnostica `lfm3-layers` scrive l'embedding e l'uscita normalizzata di ogni blocco per ogni token. Ogni blocco del riferimento, con gli stessi ingressi Q8_K, viene eseguito sugli stati del runtime per tutti i token, con le maschere causali e a finestra e le RoPE del forward ufficiale, e deve riprodurre l'uscita del runtime. Un arrotondamento al confine resta locale, a un token o ai token che leggono la key e il valore cambiati; un difetto colpirebbe la maggior parte dei token. La mediana di ogni blocco deve restare entro 1e-5 relativo, e la quota di coppie (token, blocco) oltre questa soglia viene confrontata con quella del riferimento contro se stesso, quando i suoi ingressi si spostano di un ulp.
- Riferimento FP32: l'argmax dei logit finali deve coincidere.
- Generazione: dopo un prompt di 463 token, ogni token greedy deve essere l'argmax del riferimento FP32, salvo quasi pareggi entro 0,25 logit.

Risultati del 26 novembre 2026:

| Controllo | Risultato |
| --- | --- |
| Replay, 20 token x 48 blocchi | mediana peggiore 1,2e-6; oltre 1e-5 il 3,23% delle coppie, contro il 3,85% del riferimento con se stesso |
| Replay, 463 token x 48 blocchi | mediana peggiore 1,6e-6; oltre 1e-5 il 4,72%, contro l'8,75% |
| Controllo negativo: finestra di 255 posizioni | mediana 9,3e-5 nello strato 12 a finestra: respinto |
| Argmax FP32, prompt da 1, 20, 463 e 721 token | 4 su 4; differenza massima sui logit fra 0,20 e 1,48 |
| Generazione greedy, 96 token | 95 argmax del riferimento; l'unico scostamento e un quasi pareggio di 0,018 logit |

Con un solo token, ogni blocco coincide con il riferimento entro 1e-7-1e-6 relativo, salvo due MLP con un arrotondamento al confine a 3e-5. La risposta a `What is the capital of France?` coincide token per token con il riferimento e con pithos.

Con il template ufficiale il modello Q4_K_M a volte chiude un ragionamento inesistente e ripete la risposta, ad esempio `Il colore azzurro.</think>Il colore azzurro.`; tutti i 14 token sono l'argmax del riferimento FP32, quindi il comportamento appartiene al modello. La chat scrive le risposte su stdout e le etichette su stderr. Leggendo i due flussi da pipe separate, come fa un terminale remoto o uno strumento che li riassembla a blocchi, un messaggio di stato puo inserirsi fra i byte di un carattere UTF-8 e apparire come U+FFFD. Con lo stesso input e i flussi su file, separati o uniti, l'uscita e UTF-8 valido senza caratteri di sostituzione.

## Riferimento FP32

`reference.py` costruisce l'architettura ufficiale `Lfm3ForCausalLM` dal codice `modeling_lfm3.py` del repository Hugging Face, senza allocarne i pesi. Ogni `Linear` e l'embedding dequantizzano il proprio tensore GGUF quando vengono chiamati, con i dequantizzatori del pacchetto `gguf` di llama.cpp, quindi in memoria resta una sola matrice alla volta e il modello gira in circa 2 GiB. Norme e coefficienti delle convoluzioni sono gia F32 nel GGUF. Il calcolo e in FP32 con attenzione eager.

Al caricamento lo script verifica che i metadati del GGUF coincidano con `config.json`: strati, dimensioni, teste, finestra, basi RoPE, kernel della convoluzione, token speciali e l'equivalenza fra `rope_freqs` e la rotazione parziale. Controlla anche che tutti i 287 tensori siano assegnati e che nessun parametro resti senza pesi.

~~~sh
hf download LiquidAI/LFM3-1.7B config.json configuration_lfm3.py modeling_lfm3.py lfm3_conv.py tokenizer.json tokenizer_config.json chat_template.jinja special_tokens_map.json generation_config.json --local-dir /opt/lfm3-ref
python lfm3/reference.py MODEL.gguf /opt/lfm3-ref check
python lfm3/reference.py MODEL.gguf /opt/lfm3-ref logits out.f32 ID ...
python lfm3/reference.py MODEL.gguf /opt/lfm3-ref teacher P ID ...
~~~

Servono PyTorch, Transformers 5, `gguf` e `accelerate`. Ogni passaggio dequantizza l'intero checkpoint, circa 15-20 secondi su questa macchina, quindi la generazione token per token e lenta. `teacher` verifica invece l'output di un altro motore con un solo passaggio: i primi P ID sono il prompt, gli altri la continuazione greedy, e ogni token della continuazione deve essere l'argmax del riferimento. Per gli scostamenti stampa la distanza in logit dal massimo.

## Verifiche del riferimento

- Il GGUF corrisponde a `config.json` e tutti i tensori si mappano sull'architettura ufficiale.
- Il tokenizer ufficiale e quello di pithos producono gli stessi 20 ID per il prompt di prova.
- Con `What is the capital of France?` e ragionamento disattivato, riferimento e pithos generano gli stessi nove token, EOS incluso: `The capital of France is **Paris**.`
- Con un prompt italiano di 465 token, oltre la finestra di 256 posizioni, 89 dei 92 token generati da pithos sono l'argmax del riferimento. I tre scostamenti sono quasi pareggi, con distanze di 0,215, 0,087 e 0,008 logit, coerenti con la quantizzazione Q8_K delle attivazioni e la cache FP16 di pithos.

pithos e stato eseguito con `--cpu --no-cache --reasoning off --temp 0 --repeat-penalty 1.0 --ids`. Il suo prompt senza ragionamento termina con `<think>\n</think>\n`, mentre il template ufficiale con `enable_thinking=False` produce `<think></think>`; i confronti usano lo stesso testo per entrambi i motori.

## Differenze rispetto a LFM2.5-350M

| | LFM2.5-350M | LFM3-1.7B |
| --- | --- | --- |
| Strati | 16: 10 convoluzioni, 6 attenzione | 24: 13 convoluzioni, 6 attenzione piena, 5 a finestra di 256 |
| Dimensione e MLP | 1024, 4608 | 2048, 7680 |
| Teste | 16 query, 8 KV, dimensione 64 | 16 query, 4 KV, dimensione 128 |
| Convoluzione | 3 coefficienti | 7 coefficienti, sei campioni di storia |
| Pesi | BF16, 709 MB | Q4_K e Q6_K, 1,1 GB; embedding condiviso Q6_K |
| Vocabolario | 65.536 | 128.000 righe, ID usati fino a 125.032 |

Componenti nuovi, tutti presenti nel codice ufficiale:

- sandwich norm: un RMSNorm sull'uscita del blocco di convoluzione o attenzione e uno sull'uscita dell'MLP, prima di ciascun residuo;
- un gate per testa: `sigmoid(gate_proj(x))`, calcolato dall'ingresso normalizzato dell'attenzione, moltiplica l'uscita della testa;
- negli strati ad attenzione piena la RoPE ruota solo le prime 32 coppie, con base 1e6 e le frequenze della dimensione intera; le altre 32 restano ferme. Gli strati a finestra ruotano tutte le 64 coppie con base 1e4. Le coppie sono (i, i+64), come nella versione attuale;
- negli strati a finestra ogni token vede le ultime 256 posizioni, la propria inclusa;
- il template avvia il ragionamento `<think>` per impostazione predefinita.

## Tokenizer

La regex di pre-tokenizzazione e la stessa di LFM2, quindi la suddivisione Unicode di `tokenizer.asm` resta valida. Cambiano quattro aspetti:

- `ignore_merges` e attivo: un pezzo presente intero nel vocabolario si emette direttamente, senza merge. Serve una tabella da sequenza di byte a ID;
- i merge sono 293.320, oltre la tabella hash attuale da 131.072 posizioni;
- gli ID superano 65.535;
- il token piu lungo occupa 386 byte, oltre il buffer di decodifica da 128.

## Kernel quantizzati

Il motore C pithos (`src/quant_avx2.c`) e il riferimento per i kernel. Le attivazioni di ogni riga vengono quantizzate in Q8_K: blocchi da 256 valori int8 con una scala e sedici somme parziali. I prodotti Q4_K e Q6_K sono interi, con `vpmaddubsw` e `vpmaddwd`, e una sola FMA in virgola mobile per blocco. Le correzioni dei minimi Q4_K e dell'offset -32 di Q6_K usano le somme parziali. Le varianti a piu token spacchettano ogni blocco di pesi una volta e lo riusano per tutti i token, come il prefill a blocchi attuale.

## Memoria

Con cache KV in FP32, gli 11 strati di attenzione occupano 44 KiB per posizione; con un anello di 256 posizioni per i cinque strati a finestra si scende a circa 24 KiB. Il checkpoint di 1,1 GB rientra nella RAM sotto 4 GiB usata dal kernel autonomo.
