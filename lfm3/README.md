# LFM3-1.7B: riferimento e note di porting

Questa cartella prepara il supporto a LiquidAI LFM3-1.7B in formato GGUF Q4_K_M. Il runtime assembly non lo supporta ancora; qui ci sono il riferimento numerico e le caratteristiche del modello che il porting deve rispettare.

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

## Verifiche della fase 1

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
