# PoS tagging italiano con BERT, ELECTRA e Qwen

Progetto di **Intelligenza Artificiale** sul fine-tuning di modelli linguistici per il *Part-of-Speech tagging* italiano: assegnare a ogni parola la sua categoria grammaticale nel contesto della frase, usando le etichette Universal POS (UPOS).

```text
Il   gatto   dorme   .
DET  NOUN    VERB    PUNCT
```

L'obiettivo è confrontare due encoder italiani, **BERT XXL** ed **ELECTRA XXL**, con un modello generativo multilingue, **Qwen2.5-1.5B-Instruct**, sullo stesso corpus. Un secondo obiettivo è studiare quanto l'informazione sulle maiuscole contribuisca al tagging, confrontando modelli addestrati sul testo originale e sul testo convertito in minuscolo.

## Dataset

Il progetto utilizza [UD Italian-ISDT 2.15](https://github.com/UniversalDependencies/UD_Italian-ISDT/tree/r2.15), caricato da Hugging Face tramite la configurazione `it_isdt` di [`universal-dependencies/universal_dependencies`](https://huggingface.co/datasets/universal-dependencies/universal_dependencies). Si mantengono gli split ufficiali:

| Split | Frasi | Parole |
|---|---:|---:|
| Training | 13.121 | 276.014 |
| Development | 564 | 11.907 |
| Test | 482 | 10.417 |

Le parole e i confini di frase sono quelli annotati nel corpus (*gold*): la valutazione riguarda l'assegnazione dei tag a parole già segmentate. Le revisioni dei dati e dei checkpoint sono fissate nel notebook per rendere riproducibili gli esperimenti.

## Modelli e approccio

| Modello | Checkpoint | Adattamento al task |
|---|---|---|
| BERT XXL italiano | [`dbmdz/bert-base-italian-xxl-cased`](https://huggingface.co/dbmdz/bert-base-italian-xxl-cased) | Fine-tuning completo con classificazione per parola |
| ELECTRA XXL italiano | [`dbmdz/electra-base-italian-xxl-cased-discriminator`](https://huggingface.co/dbmdz/electra-base-italian-xxl-cased-discriminator) | Fine-tuning completo con classificazione per parola |
| Qwen2.5-1.5B-Instruct | [`Qwen/Qwen2.5-1.5B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) | Fine-tuning con QLoRA a 4 bit e generazione dei tag |

**BERT ed ELECTRA** producono rappresentazioni delle subword, che vengono aggregate con **mean pooling** per ottenere un vettore per parola, seguito da un classificatore lineare. La media combina tutti i frammenti della parola senza aggiungere parametri e mantiene una sola predizione e un solo contributo alla loss per parola, indipendentemente dalla sua frammentazione. Offre così una strategia comune ai due encoder.

**Qwen** riceve un'istruzione con le parole numerate e genera un array JSON con un tag UPOS per parola, nello stesso ordine. L'adattamento QLoRA aggiorna gli adapter mantenendo quantizzati i pesi del modello di base, per contenere il consumo di memoria.

L'implementazione usa **PyTorch** e l'ecosistema **Hugging Face**: Transformers e `Trainer` per i modelli e il training, Datasets per il corpus, PEFT e bitsandbytes per QLoRA.

## Protocollo sperimentale

Il piano comprende **18 training**: tre modelli, due condizioni di input e tre seed (`42`, `123`, `456`).

- **Testo originale e minuscolo:** ogni modello viene addestrato separatamente nelle due condizioni. La trasformazione si applica a training, development e test, conservando ordine delle parole ed etichette.
- **Selezione del checkpoint:** a ogni epoca si valuta il development; si sceglie il checkpoint con la macro-F1 UPOS totale più alta e lo si usa per la valutazione sul test.
- **Valutazione:** accuracy, macro-F1 totale e sulle classi presenti, prestazioni per categoria e variabilità tra seed. Per Qwen si misura anche la validità delle risposte generate.
- **Analisi degli errori:** confusioni tra categorie ed esempi nel contesto della frase, ricavati dalle predizioni salvate.

La valutazione comparativa completa è in corso. Il [report](docs/report/report.pdf) approfondisce le architetture, le scelte metodologiche, le loss, gli iperparametri e i limiti del confronto.

## Utilizzo

Clonare la repository:

```console
git clone https://github.com/pierreerrico/italian-pos-tagging.git
cd italian-pos-tagging
```

1. Preparare un ambiente Python con Jupyter e PyTorch con supporto CUDA. La configurazione QLoRA richiede una GPU compatibile con la quantizzazione a 4 bit.
2. Aprire [`notebooks/Italian_UPOS_AI.ipynb`](notebooks/Italian_UPOS_AI.ipynb) e selezionare il kernel dell'ambiente. La prima cella installa le altre dipendenze.
3. Controllare modelli attivi, seed, parametri e directory di output nelle celle di configurazione, quindi eseguire le celle in ordine.

La cella di training avvia tutti i run configurati, 18 con le impostazioni iniziali, e riutilizza quelli già completati e compatibili. Batch e lunghezze di sequenza vanno verificati sulle risorse disponibili. Per esperimenti con impostazioni diverse, scegliere una nuova directory di output.

Configurazioni, versioni delle librerie, checkpoint, metriche e predizioni vengono salvati in `outputs/upos_trainer_v1/`. I dati locali sono in `data/`; entrambe le cartelle sono escluse da Git.

### Analisi dei risultati

Dopo aver prodotto le predizioni e il riepilogo nel notebook, gli script consentono di estrarre gli errori e generare i grafici senza ripetere training o inferenza:

```console
python scripts/analysis/extract_error_examples.py
python scripts/figures/plot_results.py
```

L'analisi salva confusioni ed esempi in `outputs/upos_trainer_v1/analysis/`. I grafici richiedono Matplotlib (`pip install matplotlib`) e il riepilogo completo dei 18 run; vengono salvati in `docs/report/figures/`.

## Struttura e documentazione

- [`notebooks/Italian_UPOS_AI.ipynb`](notebooks/Italian_UPOS_AI.ipynb): pipeline completa, dalla preparazione dei dati alla valutazione.
- [`scripts/analysis/`](scripts/analysis/): estrazione delle confusioni e degli esempi di errore.
- [`scripts/figures/`](scripts/figures/): grafici del confronto e dell'ablation sulle maiuscole.
- [`docs/report/`](docs/report/): [report PDF](docs/report/report.pdf), sorgente LaTeX, bibliografia e figure.
- [`docs/slides/`](docs/slides/): [presentazione PDF](docs/slides/Presentazione_PoS_Modelli_Linguistici.pdf) e sorgente LaTeX.

Per ricompilare i documenti con LaTeX e `latexmk`:

```console
latexmk -cd -pdf docs/report/report.tex
latexmk -cd -xelatex -jobname=Presentazione_PoS_Modelli_Linguistici docs/slides/slides.tex
```

Le slide usano i font Arial e Consolas. Per l'utilizzo di dati e modelli, fare riferimento alle licenze indicate nelle rispettive pagine di origine.
