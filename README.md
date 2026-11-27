# PoS tagging italiano con BERT, ELECTRA e GePpeTto

Progetto di **Intelligenza Artificiale** sul fine-tuning di modelli linguistici per il *Part-of-Speech tagging* italiano: assegnare a ogni parola la sua categoria grammaticale nel contesto della frase, usando le etichette Universal POS (UPOS).

```text
Il   gatto   dorme   .
DET  NOUN    VERB    PUNCT
```

L'obiettivo è confrontare **BERT XXL**, **ELECTRA XXL** e **GePpeTto** sullo stesso corpus, adattando tutti e tre alla classificazione per parola in 17 categorie. Il progetto studia l'effetto delle maiuscole e comprende una seconda fase che aggiunge la predizione degli attributi morfologici **FEATS**, come genere, numero e tempo verbale.

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
| GePpeTto | [`LorenzoDeMattei/GePpeTto`](https://huggingface.co/LorenzoDeMattei/GePpeTto) | Fine-tuning completo con classificazione per parola |

**Tutti e tre i modelli** usano **mean pooling** per ottenere un vettore per parola dalle rappresentazioni delle sue subword, seguito da dropout e classificatore a 17 classi. Se una parola fosse divisa in `in` + `utile`, usare solo il primo frammento significherebbe conservare una sola delle rappresentazioni: la media combina anche quelle delle altre parti della parola.

**GePpeTto** usa la classe ufficiale [`GPT2ForTokenClassification`](https://huggingface.co/docs/transformers/v4.57.1/en/model_doc/gpt2#transformers.GPT2ForTokenClassification), con il pooling per parola aggiunto dal progetto. Il backbone resta **causale**, mentre BERT ed ELECTRA possono usare anche il contesto successivo.

Nella fase **UPOS + FEATS**, le stesse rappresentazioni alimentano anche 14 teste morfologiche, una per attributo. Valori e categorie si ricavano dal training; ogni testa comprende `<NONE>` per l'attributo assente.

L'implementazione usa **PyTorch** e l'ecosistema **Hugging Face**: Transformers e `Trainer` per i modelli e il training, Datasets per il corpus.

## Protocollo sperimentale

Il piano comprende **due ablation separate**: 18 run UPOS confrontano testo originale e minuscolo; **9 nuovi training UPOS + FEATS usano soltanto il testo originale**. Si mantengono tre modelli e tre seed (`42`, `123`, `456`), per **27 esecuzioni complessive**. Ogni run FEATS riparte dal checkpoint UPOS originale dello stesso modello e seed, conservandone i pesi e aggiungendo le teste morfologiche.

Si usa fine-tuning completo per **10 epoche per fase**, batch di training **8**, accumulo dei gradienti **1**, batch di valutazione **16** e **AdamW** con learning rate **2e-5**. La fase congiunta avvia un nuovo ottimizzatore e scheduler. I pesi restano in FP32, con precisione mista BF16 sulle GPU compatibili.

- **Ablation delle maiuscole, solo UPOS:** ogni modello viene addestrato separatamente nelle due condizioni. La trasformazione si applica a training, development e test, conservando ordine delle parole ed etichette.
- **Selezione del checkpoint:** nella prima fase si massimizza la macro-F1 UPOS totale sul development; nella seconda la media tra questa e la macro-F1 FEATS sulle coppie presenti. Il test viene usato per la valutazione finale.
- **Valutazione:** accuracy e macro-F1 UPOS, uguaglianza dell'intero insieme FEATS, macro-F1 delle coppie attributo-valore e correttezza congiunta UPOS + FEATS, con variabilità tra seed.
- **Analisi degli errori:** confusioni tra categorie ed esempi nel contesto della frase, ricavati dalle predizioni salvate.

Nel test sono presenti 15 dei 17 tag UPOS: la macro-F1 totale include due classi senza supporto, con F1 zero e massimo teorico **15/17**. La macro-F1 sulle classi presenti facilita la lettura del confronto. Per FEATS, `<NONE>` è escluso dalla macro-F1.

La fase UPOS è completata e verificata sulle predizioni dei 18 run. Accuracy sul test (%), media ± deviazione standard campionaria sui tre seed:

| Modello | Testo originale | Testo minuscolo |
|---|---:|---:|
| BERT XXL | 98,86 ± 0,10 | 98,43 ± 0,10 |
| ELECTRA XXL | 98,90 ± 0,10 | 98,67 ± 0,03 |
| GePpeTto | 97,57 ± 0,17 | 97,04 ± 0,44 |

ELECTRA ha la maggiore accuracy media sul testo originale; BERT la maggiore macro-F1. Gli scarti sono descrittivi, senza un vincitore assoluto. Le medie diminuiscono con il minuscolo, con variazioni non uniformi tra seed.

I risultati FEATS richiedono i nuovi training e restano separati. Il [report](docs/report/report.pdf) approfondisce risultati, loss, metriche e limiti. Le variazioni UPOS tra fasi comprenderanno anche la prosecuzione del training e il diverso criterio di selezione.

## Utilizzo

Clonare la repository:

```console
git clone https://github.com/pierreerrico/italian-pos-tagging.git
cd italian-pos-tagging
```

1. Preparare un ambiente Python con Jupyter e PyTorch con supporto CUDA.
2. Aprire [`notebooks/Italian_UPOS_AI.ipynb`](notebooks/Italian_UPOS_AI.ipynb) e selezionare il kernel dell'ambiente. La prima cella installa le altre dipendenze.
3. Controllare `TASK_STAGE`, modelli attivi, seed e parametri, quindi eseguire le celle in ordine. Per partire da zero, usare prima `TASK_STAGE = "upos"`; con i checkpoint UPOS disponibili, scegliere `"upos_feats"`, il valore predefinito.

La cella di training avvia 18 run con `"upos"` oppure 9 con `"upos_feats"` e riutilizza quelli completati e compatibili. La fase FEATS verifica il checkpoint UPOS originale genitore prima di ogni nuovo training. Per impostazioni diverse, scegliere una nuova directory di output.

Configurazioni, checkpoint, metriche e predizioni sono separati in `outputs/upos_trainer_v1/` e `outputs/upos_feats_trainer_v1/`. I run UPOS esistenti restano riutilizzabili. Il riepilogo della fase congiunta include `stage_paired_deltas.csv` con le variazioni UPOS appaiate. Dati locali e output sono esclusi da Git.

### Analisi dei risultati

Dopo aver prodotto le predizioni e il riepilogo nel notebook, gli script consentono di estrarre gli errori e generare i grafici senza ripetere training o inferenza:

```console
python scripts/analysis/extract_error_examples.py
python scripts/figures/plot_results.py
```

Questi comandi analizzano per impostazione predefinita la fase UPOS. Il notebook produce anche metriche per attributo e coppia FEATS sui 9 run con testo originale. I grafici UPOS richiedono Matplotlib (`pip install matplotlib`) e il riepilogo completo dei 18 run; vengono salvati in `docs/report/figures/`.

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
