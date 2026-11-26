# PoS tagging italiano con encoder e LLM

Fine-tuning di BERT XXL italiano, ELECTRA XXL italiano e Qwen2.5-1.5B-Instruct sul corpus Hugging Face UD Italian-ISDT 2.15, con parole e confini di frase gold.

Il notebook è un'estensione di `BERT_Italian_PoS_UD_2_15.ipynb` del progetto NLP. Ne mantiene l'ordine delle sezioni e la logica di esecuzione: configurazione, dati, tokenizzazione, metriche, modelli, training, valutazione, confronto e analisi degli errori. Il training usa Hugging Face `Trainer`, con le sue barre di avanzamento.

BERT ed ELECTRA aggregano le subword di ciascuna parola con mean pooling; Qwen genera una lista di tag UPOS. L'ablation confronta testo originale e testo in minuscolo per tutti i modelli, con tre seed (`42`, `123`, `456`): 18 training complessivi. La CRF resta un possibile approfondimento.

## Contenuti

- [Notebook](notebooks/Italian_UPOS_AI.ipynb): configurazione, dati, modelli, training, valutazione e analisi degli errori.
- [Analisi](scripts/analysis/extract_error_examples.py): confusioni ed esempi contestuali dalle predizioni salvate.
- [Figure](scripts/figures/plot_results.py): confronto dei modelli e variazioni dovute all'ablation.
- [Relazione](docs/report/report.pdf), con sorgente LaTeX e bibliografia in `docs/report/`.
- [Slide](docs/slides/Presentazione_PoS_Modelli_Linguistici.pdf), con sorgente LaTeX in `docs/slides/`.

Dataset e risultati locali sono conservati in `data/` e `outputs/`, esclusi da Git. I nuovi training non sono ancora stati eseguiti; report e slide descrivono il protocollo, senza riportare come nuovi i risultati NLP.

## Utilizzo

### VS Code e Git

Aprire direttamente la cartella `italian-pos-tagging` in VS Code, anche se si trova dentro un altro workspace:

```console
cd E:\computational-imaging\italian-pos-tagging
code .
```

Il pannello **Controllo codice sorgente** gestisce la repository Git di questa cartella. Le estensioni consigliate sono indicate in `.vscode/extensions.json`; la cartella di lavoro dei notebook è impostata sulla root del progetto.

Aprire `notebooks/Italian_UPOS_AI.ipynb` e scegliere, tramite **Seleziona kernel**, l'ambiente Python con PyTorch e CUDA già configurati. È possibile usare un ambiente esistente esterno alla repository; ambienti virtuali, dataset, cache e risultati degli esperimenti restano esclusi dai commit.

### Notebook e analisi

Aprire il notebook ed eseguire le celle in ordine. Include l'installazione delle dipendenze, il caricamento dei dati e tutte le funzioni necessarie, senza package o configurazioni esterne. Revisioni, parametri e impostazioni degli esperimenti sono definiti nelle prime celle.

Come nel notebook NLP, eseguire la cella della sezione Training avvia direttamente i run configurati: 18 con le impostazioni iniziali. I training già completati vengono verificati e riutilizzati. Per cambiare il piano, modificare sistemi attivi e seed nella configurazione iniziale. Occorrono PyTorch con CUDA e una GPU adatta al fine-tuning; Qwen usa QLoRA a 4 bit. Le impostazioni iniziali vanno verificate sulle risorse disponibili.

Il `Trainer` valuta il development e salva un checkpoint a ogni epoca, quindi ricarica quello con la macro-F1 totale UPOS più alta. Per Qwen la metrica deriva dalla generazione delle risposte. La sezione Evaluation, successiva al training, ricarica i modelli selezionati e produce metriche e predizioni sul test. Configurazioni, argomenti del `Trainer` e versioni utilizzate vengono salvati insieme ai risultati. Per esperimenti diversi scegliere una nuova directory di output.

La pipeline corrente salva i run in `outputs/upos_trainer_v1/`. I risultati delle esecuzioni precedenti restano conservati in `outputs/upos_v1/`.

Per analizzare i risultati salvati in `outputs/upos_trainer_v1/`, senza ripetere training o inferenza:

```console
python scripts/analysis/extract_error_examples.py
python scripts/figures/plot_results.py
```

I grafici richiedono Matplotlib (`pip install matplotlib`) e vengono salvati in `docs/report/figures/`; confusioni ed esempi in `outputs/upos_trainer_v1/analysis/`.

Per ricompilare i documenti con LaTeX:

```console
latexmk -cd -pdf docs/report/report.tex
latexmk -cd -xelatex -jobname=Presentazione_PoS_Modelli_Linguistici docs/slides/slides.tex
```

Le slide usano i font Arial e Consolas. Il dataset mantiene la licenza CC BY-NC-SA 3.0 indicata dagli autori; i tre checkpoint mantengono le rispettive licenze Hugging Face.
