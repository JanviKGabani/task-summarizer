# Task Summarizer

Fine-tunes Hugging Face BART (`facebook/bart-large-cnn`) to turn a list of assigned tasks into a short per-employee summary. Built during an AI/ML internship at Cybersecurity Umbrella.

**Portfolio:** [janvikgabani/github.io](https://janvikgabani.github.io)

## What it does

Takes rows of `employee name · task description · expected hours` and generates a sentence summarising what each person has on. The intended use was a daily digest, so a manager could read one line per person instead of scanning a ticket queue.

## Method

1. **Input construction** — each task becomes `"<description> (Estimated Time: <n> hours)"`, with the target summary built from the employee name and their tasks.
2. **Model** — `facebook/bart-large-cnn`, an encoder-decoder already tuned for summarisation, fine-tuned further on the task data.
3. **Training** — AdamW, mixed precision (`autocast` + `GradScaler`) with gradient accumulation, so the large model fits in a single Colab GPU's memory.
4. **Generation** — beam search, 4 beams, max 128 tokens.

## Honest note on the metric

The script reports "accuracy" as the share of generated summaries that **exactly string-match** the reference. That is a poor measure for summarisation: a paraphrase that means the same thing scores zero, so the number understates quality and says nothing useful about it.

The right metrics are **ROUGE-1/2/L** for overlap, or BERTScore for semantic similarity. Treat the accuracy figure in the output as a placeholder rather than a result.

## Status

Training code only. The fine-tuned weights live in Google Drive and are not committed, so this will not run end to end without:

- `tasks_datasetnew.csv` — the task dataset (internal, not published)
- a GPU, for fine-tuning a 400M-parameter model

Paths in the script still point at Colab Drive mounts; change them to run elsewhere.

## Files

| Path | What it is |
|---|---|
| `train_summarizer.py` | Data prep, fine-tuning loop, saving and evaluation (exported from the original Colab notebook) |

## Related

Part of the same internship as [task-prioritization](https://github.com/JanviKGabani/task-prioritization), which ranks the tasks this one summarises.
