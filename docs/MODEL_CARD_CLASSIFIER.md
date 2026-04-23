# Model card: IMDB encoder classifier

## Intended use

Educational and portfolio demonstration of a Transformer **encoder** trained from scratch for binary sentiment on the IMDB review dataset.

## Training data

- [Hugging Face `imdb`](https://huggingface.co/datasets/imdb) train split for training; test split for reported accuracy.
- Whitespace tokenization with a fixed vocabulary built from training split word frequencies.

## Metrics

Fill after training (Hydra output directory contains `metrics.csv` and plots):

| Metric | Value |
|--------|------:|
| Test accuracy | _TBD_ |
| Macro F1 | _TBD_ |
| ROC-AUC | _TBD_ |

## Limitations

- Tokenizer is simple word splitting; not comparable to subword models used in production.
- Demo Gradio tab uses a placeholder vocabulary unless you ship a vocabulary file consistent with your checkpoint.

## Ethical considerations

Sentiment models can reflect dataset biases (genre, demographics, toxic language in reviews). Do not use this checkpoint for high-stakes decisions without auditing data and errors.
