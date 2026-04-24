# Model card: IMDB encoder classifier

## Intended use

Educational and portfolio demonstration of a Transformer **encoder** trained from scratch for binary sentiment on the IMDB review dataset.

## Training data

- [Hugging Face `imdb`](https://huggingface.co/datasets/imdb) train split for training; test split for reported accuracy.
- Whitespace tokenization with a fixed vocabulary built from training split word frequencies.

## Metrics

Portfolio run: `outputs/portfolio/clf/metrics.csv`, Hydra config in `outputs/portfolio/clf/.hydra/config.yaml`.
Best validation accuracy was epoch 10.

| Metric | Value |
|--------|------:|
| Test accuracy | 0.8666 |
| Macro F1 | 0.8666 |
| ROC-AUC | Not logged in CSV; ROC plot artifact exists at `docs/assets/portfolio/roc.png` |

## Limitations

- Tokenizer is simple word splitting; not comparable to subword models used in production.
- Demo Gradio tab requires the `vocab.json` saved beside the classifier checkpoint by `scripts/train_classifier.py`.
- Reported metric is from the Hugging Face IMDB test split used as this repo's validation split, not a separate hidden benchmark.

## Ethical considerations

Sentiment models can reflect dataset biases (genre, demographics, toxic language in reviews). Do not use this checkpoint for high-stakes decisions without auditing data and errors.
