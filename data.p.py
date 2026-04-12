from datasets import load_dataset

dataset = load_dataset("imdb")
print(dataset)
print(dataset["train"][0])