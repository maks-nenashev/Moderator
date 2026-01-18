import json
from keras.preprocessing.text import Tokenizer

# Пример входных данных (позже заменим реальным датасетом)
texts = [
    "You are stupid",
    "Free money!!! Click now",
    "I will kill you",
    "Hot girls near you",
    "Normal neutral message"
]

with open("artifacts/config.json", "r") as f:
    config = json.load(f)

tok_cfg = config["tokenizer"]

tokenizer = Tokenizer(
    num_words=tok_cfg["num_words"],
    oov_token=tok_cfg["oov_token"],
    lower=tok_cfg["lower"],
    filters=tok_cfg["filters"]
)

tokenizer.fit_on_texts(texts)

# Сохраняем tokenizer
tokenizer_json = tokenizer.to_json()
with open("artifacts/tokenizer.json", "w") as f:
    f.write(tokenizer_json)

print("Tokenizer saved to artifacts/tokenizer.json")
