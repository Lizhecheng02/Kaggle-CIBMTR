import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

test = pd.read_csv("../data/test.csv")

cat_cols = []
num_cols = []
RMV = ["ID", "efs", "efs_time", "target"]
FEATURES = [c for c in test.columns if not c in RMV]
print(f"There are {len(FEATURES)} FEATURES: {FEATURES}")

for c in FEATURES:
    if test[c].dtype == "object" or test[c].dtype == "category":
        cat_cols.append(c)
    else:
        num_cols.append(c)
print(f"In these features, there are {len(cat_cols)} CATEGORICAL FEATURES: {cat_cols}")


def update(df):
    global cat_cols
    for c in cat_cols:
        df[c] = df[c].astype(str).fillna("Unknown").astype("category")
    for c in num_cols:
        if df[c].dtype == "float64":
            df[c] = df[c].fillna(0).astype("float32")
        if df[c].dtype == "int64":
            df[c] = df[c].fillna(0).astype("int32")
    j_ch = ',[]{}:"\\<'
    for ch in j_ch:
        for c in cat_cols:
            df[c] = df[c].apply(lambda x: str(x).replace(ch, ""))
    return df


def create_text(row):
    text = []
    for col in FEATURES:
        text.append(f"{col}: {row[col]}")
    return "\n".join(text)


test = update(test)
test["text"] = test.apply(create_text, axis=1)

model_path = ""

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

input_text = ["This is a sample sentence for the model."]
inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
with torch.no_grad():
    outputs = model.deberta(**inputs)
    pooled_output = model.pooler(outputs.last_hidden_state)
print(pooled_output.shape)
