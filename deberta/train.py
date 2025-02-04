import pandas as pd
import numpy as np
import torch
import wandb
import yaml
import warnings
from datasets import Dataset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_squared_error
from lifelines import KaplanMeierFitter, CoxPHFitter, NelsonAalenFitter
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
warnings.filterwarnings("ignore")

with open("../config.yaml", "r") as file:
    config = yaml.safe_load(file)
wandb_api_key = config["wandb"]["api_key"]

train = pd.read_csv("../data/train.csv")

cat_cols = []
num_cols = []
RMV = ["ID", "efs", "efs_time", "target"]
FEATURES = [c for c in train.columns if not c in RMV]
print(f"There are {len(FEATURES)} FEATURES: {FEATURES}")

for c in FEATURES:
    if train[c].dtype == "object" or train[c].dtype == "category":
        cat_cols.append(c)
    else:
        num_cols.append(c)
print(f"In these features, there are {len(cat_cols)} CATEGORICAL FEATURES: {cat_cols}")


def update_target_with_survival_probabilities(df, method="kaplan", time_col="efs_time", event_col="efs"):
    res = np.zeros(df.shape[0])
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    for train_idx, val_idx in skf.split(df, df["race_group"]):
        X_trn, X_val = df.iloc[train_idx], df.iloc[val_idx]
        if method == "kaplan":
            kmf = KaplanMeierFitter()
            kmf.fit(durations=X_trn[time_col], event_observed=X_trn[event_col])
            res[val_idx] = kmf.survival_function_at_times(X_val[time_col]).values
        elif method == "nelson":
            naf = NelsonAalenFitter()
            naf.fit(durations=X_trn[time_col], event_observed=X_trn[event_col])
            res[val_idx] = -naf.cumulative_hazard_at_times(X_val[time_col]).values
        else:
            data_trn = pd.get_dummies(X_trn, columns=cat_cols, drop_first=True).drop("ID", axis=1)
            data_val = pd.get_dummies(X_val, columns=cat_cols, drop_first=True).drop("ID", axis=1)
            train_data = data_trn.loc[:, data_trn.nunique() > 1]
            valid_data = data_val[train_data.columns]
            cph = CoxPHFitter(penalizer=0.01)
            cph.fit(train_data, duration_col=time_col, event_col=event_col)
            res[val_idx] = cph.predict_partial_hazard(valid_data).values
    df["target"] = res
    df.loc[df[event_col] == 0, "target"] -= 0.15
    return df


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


MAX_LENGTH = 520
MODEL_NAME = "microsoft/deberta-v3-base"
LEARNING_RATE = 5e-5
BATCH_SIZE = 4
ACCUMULATION_STEPS = 8
WARMUP_RATIO = 0.1
EPOCHS = 3
WEIGHT_DECAY = 0.0001
STEPS = 100
SAVE_TOTAL_LIMIT = 10
LR_SCHEDULER = "cosine"


for method in ["kaplan", "nelson", "cox"]:
    train_copy = train.copy()
    train_copy = update_target_with_survival_probabilities(train_copy, method=method, time_col="efs_time", event_col="efs")
    train_copy = update(train_copy)
    train_copy["text"] = train_copy.apply(create_text, axis=1)

    skf = StratifiedKFold(n_splits=10, random_state=42, shuffle=True)
    for i, (train_index, val_index) in enumerate(skf.split(train_copy, train_copy["race_group"])):
        train_copy.loc[val_index, "fold"] = int(i)

    for fold in range(0, 10):
        print(f"Method: {method}, Fold: {fold}")

        train_df = train_copy[train_copy["fold"] != float(fold)]
        train_df = train_df[["text", "target"]].sample(frac=1.0, random_state=42)
        val_df = train_copy[train_copy["fold"] == float(fold)]
        val_df = val_df[["text", "target"]].sample(frac=1.0, random_state=42)
        print(f"Train shape: {train_df.shape}, Val shape: {val_df.shape}")

        train_df.rename(columns={"target": "labels"}, inplace=True)
        val_df.rename(columns={"target": "labels"}, inplace=True)

        tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")
        train_df["len_text"] = train_df["text"].apply(lambda x: len(tokenizer.encode(x, add_special_tokens=False)))
        val_df["len_text"] = val_df["text"].apply(lambda x: len(tokenizer.encode(x, add_special_tokens=False)))
        print(train_df["len_text"].describe())
        print(val_df["len_text"].describe())
        train_df.drop(columns=["len_text"], inplace=True)
        val_df.drop(columns=["len_text"], inplace=True)

        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

        def tokenize(sample):
            return tokenizer(sample["text"], max_length=MAX_LENGTH, truncation=True)

        ds_train = Dataset.from_pandas(train_df)
        ds_val = Dataset.from_pandas(val_df)

        ds_train = ds_train.map(tokenize).remove_columns(["text"])
        ds_val = ds_val.map(tokenize).remove_columns(["text"])

        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME,
            num_labels=1,
            trust_remote_code=True
        )

        class DataCollator:
            def __call__(self, features):
                model_inputs = [
                    {
                        "input_ids": feature["input_ids"],
                        "attention_mask": feature["attention_mask"],
                        "labels": feature["labels"]
                    } for feature in features
                ]
                batch = tokenizer.pad(
                    model_inputs,
                    padding="max_length",
                    max_length=MAX_LENGTH,
                    return_tensors="pt",
                    pad_to_multiple_of=8
                )
                return batch

        def compute_metrics(p):
            preds, labels = p
            preds = preds.astype(np.float32)
            labels = labels.astype(np.float32)
            mse = mean_squared_error(labels, preds)
            return {"mse": mse}

        wandb.login(key=wandb_api_key)
        run = wandb.init(project=f"CIB-{MODEL_NAME.split('/')[-1]}-{method}", job_type="training", anonymous="allow")

        training_args = TrainingArguments(
            output_dir=f"{MODEL_NAME.split('/')[-1]}-{method}/Fold{fold}",
            bf16=True if torch.cuda.is_bf16_supported() else False,
            fp16=False if torch.cuda.is_bf16_supported() else True,
            learning_rate=LEARNING_RATE,
            per_device_train_batch_size=BATCH_SIZE,
            per_device_eval_batch_size=BATCH_SIZE * 2,
            gradient_accumulation_steps=ACCUMULATION_STEPS,
            warmup_ratio=WARMUP_RATIO,
            num_train_epochs=EPOCHS,
            weight_decay=WEIGHT_DECAY,
            do_eval=True,
            evaluation_strategy="steps",
            eval_steps=STEPS,
            save_total_limit=SAVE_TOTAL_LIMIT,
            save_strategy="steps",
            save_steps=STEPS,
            logging_steps=STEPS,
            load_best_model_at_end=True,
            metric_for_best_model="mse",
            greater_is_better=False,
            save_only_model=True,
            lr_scheduler_type=LR_SCHEDULER,
            gradient_checkpointing=False,
            report_to="wandb"
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=ds_train,
            eval_dataset=ds_val,
            tokenizer=tokenizer,
            data_collator=DataCollator(),
            compute_metrics=compute_metrics
        )

        trainer.train()
        wandb.finish()
