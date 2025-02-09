import pandas as pd
import numpy as np
import itertools
import json
import warnings
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from lifelines import KaplanMeierFitter, NelsonAalenFitter
from tqdm import tqdm
from lifelines.utils import concordance_index
warnings.filterwarnings("ignore")

train = pd.read_csv("../data/train.csv")

cat_cols = []
num_cols = []
RMV = ["ID", "efs", "efs_time", "target"]
FEATURES = [c for c in train.columns if not c in RMV]
print(len(FEATURES))

for c in FEATURES:
    if train[c].dtype == "object" or train[c].dtype == "category":
        cat_cols.append(c)
    else:
        num_cols.append(c)
print(f"In these features, there are {len(cat_cols)} CATEGORICAL FEATURES: {cat_cols}")
print(train.shape)


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
            print("Method Error")
    df["target"] = res
    df.loc[df[event_col] == 0, "target"] -= 0.15
    return df


def update(df, cat_cols):
    for c in cat_cols:
        df[c] = df[c].astype(str).fillna("Unknown").astype("category")
    for c in num_cols:
        if df[c].dtype == "float64":
            df[c] = df[c].fillna(0).astype("float32")
        if df[c].dtype == "int64":
            df[c] = df[c].fillna(0).astype("int32")
    return df


def score(solution: pd.DataFrame, submission: pd.DataFrame, row_id_column_name: str) -> float:
    del solution[row_id_column_name]
    del submission[row_id_column_name]

    event_label = "efs"
    interval_label = "efs_time"
    prediction_label = "prediction"

    for col in submission.columns:
        if not pd.api.types.is_numeric_dtype(submission[col]):
            raise ValueError(f"Submission column {col} must be a number")

    merged_df = pd.concat([solution, submission], axis=1)
    merged_df.reset_index(inplace=True)
    merged_df_race_dict = dict(merged_df.groupby(["race_group"]).groups)
    metric_list = []
    for race in merged_df_race_dict.keys():
        indices = sorted(merged_df_race_dict[race])
        merged_df_race = merged_df.iloc[indices]
        c_index_race = concordance_index(
            merged_df_race[interval_label],
            -merged_df_race[prediction_label],
            merged_df_race[event_label]
        )
        metric_list.append(c_index_race)
    return float(np.mean(metric_list) - np.sqrt(np.var(metric_list)))


train = update_target_with_survival_probabilities(train, method="kaplan", time_col="efs_time", event_col="efs")
train = update(train, cat_cols)
print(train.shape)

for col1, col2 in tqdm(itertools.combinations(cat_cols, 2), total=int(len(cat_cols) * (len(cat_cols) - 1) / 2)):
    new_feature_name = f"{col1}+{col2}"
    train[new_feature_name] = train[col1].astype(str) + "_" + train[col2].astype(str)

combined_cat_cols = [f"{col1}+{col2}" for col1, col2 in itertools.combinations(cat_cols, 2)]

all_cat_cols = cat_cols + combined_cat_cols
for col in all_cat_cols:
    le = LabelEncoder()
    train[col] = le.fit_transform(train[col].astype(str))

print(train.shape)

# use only original features to train the model
TARGET = "target"
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
fold_scores = []
lgb_params = {
    "objective": "regression",
    "num_iterations": 1000,
    "learning_rate": 0.03,
    "metric": "rmse",
    "max_depth": 8,
    "num_leaves": 15,
    "device": "cpu",
    "verbose": -1,
    "seed": 42
}
for fold, (train_idx, val_idx) in enumerate(skf.split(train, train["race_group"])):
    print(f"Training fold {fold + 1}")

    X_train, y_train = train.iloc[train_idx][FEATURES], train.iloc[train_idx][TARGET]
    X_val, y_val = train.iloc[val_idx][FEATURES], train.iloc[val_idx][TARGET]

    model = lgb.LGBMRegressor(**lgb_params)
    model.fit(X_train, y_train)

    y_true_fold = train.iloc[val_idx][["ID", "efs", "efs_time", "race_group"]].copy()
    y_pred_fold = train.iloc[val_idx][["ID"]].copy()
    y_pred_fold["prediction"] = model.predict(X_val)
    fold_score = score(y_true_fold, y_pred_fold, "ID")
    fold_scores.append(fold_score)

print(f"Average C-INDEX across 10 folds: {np.mean(fold_scores)}")
original_fold_scores = np.mean(fold_scores)

# add new features one by one
comparison = {}

for new_feature in tqdm(combined_cat_cols, total=len(combined_cat_cols)):
    print("Add:", new_feature)
    TEMP_FEATURE = FEATURES + [new_feature]
    TARGET = "target"
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    fold_scores = []
    lgb_params = {
        "objective": "regression",
        "num_iterations": 1000,
        "learning_rate": 0.03,
        "metric": "rmse",
        "max_depth": 8,
        "num_leaves": 15,
        "device": "cpu",
        "verbose": -1,
        "seed": 42
    }
    for fold, (train_idx, val_idx) in enumerate(skf.split(train, train["race_group"])):
        print(f"Training fold {fold + 1}")

        X_train, y_train = train.iloc[train_idx][TEMP_FEATURE], train.iloc[train_idx][TARGET]
        X_val, y_val = train.iloc[val_idx][TEMP_FEATURE], train.iloc[val_idx][TARGET]

        model = lgb.LGBMRegressor(**lgb_params)
        model.fit(X_train, y_train)

        y_true_fold = train.iloc[val_idx][["ID", "efs", "efs_time", "race_group"]].copy()
        y_pred_fold = train.iloc[val_idx][["ID"]].copy()
        y_pred_fold["prediction"] = model.predict(X_val)
        fold_score = score(y_true_fold, y_pred_fold, "ID")
        fold_scores.append(fold_score)

    print(f"Average C-INDEX across 10 folds after adding {new_feature}: {np.mean(fold_scores)}")
    temp_fold_scores = np.mean(fold_scores)
    comparison[new_feature] = temp_fold_scores - original_fold_scores

    with open("category_comparison.json", "w") as json_file:
        json.dump(comparison, json_file, indent=4)
