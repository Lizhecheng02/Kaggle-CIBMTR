import pandas as pd
import numpy as np
import torch
import yaml
import warnings
from sklearn.model_selection import StratifiedKFold
from lifelines import KaplanMeierFitter, NelsonAalenFitter
from transformers import AutoTokenizer, AutoModelForSequenceClassification, BitsAndBytesConfig
from peft import PeftModel
from tqdm import tqdm
warnings.filterwarnings("ignore")

with open("../config.yaml", "r") as file:
    config = yaml.safe_load(file)
wandb_api_key = config["wandb"]["api_key"]
huggingface_api_key = config["huggingface"]["api_key"]

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
            print("Method not supported")
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


template_full = """
The patient is {age_at_hct} years old at the time of hematopoietic cell transplantation (HCT), with a donor aged {donor_age} years. The patient's disease history includes {prim_disease_hct}, along with a {prior_tumor} history of tumors.
Their health profile is marked by a {dri_score} DRI score, presence of {psych_disturb} psychological disturbances, and a {cyto_score} cytogenetic score. The patient also suffers from chronic conditions including {diabetes} diabetes, {arrhythmia} arrhythmia, {renal_issue} renal issues, and {pulm_severe} severe pulmonary problems.
In terms of immunological status, the patient exhibits {hla_match_c_high} high HLA matching for C, {hla_high_res_8} high-resolution 8, and {hla_match_dqb1_high} high HLA-DQB1 matching, as well as {hla_high_res_6} high-resolution 6. Additional immunological markers include {tbi_status} TBI status, {hla_low_res_6} low HLA-C resolution 6, and {cmv_status} CMV status.
Treatment history reveals the administration of {rituximab} rituximab, with the graft type classified as {graft_type}. The patient has a {vent_hist} history of ventilation, alongside {hla_match_drb1_low} low HLA-DRB1 matching and {hla_match_dqb1_low} low HLA-DQB1 matching.
Further health assessments include {cyto_score_detail} cytogenetic score details and {conditioning_intensity} conditioning intensity. The patient's demographic details note a race group of {race_group}, ethnicity of {ethnicity}, and {sex_match} sex match with the donor. Functional status is indicated by a {karnofsky_score} Karnofsky score and a comorbidity score of {comorbidity_score}, reflecting overall health.
The patient's condition is impacted by {hepatic_severe} severe hepatic issues, {hepatic_mild} mild hepatic issues, and {pulm_moderate} moderate pulmonary issues. They also have {mrd_hct} MRD status at HCT, {tce_div_match} TCE diversity match, and {tce_match} TCE match.
Treatment specifics include a {melphalan_dose} melphalan dose, {in_vivo_tcd} in-vivo T-cell depletion (TCD) treatment, and {gvhd_proph} graft-versus-host disease (GVHD) prophylaxis. Disease progression is influenced by {rheum_issue} rheumatic issues, {obesity} obesity, and {cardiac} cardiac health.
The patient underwent HCT in {year_hct}, with a treatment strategy that includes {prod_type} production type and {tce_imm_match} TCE immune match. Notably, {donor_related} donor-related issues have been observed.
Additional HLA details show {hla_high_res_10} high-resolution 10, {hla_match_c_low} low HLA matching for C, {hla_match_a_high} high HLA-A matching, {hla_match_b_low} low HLA-B matching, {hla_match_a_low} low HLA-A matching, and {hla_match_b_high} high HLA-B matching. Treatment history also notes {peptic_ulcer} peptic ulcer, {hla_low_res_8} low HLA resolution 8, and {hla_low_res_10} low HLA resolution 10. 
Finally, the patient has {hla_nmdp_6} high-resolution matching for NMDP and {hla_match_drb1_high} high HLA-DRB1 matching, rounding out a comprehensive overview of their medical profile.
"""


def fill_template(row):
    return template_full.format(
        age_at_hct=row["age_at_hct"],
        donor_age=row["donor_age"],
        prim_disease_hct=row["prim_disease_hct"],
        prior_tumor=row["prior_tumor"],
        dri_score=row["dri_score"],
        psych_disturb=row["psych_disturb"],
        cyto_score=row["cyto_score"],
        diabetes=row["diabetes"],
        arrhythmia=row["arrhythmia"],
        renal_issue=row["renal_issue"],
        pulm_severe=row["pulm_severe"],
        hla_match_c_high=row["hla_match_c_high"],
        hla_high_res_8=row["hla_high_res_8"],
        hla_match_dqb1_high=row["hla_match_dqb1_high"],
        hla_high_res_6=row["hla_high_res_6"],
        tbi_status=row["tbi_status"],
        hla_low_res_6=row["hla_low_res_6"],
        cmv_status=row["cmv_status"],
        rituximab=row["rituximab"],
        graft_type=row["graft_type"],
        vent_hist=row["vent_hist"],
        hla_match_drb1_low=row["hla_match_drb1_low"],
        hla_match_dqb1_low=row["hla_match_dqb1_low"],
        cyto_score_detail=row["cyto_score_detail"],
        conditioning_intensity=row["conditioning_intensity"],
        race_group=row["race_group"],
        ethnicity=row["ethnicity"],
        sex_match=row["sex_match"],
        karnofsky_score=row["karnofsky_score"],
        comorbidity_score=row["comorbidity_score"],
        hepatic_severe=row["hepatic_severe"],
        hepatic_mild=row["hepatic_mild"],
        pulm_moderate=row["pulm_moderate"],
        mrd_hct=row["mrd_hct"],
        tce_div_match=row["tce_div_match"],
        tce_match=row["tce_match"],
        melphalan_dose=row["melphalan_dose"],
        in_vivo_tcd=row["in_vivo_tcd"],
        gvhd_proph=row["gvhd_proph"],
        rheum_issue=row["rheum_issue"],
        obesity=row["obesity"],
        cardiac=row["cardiac"],
        year_hct=row["year_hct"],
        prod_type=row["prod_type"],
        tce_imm_match=row["tce_imm_match"],
        donor_related=row["donor_related"],
        hla_high_res_10=row["hla_high_res_10"],
        hla_match_c_low=row["hla_match_c_low"],
        hla_match_a_high=row["hla_match_a_high"],
        hla_match_b_low=row["hla_match_b_low"],
        hla_match_a_low=row["hla_match_a_low"],
        hla_match_b_high=row["hla_match_b_high"],
        peptic_ulcer=row["peptic_ulcer"],
        hla_low_res_8=row["hla_low_res_8"],
        hla_low_res_10=row["hla_low_res_10"],
        hla_nmdp_6=row["hla_nmdp_6"],
        hla_match_drb1_high=row["hla_match_drb1_high"]
    )


test = update(test)
test["text"] = test.apply(fill_template, axis=1)

MODEL_NAME = "google/gemma-2-2b"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=huggingface_api_key)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16
)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    num_labels=1,
    trust_remote_code=True,
    token=huggingface_api_key,
    device_map="auto"
)
print(model.config.pad_token_id)
model.config.pad_token_id = model.config.eos_token_id
print(model.config.pad_token_id)

lora_adapter_path = ""
lora_model = PeftModel.from_pretrained(model=model, model_id=lora_adapter_path)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
test_texts = test["text"].to_list()
mean_features = []
with torch.no_grad():
    for test_text in tqdm(test_texts, total=len(test_texts)):
        features = []
        inputs = tokenizer(test_text, return_tensors="pt", padding=True, truncation=True, max_length=684).to(DEVICE)
        outputs = lora_model(**inputs)
        features = np.mean(features, axis=0)
        mean_features.append(features)
mean_features = np.vstack(mean_features)
