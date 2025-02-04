from pathlib import Path

kaggle = False


class Config:
    if not kaggle:
        train_path = "./data/train.csv"
        test_path = "./data/test.csv"
        subm_path = "./data/sample_submission.csv"
    else:
        train_path = Path("/kaggle/input/equity-post-HCT-survival-predictions/train.csv")
        test_path = Path("/kaggle/input/equity-post-HCT-survival-predictions/test.csv")
        subm_path = Path("/kaggle/input/equity-post-HCT-survival-predictions/sample_submission.csv")
