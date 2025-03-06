### This Repo is for [Kaggle - CIBMTR - Equity in post-HCT Survival Predictions](https://www.kaggle.com/competitions/equity-post-HCT-survival-predictions)

#### Python Environment

##### 1. Install Packages

```b
pip install --upgrade -r requirements.txt
```

##### 2. Create ``config.yaml`` File

```bash
wandb:
  api_key: "YOUR_WANDB_API_KEY"
huggingface:
  api_key: "YOUR_HUGGINGFACE_API_KEY"
```

#### Prepare Datasets

##### 1. Set Up Kaggle Env

```bash
export KAGGLE_USERNAME="YOUR_KAGGLE_USERNAME"
export KAGGLE_KEY="YOUR_KAGGLE_API_KEY"
```

##### 2. Download Datasets

```bash
sudo apt install unzip
kaggle competitions download -c equity-post-HCT-survival-predictions
unzip equity-post-HCT-survival-predictions.zip
```

#### Submissions

The best final submissions are located in the ``submissions`` folder.

#### Conclusion

- The NLP-based method does not work for this competition.
- Feature selection does not work very well, probably because there is a lot of synthetic data in the training dataset.