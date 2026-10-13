# Dataset

This folder holds the dataset used for training. The data file itself is **not
committed to Git** (see `.gitignore`); you download it with one command.

## Source

**SMS Spam Collection** – UCI Machine Learning Repository
https://archive.ics.uci.edu/dataset/228/sms+spam+collection

It contains about 5,574 English SMS messages labeled `ham` (legitimate) or `spam`.

## Usage terms

The UCI repository lists this dataset under the
**Creative Commons Attribution 4.0 International (CC BY 4.0)** license, which
allows use and sharing as long as you give credit. Please check the dataset
page above for the current terms before redistributing it.

Attribution: Almeida, T. & Hidalgo, J. (2011). *SMS Spam Collection.* UCI
Machine Learning Repository.

## Setup

From the project root:

```
python download_data.py
```

This downloads the official zip from UCI and saves `data/SMSSpamCollection`.

### Manual setup (if the download fails)

1. Open the dataset page above and download the zip.
2. Extract it and copy the file `SMSSpamCollection` into this `data/` folder.

Other common formats also work. The loader accepts any of these files in `data/`:
`SMSSpamCollection`, `spam.csv` (Kaggle version, columns `v1`/`v2`),
`sms.tsv`, or a file with a header such as `label,message` / `category,text`.
