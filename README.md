# Data Engineering II (DS 5110) Final Project - Group 16
# Predicting Political Party Affiliation from Campaign Finance Data

Authors:
Ismael AL-Hadjrami, Cecelia Auerswald, Tricia Hicks, Carlos Revilla

## Project Overview

This project investigates two approaches to predicting political party affiliation using machine learning and campaign finance data:

1. **Aggregate-Level Model**: Predicts party affiliation of candidates/recipients using aggregate fundraising data from Stanford DIME and Federal Election Commission (FEC) sources.

2. **Transaction-Level Model**: Predicts whether an individual campaign contribution went to a Democratic, Republican, or Other-party recipient using only donor-level and transaction-level features from Stanford DIME's itemized contribution database.

Both approaches build and compare multiple machine learning models (Logistic Regression, Random Forest, Naive Bayes) using Apache Spark for distributed processing.

### Research Questions

**Aggregate-Level Model (EDA.ipynb):** Can we predict the political party affiliation of candidates/recipients using aggregate fundraising data from DIME and FEC?

**Transaction-Level Model (Final_Project.ipynb):** Can we predict whether an individual campaign contribution went to a Democratic, Republican, or Other-party recipient, using only donor-level and transaction-level features?


## Project Structure
```
group16_project_ds5110/
├── EDA.ipynb # Aggregate-level model (candidate/recipient prediction)
├── Final_Project.ipynb # Transaction-level model (donor/contribution prediction)
├── README.md # Project documentation
├── requirements.txt # Python package dependencies
├── .gitignore # Git ignore rules
├── data/
│ └── dime_fec_final_project/ # Data directory (ignored by git)
│ ├── contrib/ # Contribution data
│ ├── fec/ # FEC data
│ └── processed/ # Processed Parquet files
└── final_project_outputs/ # Model outputs and evaluation results
```
## Models Implemented

### 1. Aggregate-Level Model (`EDA.ipynb`)

**Purpose**: Predict party affiliation of candidates/recipients using aggregate fundraising data

**Data Sources**:
- Stanford DIME recipients/candidates aggregate table
- FEC Candidate Summary
- FEC Candidate Master
- FEC Candidate-Committee Linkage
- FEC Committee Master / Committee Summary

**Features**:
- DIME aggregate financial totals (receipts, contributions, etc.)
- FEC candidate financial metrics
- FEC committee statistics
- Ideology scores (cfscore)

**Models**:
1. Benchmark Logistic Regression (limited features)
2. Full Multinomial Logistic Regression
3. Random Forest

### 2. Transaction-Level Model (`Final_Project.ipynb`)

**Purpose**: Predict recipient party from individual contribution characteristics

**Data Source**:
- Stanford DIME itemized `contribDB` file (2012 cycle, 10% sample)

**Features**:
- **Numeric (1)**: Log-transformed contribution amount
- **Categorical (9)**:
  - Occupation group (250 buckets via fuzzy matching + synonym mapping)
  - Transaction type
  - Seat primary (federal/state/local)
  - Seat secondary (chamber/committee type)
  - Contributor state
  - Recipient state
  - Election type
  - Contributor gender
  - Contributor type (Individual/Committee)

**Key Preprocessing Steps**:
1. **Occupation bucketing**: Free-text occupation collapsed to 250 buckets using:
   - Top-N frequency selection
   - Fuzzy matching (Levenshtein distance)
   - Synonym mapping
   - Fit on training data only to prevent leakage

2. **Donor-grouped split**: 80/20 train/test split by donor ID (same donor never appears in both sets)

3. **Log transformation**: `log1p(amount)` to handle right-skewed distribution

**Models**:
1. Benchmark Logistic Regression (amount only)
2. Full Multinomial Logistic Regression (all features)
3. Random Forest (all features)
4. Naive Bayes (all features)

## Major Functionality

The notebooks contain extensive data processing, feature engineering, and modeling functions implemented as Jupyter notebook cells. Key functionality includes:

### Data Processing

- Download data files from URLs if not present locally
- Decompress .gz files to .csv for Spark reading
- Read CSV with all columns as strings and deduplicate names
- Build the main transaction-level DataFrame
- Split data by donor ID to prevent leakage

### Feature Engineering

- Clean raw occupation text (uppercase, remove punctuation, collapse spaces)
- Map synonyms to canonical occupation labels
- Create fuzzy matching rules using Levenshtein distance
- Full occupation bucketing pipeline (fit on train, apply to all)
- Create Spark ML pipeline stages (StringIndexer, OneHotEncoder, VectorAssembler)

### Modeling & Evaluation

- Compute accuracy, F1, precision, recall, AUROC
- Generate confusion matrix as a Spark DataFrame
- Summarize cross-validation results
- Compute ROC curve points for one-vs-rest classification
- Extract Random Forest feature importances
- Extract Logistic Regression coefficients
- Group one-hot encoded feature levels back to original features

## Getting Started

### Prerequisites

- Python 3.8+
- Apache Spark 3.4+ (local mode or cluster)
- 16GB+ RAM recommended
- 50GB+ free disk space for data

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/group16_project_ds5110.git
cd group16_project_ds5110
```

2. Install Python dependencies:
```bash
pip install -r requirements.txt
```

### Running the Notebooks

1. Start Jupyter:
```bash
jupyter notebook
```

2. Open and run the notebooks:
   - `EDA.ipynb` - Aggregate-level model (candidate/recipient prediction)
   - `Final_Project.ipynb` - Transaction-level model (donor/contribution prediction)
