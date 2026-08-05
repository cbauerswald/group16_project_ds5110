# Data Engineering II (DS 5110) Final Project - Group 16
# Predicting Contribution Party from Donor-Level Features

Authors:
Ismael AL-Hadjrami, Cecelia Auerswald, Tricia Hicks, Carlos Revilla

## Project Overview

This project investigates whether we can predict the political party (Democrat, Republican, or Other) of a campaign contribution recipient using only donor-level and transaction-level features. We build and compare multiple machine learning models using Stanford DIME (Database on Ideology, Money in Politics, and Elections) campaign contribution data, with additional enrichment from Federal Election Commission (FEC) data.

### Research Question

**Can we predict whether an individual campaign contribution went to a Democratic, Republican, or Other-party recipient, using only donor-level and transaction-level features?**


## Project Structure
```
group16_project_ds5110/
├── EDA.ipynb # Aggregate-level model (candidate/recipient prediction)
├── Final_Project.ipynb # Transaction-level model (donor/contribution prediction)
├── README.md # Project documentation
├── requirements.txt # Python package dependencies
├── .gitignore # Git ignore rules
└── data/
    └── dime_fec_final_project/ # Data directory (ignored by git)
        ├── contrib/ # Contribution data
        ├── fec/ # FEC data
        └── processed/ # Processed Parquet files
```
## Models Implemented

### 1. Aggregate-Level Model (`EDA.ipynb`)

**Purpose**: Predict party affiliation of candidates/recipients using aggregate fundraising data

**Data Sources**:
- Stanford DIME recipients/candidates aggregate table (1979-2024)
- FEC Candidate Summary (2018, 2020, 2022, 2024 cycles)
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

**Key Features**:
- Multi-year analysis (2018-2024)
- Automatic constant/all-zero column removal
- K-fold cross-validation with hyperparameter tuning
- Model evaluation with accuracy, precision, recall, F1, confusion matrices, and one-vs-rest AUROC
- Random Forest feature importance and Logistic Regression coefficient importance analysis

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
   - Synonym mapping (100+ synonym mappings defined)
   - Fit on training data only to prevent leakage

2. **Donor-grouped split**: 80/20 train/test split by donor ID (same donor never appears in both sets)

3. **Log transformation**: `log1p(amount)` to handle right-skewed distribution

4. **Numeric bucketing for Naive Bayes**: Contribution amount bucketed into discrete rungs [0, 50, 500, 1000, 10000, 100000, 500000]

**Models**:
1. Benchmark Logistic Regression (amount only)
2. Full Multinomial Logistic Regression (all features)
3. Random Forest (all features)
4. Naive Bayes (all features)

**Key Features**:
- Donor-level train/test split to prevent data leakage
- Comprehensive occupation preprocessing pipeline
- K-fold cross-validation with hyperparameter tuning
- Model evaluation with accuracy, precision, recall, F1, confusion matrices, and one-vs-rest AUROC
- Random Forest feature importance and Logistic Regression coefficient importance analysis

## Major Functionality

### Data Processing (Final_Project.ipynb)

| Function | Purpose |
|----------|---------|
| `download_if_missing()` | Download data files from URLs if not present locally |
| `decompress_gz_if_needed()` | Decompress .gz files to .csv for Spark reading |
| `build_contrib_model_table()` | Build the main transaction-level DataFrame |
| `add_donor_split()` | Split data by donor ID to prevent leakage |

### Feature Engineering (Final_Project.ipynb)

| Function | Purpose |
|----------|---------|
| `clean_free_text()` | Clean raw occupation text (uppercase, remove punctuation, collapse spaces) |
| `apply_synonym_map()` | Map synonyms to canonical occupation labels |
| `build_fuzzy_category_map()` | Create fuzzy matching rules using Levenshtein distance |
| `build_bucket_with_fuzzy_match()` | Full occupation bucketing pipeline (fit on train, apply to all) |
| `build_preprocess_stages()` | Create Spark ML pipeline stages (StringIndexer, OneHotEncoder, VectorAssembler, Bucketizer) |

### Modeling & Evaluation (Both Notebooks)

| Function | Purpose |
|----------|---------|
| `evaluate_predictions()` | Compute accuracy, F1, precision, recall, AUROC |
| `confusion_table()` | Generate confusion matrix as a Spark DataFrame |
| `cv_results_table()` | Summarize cross-validation results |
| `roc_curve_points()` | Compute ROC curve points for one-vs-rest classification |
| `random_forest_feature_importance_df()` | Extract Random Forest feature importances |
| `logistic_coefficient_importance_df()` | Extract Logistic Regression coefficients |
| `group_ohe_importance()` | Group one-hot encoded feature levels back to original features |

## Getting Started

### Prerequisites

- Python 3.8+
- Apache Spark 3.4+ (local mode or cluster)
- Jupyter Notebook
- 16GB+ RAM recommended
- 50GB+ free disk space for data

### Installation

1. Clone the repository:
```bash
git clone https://github.com/cbauerswald/group16_project_ds5110.git
cd group16_project_ds5110
```

2. Install Python dependencies:
```bash
pip install -r requirements.txt
```

3. Start Jupyter Notebook:
```bash
jupyter notebook
```

### Usage

**For Aggregate-Level Model (EDA.ipynb):**
- Open `EDA.ipynb` in Jupyter
- Run cells sequentially to perform EDA and build aggregate-level models
- The notebook will automatically download required DIME and FEC data
- Models include Logistic Regression and Random Forest with cross-validation

**For Transaction-Level Model (Final_Project.ipynb):**
- Open `Final_Project.ipynb` in Jupyter
- Run cells sequentially to build donor-level prediction models
- The notebook will download the DIME contribDB file (2012 cycle, 10% sample)
- Models include Logistic Regression, Random Forest, and Naive Bayes
- Features advanced occupation bucketing with fuzzy matching

### Configuration

Both notebooks have configuration sections at the top where you can adjust:
- Data download settings
- Model parameters
- Cross-validation settings
- Performance optimization options
- Debugging flags

## Data Sources

### Stanford DIME (Database on Ideology, Money in Politics, and Elections)
- **Recipients/Candidates Aggregate**: Comprehensive recipient-level data (1979-2024)
- **Itemized Contributions (contribDB)**: Individual contribution records with donor information
- **Ideology Scores**: cfscore measurements for political ideology

### Federal Election Commission (FEC)
- **Candidate Summary**: Financial summaries for political candidates
- **Candidate Master**: Candidate demographic and contact information
- **Candidate-Committee Linkage**: Connections between candidates and committees
- **Committee Master/Summary**: Committee financial and organizational data

## Key Features

### Robust Data Handling
- Automatic data downloading and caching
- Parquet file optimization for faster subsequent runs
- Configurable memory and disk usage limits
- Automatic handling of compressed files (.gz, .zip)

### Advanced Feature Engineering
- Occupation text normalization and synonym mapping
- Fuzzy matching for occupation categorization
- Donor-level train/test splitting to prevent leakage
- Log transformation for skewed distributions
- Automatic removal of constant/zero columns

### Comprehensive Model Evaluation
- Multi-metric evaluation (accuracy, precision, recall, F1, AUROC)
- Confusion matrices for all models
- Cross-validation with hyperparameter tuning
- Feature importance analysis
- One-vs-rest ROC analysis for multiclass problems

## Project Outcomes

This project demonstrates that:
1. Aggregate-level fundraising data can effectively predict candidate party affiliation
2. Donor-level features (occupation, location, amount) provide predictive signal for recipient party
3. Feature engineering techniques like occupation bucketing significantly improve model performance
4. Random Forest models generally outperform logistic regression for this classification task
5. Proper train/test splitting by donor ID is crucial to prevent data leakage in contribution analysis
