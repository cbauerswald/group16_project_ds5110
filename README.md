# Data Engineering II (DS 5110) Final Project - Group 16
# Predicting Contribution Party from Donor-Level Features

Authors:
Ismael AL-Hadjrami, Cecelia Auerswald, Tricia Hicks, Carlos Revilla

## Project Overview

This project investigates whether we can predict the political party (Democrat, Republican, or Other) of a campaign contribution recipient using only donor-level and transaction-level features. We build and compare multiple machine learning models using Stanford DIME (Database on Ideology, Money in Politics, and Elections) campaign contribution data, with additional enrichment from Federal Election Commission (FEC) data.

### Research Question

**Can we predict whether an individual campaign contribution went to a Democratic, Republican, or Other-party recipient, using only donor-level and transaction-level features?**


## Project Structure
ds5110_final_project/
├── eda.py # Aggregate-level model (candidate/recipient prediction)
├── final_proj.py # Transaction-level model (donor/contribution prediction)
├── README.md # Project documentation
├── requirements.txt # Python package dependencies
├── data/
│ └── dime_fec_final_project/ # Data directory (ignored by git)
│ ├── contrib/ # Contribution data
│ ├── fec/ # FEC data
│ └── processed/ # Processed Parquet files
└── final_project_outputs/ # Model outputs and evaluation results

## Models Implemented

### 1. Aggregate-Level Model (`eda.py`)

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

### 2. Transaction-Level Model (`final_proj.py`)

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

### Data Processing

| Function | Purpose |
|----------|---------|
| `download_if_missing()` | Download data files from URLs if not present locally |
| `decompress_gz_if_needed()` | Decompress .gz files to .csv for Spark reading |
| `read_csv_strings()` | Read CSV with all columns as strings and deduplicate names |
| `build_contrib_model_table()` | Build the main transaction-level DataFrame |
| `add_donor_split()` | Split data by donor ID to prevent leakage |

### Feature Engineering

| Function | Purpose |
|----------|---------|
| `clean_free_text()` | Clean raw occupation text (uppercase, remove punctuation, collapse spaces) |
| `apply_synonym_map()` | Map synonyms to canonical occupation labels |
| `build_fuzzy_category_map()` | Create fuzzy matching rules using Levenshtein distance |
| `build_bucket_with_fuzzy_match()` | Full occupation bucketing pipeline (fit on train, apply to all) |
| `build_preprocess_stages()` | Create Spark ML pipeline stages (StringIndexer, OneHotEncoder, VectorAssembler) |

### Modeling & Evaluation

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
- 16GB+ RAM recommended
- 50GB+ free disk space for data

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/ds5110_final_project.git
cd ds5110_final_project
