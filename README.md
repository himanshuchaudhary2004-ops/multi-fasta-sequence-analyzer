# 🧬 Multi-FASTA Sequence Analyzer

A Python/Streamlit bioinformatics dashboard for analyzing multiple DNA sequences stored in FASTA format.

## Features

- Multi-FASTA upload (`.fasta`, `.fa`, `.fna`, `.fas`)
- FASTA parsing with Biopython
- Sequence ID and description extraction
- Sequence length calculation
- A/T/G/C nucleotide counts
- GC% and AT%
- GC skew and AT skew
- AT-rich / Balanced / GC-rich classification
- Invalid-base detection
- GC content distribution
- Sequence-length comparison
- GC vs sequence-length relationship
- Linear regression statistics
- GC content box plot
- Sliding-window GC analysis
- Interactive Plotly visualizations
- CSV download
- Excel download
- Normalized FASTA download

## Scientific definitions

GC content:

`GC% = (G + C) / (A + T + G + C) × 100`

AT content:

`AT% = (A + T) / (A + T + G + C) × 100`

GC skew:

`GC skew = (G - C) / (G + C)`

AT skew:

`AT skew = (A - T) / (A + T)`

Characters other than A, T, G and C are treated as invalid/ambiguous characters and are excluded from the GC/AT denominator.

## Run locally

### 1. Install Python

Install Python 3.10 or newer.

### 2. Install dependencies

Open a terminal in this project folder:

```bash
python -m pip install -r requirements.txt
```

### 3. Start the application

```bash
python -m streamlit run app.py
```

The application will normally open at:

`http://localhost:8501`

## Deploy with Streamlit Community Cloud

1. Create a public GitHub repository.
2. Upload `app.py`, `requirements.txt` and `README.md`.
3. Open Streamlit Community Cloud.
4. Connect your GitHub account.
5. Select this repository.
6. Select branch `main`.
7. Set the main file to `app.py`.
8. Deploy.

## Project structure

```text
multi-fasta-sequence-analyzer/
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Important note

GC content is a sequence-composition measurement. It should not by itself be interpreted as proof of gene expression, thermophily, evolutionary adaptation, or any other biological phenotype. Biological interpretation requires appropriate experimental or comparative evidence.

## Author

**HIMANSHU KUMAR CHAUDHARY**
