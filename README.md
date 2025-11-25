# Nova Financial Analysis README

# Nova Financial Analysis

A comprehensive financial analysis project using Python, focused on stock data analysis, technical indicators, and sentiment correlation with stock movements. This project was developed as part of a hands-on data analysis and quantitative finance workflow.

---

## Project Structure

```
nova-financial-analysis-week1/
│
├─ data/
│   ├─ yfinanceData/           # Stock CSV files
│   └─ newsData/               # News and analyst reports CSV files
│
├─ notebooks/
│   ├─ 01_eda_analysis.ipynb   # Exploratory Data Analysis (EDA)
│   ├─ 02_quantitative_analysis.ipynb  # Technical indicators & financial metrics
│   └─ 03_correlation_analysis.ipynb   # Correlation between news sentiment and stock returns
│
├─ requirements.txt            # Project dependencies
└─ README.md
```

---

## Installation

1. **Clone the repository**

```bash
git clone https://github.com/MekdelawitGebre/nova-financial-analysis-week1.git
cd nova-financial-analysis-week1
```

2. **Create and activate a virtual environment**

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

3. **Install dependencies**

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Project Overview

### **Task 1 — Exploratory Data Analysis (EDA)**
- Investigated dataset structure, missing values, and basic statistics.
- Visualized distributions of stock prices and news headlines.
- Generated insights on news headline length, active publishers, and stock trends.

### **Task 2 — Quantitative Analysis**
- Loaded historical stock data from multiple CSVs automatically.
- Calculated technical indicators using `TA-Lib`:
  - Simple & Exponential Moving Averages (SMA, EMA)
  - Relative Strength Index (RSI)
  - MACD
- Computed financial metrics using `PyNance`.
- Visualized stock price trends and technical indicators.

### **Task 3 — News Sentiment & Stock Correlation**
- Performed sentiment analysis on news headlines using `TextBlob`.
- Aggregated daily sentiment per stock.
- Computed daily stock returns from closing prices.
- Calculated Pearson correlation between daily sentiment and stock returns.
- Created a single combined dashboard for all stocks, ideal for reports and presentations.

---

## Key Features

- Automated loading of all stock CSVs.
- Sentiment analysis of financial news and correlation with stock movement.
- Presentation-ready visualizations.
- Clean, modular, and reproducible Jupyter notebooks.

---

## 🛠 Tools & Libraries

- **Python 3.11**
- `pandas` — Data manipulation
- `numpy` — Numerical computations
- `matplotlib` & `seaborn` — Data visualization
- `TA-Lib` — Technical analysis indicators
- `PyNance` — Financial metrics
- `TextBlob` & `nltk` — Sentiment analysis

---

## How to Run

1. Activate the virtual environment.
2. Open Jupyter Notebook:

```bash
jupyter notebook
```

3. Navigate to `notebooks/` and run notebooks in order:
   1. `01_eda_analysis.ipynb`
   2. `02_quantitative_analysis.ipynb`
   3. `03_correlation_analysis.ipynb`

---

