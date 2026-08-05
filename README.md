# End-to-End Data Science, Machine Learning & Algorithmic Practice Portfolio

A production-grade Python repository designed to showcase core technical capabilities for **Data Science, Machine Learning, and Software Engineering** roles.

---

## 📌 Technical Highlights & Portfolio Overview

This project is structured into four core engineering modules:

### 1. 🐍 Python Core, Data Structures & Algorithms (`python_fundamentals/`)
- **Syntax, Declarations & Traversal**: Dynamic declarations, primitive types, array slicing, mutability, and traversal techniques (`for`, `range(len())`, `enumerate`, list comprehensions).
- **Array Operations & Algorithms**: Built-in array operations (search, sort, insert, delete) vs manual algorithmic implementations (Linear Search vs Binary Search, Bubble Sort vs Quick Sort vs Timsort).
- **Algorithmic Problem Solving Practice**: 10 Easy-to-Medium interview coding challenges with time/space complexity analysis and unit test assertions (Two Sum, Kadane's Algorithm, In-place Reversal, Valid Anagram, Binary Search, etc.).

### 2. 📊 Data Science Core Suite (`data_science_core/`)
- **NumPy & Pandas**: Array vectorization, matrix math, Pandas DataFrames indexing (`loc`/`iloc`), missing data imputation, `groupby` aggregations, and dataset merges.
- **Visualization Suite**: Matplotlib & Seaborn plots (Histograms/KDE distributions, Boxplots for outlier detection, Scatter regression trendlines, Annotated Seaborn correlation heatmaps).
- **Machine Learning Foundations**: Applied reference implementation of Regression (Linear, Ridge, Lasso), Classification (Logistic Regression, Decision Trees, Random Forests), and Clustering (K-Means, PCA 2D reduction) + Metric suite (Accuracy, Precision, Recall, F1, ROC-AUC, RMSE, MAE, $R^2$, Silhouette Score).

### 3. 🤖 End-to-End Machine Learning Case Studies (`projects/`)
- **Case Study 1: Real Estate Monotonic XGBoost Valuation Pipeline**
  - **Dataset**: Ames Housing Dataset (2,924 records, 80 features).
  - **Key Metrics**: **$R^2 = 86.6\%$**, **$\text{RMSE} = \$28,619$**, **$\text{MAE} = \$20,043$**.
  - **Key Innovation**: Enforced domain monotonicity constraints (`monotone_constraints`) and engineered `Quality x Area` interaction terms to eliminate pricing inversions.
- **Case Study 2: Passenger Survival Binary Classification Pipeline**
  - **Dataset**: Passenger Manifest Dataset (891 records).
  - **Key Metrics**: **Accuracy = $83.1\%$**, **Precision = $81.6\%$**, **ROC-AUC = $0.8715$**.
  - **Key Innovation**: Scikit-Learn `ColumnTransformer` preprocessing pipelines (median/mode imputation, scaling, one-hot encoding) evaluated across 5-Fold Stratified Cross-Validation.

### 4. 📄 Resume & Technical Interview Guide (`docs/`)
- **Resume-Ready Bullet Points**: Copy-and-paste metrics and technical bullet points for resumes.
- **Elevator Pitches & REACT Framework**: 60-second project summaries, REACT problem-solving communication framework, and technical Q&A cheatsheet.

---

## 📁 Repository Structure

```
real-estate-ml-api/
│
├── python_fundamentals/                  # Module 1: Python Core & Problem Solving
│   ├── 01_syntax_declarations_arrays.py  # Declarations, array structures, traversal methods
│   ├── 02_array_operations_algorithms.py # Built-in ops, Linear/Binary search, Sorting algorithms
│   └── 03_problem_solving_practice.py    # 10 Easy-to-Medium interview coding challenges
│
├── data_science_core/                    # Module 2: Data Science & ML Reference Suite
│   ├── 01_numpy_pandas_basics.py         # NumPy vectorization & Pandas DataFrames manipulation
│   ├── 02_visualization_matplotlib_seaborn.py # Publication-grade EDA charts
│   └── 03_ml_algorithms_and_metrics.py   # Regression, Classification & Clustering algorithms + metrics
│
├── projects/                             # Module 3: End-to-End ML Case Studies
│   ├── 01_ames_housing_valuation/
│   │   ├── case_study_overview.md       # Case Study architecture & metrics breakdown
│   │   ├── pipeline.ipynb               # Executable Jupyter Notebook
│   │   ├── pipeline.py                  # 5-Fold CV benchmark script
│   │   └── AmesHousing.csv              # Ames dataset
│   │
│   └── 02_titanic_survival_classification/
│       ├── case_study_overview.md       # Case Study architecture & metrics breakdown
│       ├── pipeline.ipynb               # Executable Jupyter Notebook
│       ├── pipeline.py                  # Classification benchmark script
│       └── titanic.csv                  # Dataset
│
├── docs/                                 # Module 4: Resume & Interview Prep Guide
│   └── resume_and_interview_guide.md     # Resume bullet points, pitches & technical Q&A
│
├── requirements.txt                      # Python dependencies
└── README.md
```

---

## ⚡ Execution Instructions

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Execute Python Fundamentals & Algorithmic Practice
```bash
python python_fundamentals/01_syntax_declarations_arrays.py
python python_fundamentals/02_array_operations_algorithms.py
python python_fundamentals/03_problem_solving_practice.py
```

### 3. Execute Data Science & ML Reference Suite
```bash
python data_science_core/01_numpy_pandas_basics.py
python data_science_core/02_visualization_matplotlib_seaborn.py
python data_science_core/03_ml_algorithms_and_metrics.py
```

### 4. Execute End-to-End Machine Learning Case Studies
```bash
# Case Study 1: Real Estate Valuation Monotonic XGBoost Pipeline
python projects/01_ames_housing_valuation/pipeline.py

# Case Study 2: Passenger Survival Classification Pipeline
python projects/02_titanic_survival_classification/pipeline.py
```
