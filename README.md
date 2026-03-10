# Machine Learning Pipeline: Naive Bayes, Feature Selection, and PCA

This project implements a complete, independent machine learning pipeline to compare the effects of **Feature Selection** and **From-Scratch Principal Component Analysis (PCA)** on classification performance. The system is designed to handle both categorical and numerical data types using scratch-built implementations of core algorithms.

---

## 1. Dataset Description

To ensure full independence, all datasets are loaded from local CSV files rather than library-provided helpers.

### Dataset 1: Categorical (Mushroom Dataset)
* **Source:** `data/mushrooms.csv`
* **Features:** 22 categorical attributes (e.g., cap-shape, odor, gill-color).
* **Target:** `class` (Edible vs. Poisonous).
* **Processing:** Handled by a custom **Categorical Naive Bayes** with Laplacian smoothing.

### Dataset 2: Numerical (Breast Cancer Wisconsin)
* **Source:** `data/breastcancer.csv`
* **Features:** 30 continuous numerical features (e.g., mean radius, texture, smoothness).
* **Target:** `diagnosis` (Malignant vs. Benign).
* **Processing:** Handled by a **Gaussian Naive Bayes** implementation.

---

## 2. Implementation Details

### Robust Data Cleaning & Preprocessing
The pipeline includes a "Universal Loader" in `src/preprocessing.py` that handles real-world data issues:
* **Standardization:** Replaces missing indicators like `'?'` with `NaN`.
* **Numerical Imputation:** Fills missing numerical values with the **Mean**.
* **Categorical Imputation:** Fills missing categorical gaps with the **Mode**.
* **Encoding:** Uses `OrdinalEncoder` for features and `LabelEncoder` for targets to prepare data for mathematical operations.



### PCA from Scratch
The PCA algorithm was implemented using **NumPy** to perform dimensionality reduction via eigen-decomposition.

**Mathematical Workflow:**
1.  **Standardization:** Features are centered ($mean=0$) and scaled ($std=1$).
2.  **Covariance Matrix:** Computed to identify feature correlations.
3.  **Eigen-decomposition:** Solves $Av = \lambda v$ to find principal axes.
4.  **Projection:** Projects the original $n$-dimensional data onto the top $k$ eigenvectors.



### Naive Bayes Variants
* **NaiveBayesScratch:** A frequency-based model for discrete categorical data.
* **GaussianNBScratch:** A probability density-based model used for numerical data and PCA-transformed latent variables.

---

## 3. Final Results & Performance

The following results were achieved using a 20% test split and local data loading:

### Accuracy Summary Table

| Experiment | Mushroom (Categorical) | Breast Cancer (Numerical) |
| :--- | :--- | :--- |
| **Baseline (All Features)** | 88.18% | 96.49% |
| **Feature Selection (k=5)** | 85.11% | 96.49% |
| **PCA (Best k)** | **92.74% (k=15)** | **97.37% (k=2, 5, 10)** |



---

## 4. Discussion & Analysis

### The Success of PCA on Numerical Data
On the Breast Cancer dataset, PCA (k=2) outperformed the baseline (97.37% vs 96.49%). This demonstrates that by compressing 30 features into just 2 principal components, we removed "noise" and redundant correlations, allowing the Naive Bayes model to find a cleaner decision boundary.



### Categorical Challenges
While PCA technically requires continuous data, applying it to the encoded Mushroom dataset (k=15) actually boosted accuracy to 92.74%. This suggests that even in discrete spaces, capturing the directions of maximum variance can act as a powerful form of regularization for Naive Bayes.

### Trade-offs
* **Feature Selection:** Preserves **Interpretability**. It identified that 5 specific features provide nearly the same predictive power as 30 in the numerical set.
* **PCA:** Maximizes **Accuracy**. It provides the highest performance but results in "latent features" that are mathematically optimal but physically unexplainable.

---

## 5. How to Run

1.  Ensure your data is placed in the `data/` folder as `mushrooms.csv` and `breast_cancer.csv`.
2.  Install dependencies: `pip install pandas numpy scikit-learn matplotlib`.
3.  Run the master script:
    ```bash
    python main.py
    ```