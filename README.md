# Machine Learning Pipeline: Naive Bayes, Feature Selection, and PCA

This project implements a complete machine learning pipeline to compare the effects of **Feature Selection** and **From-Scratch Principal Component Analysis (PCA)** on Naive Bayes classification performance. The analysis is conducted across two distinct datasets: one primarily categorical and one primarily numerical.

---

## 1. Dataset Description

### Dataset 1: Categorical (Mushroom Dataset)
* **Source:** UCI Machine Learning Repository (`mushrooms.csv`).
* **Size:** 8,124 instances.
* **Features:** 22 categorical features (e.g., cap-shape, odor, gill-color).
* **Class Distribution:** Edible (51.8%) and Poisonous (48.2%).
* **Justification:** This dataset is 100% discrete, making it ideal for testing a **from-scratch Categorical Naive Bayes** and observing how PCA (a variance-based method) struggles with non-continuous data.

### Dataset 2: Numerical (Breast Cancer Wisconsin)
* **Source:** `sklearn.datasets.load_breast_cancer`.
* **Size:** 569 instances.
* **Features:** 30 continuous numerical features (e.g., mean radius, texture, smoothness).
* **Class Distribution:** Malignant (212) and Benign (357).
* **Justification:** The features are all continuous real numbers, which is a prerequisite for a mathematically sound PCA implementation.

---

## 2. Implementation Details

### PCA from Scratch
The PCA algorithm was implemented using **NumPy** to ensure a deep understanding of dimensionality reduction. 



**Key Mathematical Steps:**
1.  **Standardization:** Features are scaled to have a mean of 0 and a standard deviation of 1. A **stability fix** was implemented to handle zero-variance features by replacing a $0$ standard deviation with $1.0$ to prevent division-by-zero errors.
2.  **Covariance Matrix Calculation:** We compute the covariance matrix to identify how features correlate with one another.
3.  **Eigen-decomposition:** Using `numpy.linalg.eigh`, we calculate the eigenvalues and eigenvectors of the covariance matrix:
    $$Av = \lambda v$$
4.  **Sorting and Selection:** Eigenvectors are sorted by their corresponding eigenvalues in descending order. We select the top $k$ eigenvectors to form our principal components.
5.  **Projection:** The original data is projected onto the new $k$-dimensional subspace.

### Naive Bayes (Scratch vs. Gaussian)
* **Scratch Implementation:** For categorical data, a from-scratch Naive Bayes was built using Laplacian smoothing and frequency-based probability estimation.
* **Gaussian Implementation:** For PCA-reduced data and numerical data, `GaussianNB` is utilized because PCA outputs continuous "latent" variables that require a normal distribution assumption.



---

## 3. Results and Comparison

### Accuracy Summary Table

| Experiment | Mushroom (Categorical) | Breast Cancer (Numerical) |
| :--- | :--- | :--- |
| **Baseline (All Features)** | ~99.41% | ~94.12% |
| **Feature Selection (k=5)** | ~95.07% | ~91.23% |
| **PCA (Best k)** | ~85.20% | ~96.49% |

> **Note:** Actual percentages reflect results using `random_state=42`.

### Visualizations
The pipeline generates the following plots for analysis:
* **Confusion Matrices:** Visualizing Type I and Type II errors for each experiment.
* **Scree Plot:** Visualizing the proportion of variance explained by each principal component in the numerical dataset.

* **Bar Charts:** A final comparison of accuracy across the Baseline, Feature Selection, and PCA methods.

---

## 4. Discussion & Analysis

### How did Naive Bayes perform on categorical versus numerical data?
Naive Bayes performed exceptionally well on categorical data because the counts-based probability estimation naturally fits discrete attributes. For numerical data, **GaussianNB** assumes a normal distribution, which yielded high accuracy but was more sensitive to feature correlations.

### Which approach achieved better results for each dataset? Why?
* **Categorical:** Baseline and Feature Selection performed significantly better. PCA performed worse because categorical labels (1, 2, 3) do not have a linear relationship, meaning variance does not equal "information" in this context.
* **Numerical:** PCA often achieved the best results with high $k$ values because it removed noise and redundancy (e.g., highly correlated features like radius and perimeter) while retaining the most significant variance.

### What are the trade-offs between feature selection and feature reduction (PCA)?
* **Feature Selection:** Maintains interpretability. You know exactly which physical features (like "odor") are driving the prediction.
* **PCA:** Maximizes information retention in fewer dimensions but creates "latent" features that are abstract and difficult to explain to stakeholders.

### Is PCA appropriate for categorical data?
Technically, no. PCA is designed for continuous variables where "distance" and "variance" have physical meaning. In the Mushroom dataset, the difference between a "bell" shape and a "flat" shape is arbitrary. Treating these as continuous numbers introduces noise that degrades model performance.