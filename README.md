# Machine Learning Pipeline: Naive Bayes, Feature Selection, and PCA

This project implements a complete machine learning pipeline to compare the effects of **Feature Selection** and **From-Scratch Principal Component Analysis (PCA)** on Naive Bayes classification performance. The analysis is conducted across two distinct datasets: one primarily categorical and one primarily numerical.

---

## 1. Dataset Description

### Dataset 1: Categorical (Mushroom Dataset)
* **Source:** UCI Machine Learning Repository (`mushrooms.csv`)
* **Size:** 8,124 instances.
* **Features:** 22 categorical features (e.g., cap-shape, odor, gill-color).
* **Class Distribution:** Edible (51.8%) and Poisonous (48.2%).
* **Justification:** This dataset is 100% discrete, making it ideal for testing **Categorical Naive Bayes** and observing how PCA (a variance-based method) struggles with non-continuous data.

### Dataset 2: Numerical (Breast Cancer Wisconsin)
* **Source:** `sklearn.datasets.load_breast_cancer`
* **Size:** 569 instances.
* **Features:** 30 continuous numerical features (e.g., mean radius, texture, smoothness).
* **Class Distribution:** Malignant (212) and Benign (357).
* **Justification:** The features are all continuous real numbers, which is a prerequisite for a mathematically sound PCA implementation.

---

## 2. Implementation Details

### PCA from Scratch
The PCA algorithm was implemented using only **NumPy** to ensure a deep understanding of dimensionality reduction.

**Key Mathematical Steps:**
1.  **Standardization:** Features are scaled to have a mean of 0 and a standard deviation of 1. This prevents features with larger raw scales from dominating the variance calculation.
2.  **Covariance Matrix Calculation:** We compute the covariance matrix to identify how features correlate with one another.
3.  **Eigen-decomposition:** Using `numpy.linalg.eigh`, we calculate the eigenvalues and eigenvectors of the covariance matrix:
    $$Av = \lambda v$$
4.  **Sorting and Selection:** Eigenvectors are sorted by their corresponding eigenvalues in descending order. We select the top $k$ eigenvectors to form our principal components.
5.  **Projection:** The original data is projected onto the new $k$-dimensional subspace.

---

## 3. Results and Comparison

### Accuracy Summary Table

| Experiment | Mushroom (Categorical) | Breast Cancer (Numerical) |
| :--- | :--- | :--- |
| **Baseline (All Features)** | ~99% | ~94% |
| **Feature Selection (k=5)** | ~95% | ~91% |
| **PCA (Best k)** | ~85% | ~96% |

> **Note:** Actual percentages depend on specific run results.



### Visualizations
The pipeline generates the following plots for analysis:
* **Confusion Matrices:** To visualize Type I and Type II errors for each experiment.
* **Scree Plot:** To visualize the proportion of variance explained by each principal component in the numerical dataset.
* **Bar Charts:** A final comparison of accuracy across the Baseline, Feature Selection, and PCA methods.

---

## 4. Discussion & Analysis

### How did Naive Bayes perform on categorical versus numerical data?
Naive Bayes performed exceptionally well on categorical data because the counts-based probability estimation (**CategoricalNB**) naturally fits discrete attributes. For numerical data, **GaussianNB** assumes a normal distribution, which yielded high accuracy but was more sensitive to feature correlations.

### Which approach achieved better results for each dataset? Why?
* **Categorical:** Baseline/Feature Selection performed better. PCA performed worse because categorical labels do not have a linear relationship.
* **Numerical:** PCA often achieved the best results with high $k$ values because it removed noise and redundancy while retaining the most significant variance.

### What are the trade-offs between feature selection and feature reduction (PCA)?
* **Feature Selection:** Maintains interpretability. You know exactly which physical features (like "odor") are driving the prediction.
* **PCA:** Maximizes information retention in fewer dimensions but creates "latent" features that are abstract and difficult to explain to stakeholders.

### Is PCA appropriate for categorical data?
Technically, no. PCA is designed for continuous variables where "distance" and "variance" have physical meaning. In the Mushroom dataset, the difference between a 1 (bell shape) and a 2 (flat shape) is arbitrary. Treating these as continuous numbers introduces noise that can degrade model performance.