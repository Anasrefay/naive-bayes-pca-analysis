import numpy as np
from src.preprocessing import load_and_clean_data, preprocess_catdata, split_data
from naive_bayes_scratch import NaiveBayesScratch
from src.feature_selection import categorical_features
from src.evaluation import evaluate_model, plot_scree, plot_accuracy_comparison
from src.pca_scratch import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import load_breast_cancer
from sklearn.feature_selection import SelectKBest, f_classif

def main():
    results_cat = {}
    results_num = {}

    print("\n" + "="*30 + "\nDATASET 1: CATEGORICAL\n" + "="*30)
    data = load_and_clean_data('data/mushrooms.csv')
    X_encoded, y_encoded, _, _ = preprocess_catdata(data, target_column='class')
    X_train, X_test, y_train, y_test = split_data(X_encoded, y_encoded)

    print("\nExperiment 0: Baseline...")
    nb_baseline = NaiveBayesScratch()
    nb_baseline.fit(X_train, y_train)
    y_pred_0 = nb_baseline.predict(X_test)
    acc_0 = evaluate_model(y_test, y_pred_0, title="Categorical Baseline")
    results_cat['Baseline'] = acc_0

    print("\nExperiment A: Feature Selection (k=5)...")
    X_train_sel, selected_indices = categorical_features(X_train, y_train, k=5)
    X_test_sel = X_test[:, selected_indices]
    nb_selection = NaiveBayesScratch()
    nb_selection.fit(X_train_sel, y_train)
    y_pred_A = nb_selection.predict(X_test_sel)
    acc_A = evaluate_model(y_test, y_pred_A, title="Categorical Feature Selection")
    results_cat['Feature Selection (k=5)'] = acc_A

    print("\nExperiment B: PCA (Testing multiple k)...")
    best_acc_cat = 0
    best_k_cat = 0
    
    for k in [2, 5, 10, 15]:
        pca_cat = PCA(n_components=k)
        pca_cat.fit(X_train)
        X_train_pca = pca_cat.transform(X_train)
        X_test_pca = pca_cat.transform(X_test)

        nb_pca = GaussianNB() 
        nb_pca.fit(X_train_pca, y_train)
        y_pred_k = nb_pca.predict(X_test_pca)
        
        acc = np.mean(y_pred_k == y_test)
        print(f"PCA (Categorical) with k={k}: Accuracy = {acc:.4f}")
        
        if acc > best_acc_cat:
            best_acc_cat = acc
            best_k_cat = k
            
    results_cat[f'PCA (k={best_k_cat})'] = best_acc_cat
    plot_accuracy_comparison(results_cat, title="Mushroom Dataset Accuracy Comparison")

    print("\n" + "="*30 + "\nDATASET 2: NUMERICAL\n" + "="*30)
    cancer = load_breast_cancer()
    X_num, y_num = cancer.data, cancer.target
    X_train_n, X_test_n, y_train_n, y_test_n = split_data(X_num, y_num)

    print("\nExperiment 0: Baseline (Numerical)...")
    nb_num_baseline = GaussianNB()
    nb_num_baseline.fit(X_train_n, y_train_n)
    y_pred_n_0 = nb_num_baseline.predict(X_test_n)
    acc_n_0 = evaluate_model(y_test_n, y_pred_n_0, title="Numerical Baseline")
    results_num['Baseline'] = acc_n_0

    print("\nExperiment A: Feature Selection (Numerical k=5)...")
    selector = SelectKBest(score_func=f_classif, k=5)
    X_train_n_sel = selector.fit_transform(X_train_n, y_train_n)
    X_test_n_sel = selector.transform(X_test_n)
    nb_num_sel = GaussianNB()
    nb_num_sel.fit(X_train_n_sel, y_train_n)
    y_pred_n_A = nb_num_sel.predict(X_test_n_sel)
    acc_n_A = evaluate_model(y_test_n, y_pred_n_A, title="Numerical Feature Selection")
    results_num['Feature Selection (k=5)'] = acc_n_A

    print("\nExperiment B: PCA (Testing multiple k)...")
    best_accuracy = 0
    best_k = 0
    for k in [2, 5, 10, 15]:
        pca_num = PCA(n_components=k)
        pca_num.fit(X_train_n)
        X_train_pca = pca_num.transform(X_train_n)
        X_test_pca = pca_num.transform(X_test_n)

        nb_pca = GaussianNB()
        nb_pca.fit(X_train_pca, y_train_n)
        y_pred_k = nb_pca.predict(X_test_pca)
        
        acc = np.mean(y_pred_k == y_test_n)
        print(f"PCA (Numerical) with k={k}: Accuracy = {acc:.4f}")
        
        if acc > best_accuracy:
            best_accuracy = acc
            best_k = k
            if k == 15:
                plot_scree(pca_num)

    print(f"\nBest PCA k value: {best_k} with Accuracy: {best_accuracy:.4f}")
    results_num[f'PCA (k={best_k})'] = best_accuracy
    plot_accuracy_comparison(results_num, title="Breast Cancer Dataset Accuracy Comparison")

if __name__ == "__main__":
    main()