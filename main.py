from src.preprocessing import load_and_clean_data, preprocess_catdata, split_data
from src.naive_bayes_scratch import NaiveBayesScratch
from src.feature_selection import categorical_features
from src.evaluation import evaluate_model, plot_scree, plot_accuracy_comparison
from src.pca_scratch import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import load_breast_cancer
from sklearn.feature_selection import SelectKBest, f_classif

def run_exp(model, xtrain, xtest, ytrain, ytest, name):
    model.fit(xtrain, ytrain)
    preds = model.predict(xtest)
    return evaluate_model(ytest, preds, title=name)

def main():
    # --- DATASET 1: CATEGORICAL ---
    df_cat = load_and_clean_data('data/mushrooms.csv')
    X_c, y_c = preprocess_catdata(df_cat, target='class')
    xt_c, xv_c, yt_c, yv_c = split_data(X_c, y_c)
    
    res_cat = {'Baseline': run_exp(NaiveBayesScratch(), xt_c, xv_c, yt_c, yv_c, "Cat Baseline")}

    xt_sel, idx = categorical_features(xt_c, yt_c, k=5)
    res_cat['FS (k=5)'] = run_exp(NaiveBayesScratch(), xt_sel, xv_c[:, idx], yt_c, yv_c, "Cat FS")

    best_pca_c = 0
    for k in [2, 5, 10, 15]:
        pca = PCA(n_components=k)
        pca.fit(xt_c)
        acc = run_exp(GaussianNB(), pca.transform(xt_c), pca.transform(xv_c), yt_c, yv_c, f"Cat PCA k={k}")
        best_pca_c = max(best_pca_c, acc)
    res_cat['PCA Best'] = best_pca_c
    plot_accuracy_comparison(res_cat, "Mushroom Results")

    # --- DATASET 2: NUMERICAL ---
    cancer = load_breast_cancer()
    xt_n, xv_n, yt_n, yv_n = split_data(cancer.data, cancer.target)
    
    res_num = {'Baseline': run_exp(GaussianNB(), xt_n, xv_n, yt_n, yv_n, "Num Baseline")}

    sel = SelectKBest(f_classif, k=5)
    res_num['FS (k=5)'] = run_exp(GaussianNB(), sel.fit_transform(xt_n, yt_n), sel.transform(xv_n), yt_n, yv_n, "Num FS")

    best_pca_n = 0
    for k in [2, 5, 10, 15]:
        pca = PCA(n_components=k)
        pca.fit(xt_n)
        acc = run_exp(GaussianNB(), pca.transform(xt_n), pca.transform(xv_n), yt_n, yv_n, f"Num PCA k={k}")
        if acc > best_pca_n:
            best_pca_n = acc
            if k == 15: plot_scree(pca)
    res_num['PCA Best'] = best_pca_n
    plot_accuracy_comparison(res_num, "Cancer Results")

if __name__ == "__main__":
    main()