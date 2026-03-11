from src.preprocessing import load_and_clean_data, preprocess_catdata, split_data
from src.naive_bayes_scratch import NaiveBayesScratch
from src.feature_selection import select_top_features
from src.evaluation import evaluate_model, plot_scree, plot_accuracy_comparison
from src.pca_scratch import PCA
from src.GaussianNBScratch import GaussianNBScratch

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

    xt_sel, idx = select_top_features(xt_c, yt_c, k=5)
    res_cat['FS (k=5)'] = run_exp(NaiveBayesScratch(), xt_sel, xv_c[:, idx], yt_c, yv_c, "Cat FS")

    pca_full_c = PCA(n_components=xt_c.shape[1])
    pca_full_c.fit(xt_c)
    plot_scree(pca_full_c)  # scree plot for categorical

    best_pca_c = 0
    best_k_c = 0
    for k in [2, 5, 10, 15]:
        pca = PCA(n_components=k)
        pca.fit(xt_c)
        acc = run_exp(GaussianNBScratch(), pca.transform(xt_c), pca.transform(xv_c), yt_c, yv_c, f"Cat PCA k={k}")
        if acc > best_pca_c:
            best_pca_c = acc
            best_k_c = k
    print(f"Best PCA k for categorical: {best_k_c} with accuracy {best_pca_c:.4f}")
    res_cat['PCA Best'] = best_pca_c
    plot_accuracy_comparison(res_cat, "Mushroom Results")

    # --- DATASET 2: NUMERICAL ---
    df_num = load_and_clean_data('data/breastcancer.csv')

    X_n, y_n = preprocess_catdata(df_num, target='diagnosis')
    xt_n, xv_n, yt_n, yv_n = split_data(X_n, y_n) 

    res_num = {'Baseline': run_exp(GaussianNBScratch(), xt_n, xv_n, yt_n, yv_n, "Num Baseline")}

    xt_n_sel, idx_n = select_top_features(xt_n, yt_n, k=5)
    res_num['FS (k=5)'] = run_exp(GaussianNBScratch(), xt_n_sel, xv_n[:, idx_n], yt_n, yv_n, "Num FS")
    
    pca_full_n = PCA(n_components=xt_n.shape[1])
    pca_full_n.fit(xt_n)
    plot_scree(pca_full_n)  # scree plot for numerical

    best_pca_n = 0
    best_k_n = 0
    for k in [2, 5, 10, 15]:
        pca = PCA(n_components=k)
        pca.fit(xt_n)
        acc = run_exp(GaussianNBScratch(), pca.transform(xt_n), pca.transform(xv_n), yt_n, yv_n, f"Num PCA k={k}")
        if acc > best_pca_n:
            best_pca_n = acc
            best_k_n = k
    print(f"Best PCA k for numerical: {best_k_n} with accuracy {best_pca_n:.4f}")
    res_num['PCA Best'] = best_pca_n
    plot_accuracy_comparison(res_num, "Cancer Results")

if __name__ == "__main__":
    main()