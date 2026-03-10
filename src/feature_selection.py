from sklearn.feature_selection import SelectKBest, chi2

def categorical_features(X, y, k=5):
    selector = SelectKBest(score_func=chi2, k=k)
    X_new = selector.fit_transform(X, y)
    
    selected_indices = selector.get_support(indices=True)
    return X_new, selected_indices