import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay

def evaluate_model(y_true, y_pred, title="Model Evaluation"):
    acc = accuracy_score(y_true, y_pred)
    print(f"\n{title} Accuracy: {acc:.4f}")
    print(classification_report(y_true, y_pred))
    
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, cmap='Blues')
    plt.title(f'Confusion Matrix: {title}')
    plt.show()
    return acc

def plot_scree(pca):
    plt.plot(range(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_, 'o-')
    plt.title('Scree Plot')
    plt.xlabel('Component')
    plt.ylabel('Variance')
    plt.show()

def plot_accuracy_comparison(results, title="Accuracy Comparison"):
    plt.bar(results.keys(), results.values())
    plt.title(title)
    plt.ylabel('Accuracy')
    plt.xticks(rotation=45)
    plt.show()