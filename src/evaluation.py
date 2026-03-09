from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_model(y_true, y_pred, title="Model Evaluation"):
    print(f"\n--- {title} ---")
    accuracy = accuracy_score(y_true, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))
    
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix: {title}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()
    return accuracy

def plot_scree(pca_instance):
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(pca_instance.explained_variance_ratio_) + 1), 
             pca_instance.explained_variance_ratio_, 'ro-', linewidth=2)
    plt.title('Scree Plot')
    plt.xlabel('Principal Component')
    plt.ylabel('Proportion of Variance Explained')
    plt.grid(True)
    plt.show()

def plot_accuracy_comparison(results, title="Accuracy Comparison across Experiments"):
    labels = list(results.keys())
    values = list(results.values())
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(labels, values, color=sns.color_palette("viridis", len(labels)))
    
    plt.ylim(0, 1.1)
    plt.ylabel('Accuracy Score')
    plt.title(title)
    plt.xticks(rotation=45, ha='right')
    
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, f'{yval:.4f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()