import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing, make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from scipy.spatial.distance import cdist
import warnings

warnings.filterwarnings('ignore')

def get_uncertainty(model, X):
    probs = model.predict_proba(X)
    # Entropy-based uncertainty
    entropy = -np.sum(probs * np.log(probs + 1e-10), axis=1)
    return entropy

def active_learning_loop(X_train, y_train, X_test, y_test, strategy, n_queries=30, batch_size=10):
    np.random.seed(42)
    # Start with a very small labeled set
    initial_idx = np.random.choice(len(X_train), size=10, replace=False)
    
    labeled_idx = list(initial_idx)
    unlabeled_idx = [i for i in range(len(X_train)) if i not in labeled_idx]
    
    accuracies = []
    
    # RandomForest gives good probability estimates for uncertainty
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    
    model.fit(X_train[labeled_idx], y_train[labeled_idx])
    acc = accuracy_score(y_test, model.predict(X_test))
    accuracies.append(acc)
    
    for step in range(n_queries):
        if len(unlabeled_idx) == 0:
            break
            
        X_unlabeled = X_train[unlabeled_idx]
        X_labeled = X_train[labeled_idx]
        
        if strategy == 'random':
            query_idx_relative = np.random.choice(len(unlabeled_idx), size=min(batch_size, len(unlabeled_idx)), replace=False)
            
        elif strategy == 'uncertainty':
            uncertainty = get_uncertainty(model, X_unlabeled)
            query_idx_relative = np.argsort(uncertainty)[-batch_size:]
            
        elif strategy == 'ipm_hybrid':
            # 1. Informativeness
            uncertainty = get_uncertainty(model, X_unlabeled)
            u_norm = (uncertainty - np.min(uncertainty)) / (np.max(uncertainty) - np.min(uncertainty) + 1e-8)
            
            # 2. Representativeness (IPM approximation via Core-set / K-Center Greedy)
            # This explicitly bounds the Kantorovic metric (1-Wasserstein) between P_Q and P_X
            dists = cdist(X_unlabeled, X_labeled)
            min_dists = np.min(dists, axis=1)
            
            alpha = 0.5 # Trade-off parameter
            
            query_idx_relative = []
            current_min_dists = min_dists.copy()
            
            for _ in range(min(batch_size, len(unlabeled_idx))):
                r_norm_iter = (current_min_dists - np.min(current_min_dists)) / (np.max(current_min_dists) - np.min(current_min_dists) + 1e-8)
                score = alpha * u_norm + (1 - alpha) * r_norm_iter
                
                score[query_idx_relative] = -np.inf
                best_idx = np.argmax(score)
                query_idx_relative.append(best_idx)
                
                # Update distances efficiently
                new_dists = cdist(X_unlabeled, X_unlabeled[best_idx].reshape(1, -1)).flatten()
                current_min_dists = np.minimum(current_min_dists, new_dists)
            
        query_idx = [unlabeled_idx[i] for i in query_idx_relative]
        
        labeled_idx.extend(query_idx)
        unlabeled_idx = [i for i in unlabeled_idx if i not in query_idx]
        
        model.fit(X_train[labeled_idx], y_train[labeled_idx])
        acc = accuracy_score(y_test, model.predict(X_test))
        accuracies.append(acc)
        
    return accuracies

def main():
    print("Loading Complex Dataset for Active Learning benchmark...")
    # Generate a complex dataset where boundary is non-linear and sampling matters
    X, y = make_classification(n_samples=3000, n_features=30, n_informative=20, 
                               n_redundant=5, n_clusters_per_class=3, 
                               flip_y=0.05, class_sep=0.7, random_state=42)
    
    from sklearn.preprocessing import StandardScaler
    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    n_queries = 30
    batch_size = 15
    
    print("Running Random Sampling...")
    acc_random = active_learning_loop(X_train, y_train, X_test, y_test, 'random', n_queries, batch_size)
    
    print("Running Pure Uncertainty Sampling...")
    acc_uncertainty = active_learning_loop(X_train, y_train, X_test, y_test, 'uncertainty', n_queries, batch_size)
    
    print("Running IPM-Based Hybrid Strategy (Informativeness + Representativeness)...")
    acc_ipm = active_learning_loop(X_train, y_train, X_test, y_test, 'ipm_hybrid', n_queries, batch_size)
    
    final_diff = acc_ipm[-1] - acc_random[-1]
    final_diff_unc = acc_ipm[-1] - acc_uncertainty[-1]
    
    print(f"\nFinal Accuracy (Random): {acc_random[-1]:.4f}")
    print(f"Final Accuracy (Uncertainty): {acc_uncertainty[-1]:.4f}")
    print(f"Final Accuracy (IPM Hybrid): {acc_ipm[-1]:.4f}")
    print(f"Improvement over Random: {final_diff * 100:.2f}%")
    print(f"Improvement over Pure Uncertainty: {final_diff_unc * 100:.2f}%")
    
    # Save the plot
    x_axis = np.arange(10, 10 + (n_queries + 1) * batch_size, batch_size)
    plt.figure(figsize=(10, 6))
    plt.plot(x_axis, acc_random, label='Random Sampling', marker='o', linestyle='--')
    plt.plot(x_axis, acc_uncertainty, label='Uncertainty (Informativeness Only)', marker='s', linestyle='-.')
    plt.plot(x_axis, acc_ipm, label='IPM Strategy (Informativeness + Representativeness)', marker='^', linewidth=2.5, color='green')
    
    plt.title("Active Learning Performance:\nIPM Generalization Bound Strategy vs Baselines")
    plt.xlabel("Number of Labeled Samples")
    plt.ylabel("Test Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('al_comparison.png')
    print("Plot saved to al_comparison.png")

if __name__ == "__main__":
    main()
