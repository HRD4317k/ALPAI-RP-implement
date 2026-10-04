import os
import io
import zipfile
import urllib.request
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    log_loss,
    roc_auc_score,
    f1_score
)

warnings.filterwarnings("ignore")


# ================================================================
# CONFIGURATION
# ================================================================

DATA_URL = (
    "https://archive.ics.uci.edu/static/public/"
    "222/bank+marketing.zip"
)

DATA_DIR = "data"
RESULTS_DIR = "results"

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

ZIP_PATH = os.path.join(
    DATA_DIR,
    "bank+marketing.zip"
)

CSV_PATH = os.path.join(
    DATA_DIR,
    "bank-full.csv"
)


# ------------------------------------------------
# Active Learning configuration
# ------------------------------------------------

INITIAL_LABELS = 100

BATCH_SIZE = 100

N_QUERIES = 20

# Multiple seeds are essential for a research experiment.
SEEDS = [42, 52, 62, 72, 82]

ALPHA = 0.50
CANDIDATE_MULTIPLIER = 10
MMD_SIGMA = None


# ================================================================
# 1. DOWNLOAD DATASET
# ================================================================

def download_dataset():

    if os.path.exists(CSV_PATH):
        print(f"Dataset already exists: {CSV_PATH}")
        return

    if not os.path.exists(ZIP_PATH):
        print("\nDownloading UCI Bank Marketing dataset...")
        urllib.request.urlretrieve(DATA_URL, ZIP_PATH)
        print("Download complete.")

    print("Extracting nested bank.zip...")

    with zipfile.ZipFile(ZIP_PATH, "r") as z:
        # Extract the nested bank.zip in-memory
        with z.open("bank.zip") as nested_zip_file:
            nested_zip_data = io.BytesIO(nested_zip_file.read())

            with zipfile.ZipFile(nested_zip_data) as nz:
                members = nz.namelist()
                target = None

                for member in members:
                    if member.endswith("bank-full.csv"):
                        target = member
                        break

                if target is None:
                    raise FileNotFoundError(
                        "bank-full.csv not found inside nested bank.zip archive."
                    )

                with nz.open(target) as source:
                    with open(CSV_PATH, "wb") as destination:
                        destination.write(source.read())

    print(f"Dataset extracted to: {CSV_PATH}")


# ================================================================
# 2. LOAD DATASET
# ================================================================

def load_raw_data():
    download_dataset()
    print("\nLoading Bank Marketing dataset...")
    df = pd.read_csv(CSV_PATH, sep=";")
    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")
    return df


# ================================================================
# 3. PREPROCESS DATA
# ================================================================

def preprocess_data(df):
    df = df.copy()

    if "duration" in df.columns:
        df = df.drop(columns=["duration"])
        print("\nRemoved 'duration' to avoid post-call information leakage.")

    df["y"] = df["y"].map({"no": 0, "yes": 1}).astype(int)

    X = df.drop(columns=["y"])
    y = df["y"].values

    categorical_columns = X.select_dtypes(include=["object"]).columns.tolist()
    numerical_columns = X.select_dtypes(exclude=["object"]).columns.tolist()

    print(f"\nNumerical features: {len(numerical_columns)}")
    print(f"Categorical features: {len(categorical_columns)}")
    print(f"Positive class rate: {np.mean(y) * 100:.2f}%")

    return X, y, numerical_columns, categorical_columns


# ================================================================
# 4. CREATE PREPROCESSOR
# ================================================================

def create_preprocessor(numerical_columns, categorical_columns):
    numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])
    categorical_transformer = Pipeline(steps=[("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numerical_columns),
            ("cat", categorical_transformer, categorical_columns)
        ]
    )
    return preprocessor


# ================================================================
# 5. TRANSFORM ENTIRE DATASET
# ================================================================

def transform_dataset(X, y, numerical_columns, categorical_columns):
    preprocessor = create_preprocessor(numerical_columns, categorical_columns)
    X_transformed = preprocessor.fit_transform(X)
    X_transformed = np.asarray(X_transformed, dtype=np.float64)
    return X_transformed, y


# ================================================================
# 6. MODEL
# ================================================================

def create_model():
    return LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs", class_weight=None)


# ================================================================
# 7. EMPIRICAL RISK
# ================================================================

def empirical_risk(model, X, y):
    probabilities = model.predict_proba(X)
    return log_loss(y, probabilities, labels=[0, 1])


# ================================================================
# 8. TEST RISK
# ================================================================

def calculate_test_risk(model, X_test, y_test):
    probabilities = model.predict_proba(X_test)
    return log_loss(y_test, probabilities, labels=[0, 1])


# ================================================================
# 9. TEST METRICS
# ================================================================

def calculate_metrics(model, X_test, y_test):
    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    return {
        "accuracy": accuracy_score(y_test, predictions),
        "balanced_accuracy": balanced_accuracy_score(y_test, predictions),
        "f1": f1_score(y_test, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y_test, probabilities),
        "test_risk": log_loss(y_test, np.column_stack([1 - probabilities, probabilities]), labels=[0, 1])
    }


# ================================================================
# 10. ENTROPY UNCERTAINTY
# ================================================================

def entropy_uncertainty(model, X):
    probabilities = model.predict_proba(X)
    entropy = -np.sum(probabilities * np.log(probabilities + 1e-12), axis=1)
    return entropy


# ================================================================
# 11. NORMALIZE
# ================================================================

def normalize(values):
    values = np.asarray(values, dtype=float)
    minimum = np.min(values)
    maximum = np.max(values)
    denominator = (maximum - minimum)
    if denominator < 1e-12:
        return np.zeros_like(values)
    return (values - minimum) / denominator


# ================================================================
# 12. RBF KERNEL
# ================================================================

def rbf_kernel(X, Y, sigma):
    X_squared = np.sum(X * X, axis=1).reshape(-1, 1)
    Y_squared = np.sum(Y * Y, axis=1).reshape(1, -1)
    distances_squared = (X_squared + Y_squared - 2.0 * X @ Y.T)
    distances_squared = np.maximum(distances_squared, 0.0)
    return np.exp(-distances_squared / (2.0 * sigma * sigma))


# ================================================================
# 13. ESTIMATE MMD SIGMA
# ================================================================

def estimate_sigma(X, max_samples=1000):
    rng = np.random.default_rng(12345)
    if len(X) > max_samples:
        indices = rng.choice(len(X), size=max_samples, replace=False)
        X_sample = X[indices]
    else:
        X_sample = X

    X_squared = np.sum(X_sample * X_sample, axis=1).reshape(-1, 1)
    distances_squared = (X_squared + X_squared.T - 2.0 * X_sample @ X_sample.T)
    
    upper_triangle = np.triu_indices(len(X_sample), k=1)
    distances_squared = distances_squared[upper_triangle]
    distances_squared = distances_squared[distances_squared > 1e-12]

    if len(distances_squared) == 0:
        return 1.0

    sigma = np.sqrt(np.median(distances_squared))
    return max(sigma, 1e-6)


# ================================================================
# 14. MMD / IPM
# ================================================================

def compute_mmd(X, Y, sigma=None):
    if len(X) == 0 or len(Y) == 0:
        return 0.0

    if sigma is None:
        # Avoid huge matrix ops by sampling
        sample_size = min(len(X), 500)
        rng = np.random.default_rng(42)
        X_sample = X[rng.choice(len(X), size=sample_size, replace=False)] if len(X) > sample_size else X
        Y_sample = Y[rng.choice(len(Y), size=sample_size, replace=False)] if len(Y) > sample_size else Y
        
        combined = np.vstack([X_sample, Y_sample])
        sigma = estimate_sigma(combined)

    K_xx = rbf_kernel(X, X, sigma)
    K_yy = rbf_kernel(Y, Y, sigma)
    K_xy = rbf_kernel(X, Y, sigma)

    mmd_squared = np.mean(K_xx) + np.mean(K_yy) - 2.0 * np.mean(K_xy)
    mmd_squared = max(mmd_squared, 0.0)
    return np.sqrt(mmd_squared)


# ================================================================
# 15. RANDOM SAMPLING
# ================================================================

def query_random(X_pool, batch_size, rng):
    batch_size = min(batch_size, len(X_pool))
    return rng.choice(len(X_pool), size=batch_size, replace=False)


# ================================================================
# 16. UNCERTAINTY SAMPLING
# ================================================================

def query_uncertainty(model, X_pool, batch_size):
    uncertainty = entropy_uncertainty(model, X_pool)
    batch_size = min(batch_size, len(X_pool))
    return np.argsort(uncertainty)[-batch_size:]


# ================================================================
# 17. REPRESENTATIVENESS SAMPLING
# ================================================================

def query_representative(X_reference, X_pool, X_current_labeled, batch_size):
    batch_size = min(batch_size, len(X_pool))
    remaining = list(range(len(X_pool)))
    selected = []
    current_distribution = X_current_labeled.copy()

    for _ in range(batch_size):
        best_candidate = None
        best_mmd = np.inf

        # Subsample for speed if pool is huge
        if len(remaining) > 500:
            sample_candidates = np.random.choice(remaining, 500, replace=False)
        else:
            sample_candidates = remaining

        for idx in sample_candidates:
            candidate = X_pool[idx]
            candidate_distribution = np.vstack([current_distribution, candidate])
            mmd = compute_mmd(X_reference, candidate_distribution, sigma=MMD_SIGMA)
            
            if mmd < best_mmd:
                best_mmd = mmd
                best_candidate = idx

        if best_candidate is not None:
            selected.append(best_candidate)
            current_distribution = np.vstack([current_distribution, X_pool[best_candidate]])
            remaining.remove(best_candidate)
        else:
            break

    return np.asarray(selected, dtype=int)


# ================================================================
# 18. HYBRID IPM + INFORMATIVENESS
# ================================================================

def query_ipm_hybrid(model, X_reference, X_pool, X_current_labeled, batch_size, alpha=0.5):
    batch_size = min(batch_size, len(X_pool))
    uncertainty = entropy_uncertainty(model, X_pool)
    uncertainty_normalized = normalize(uncertainty)

    candidate_count = min(len(X_pool), batch_size * CANDIDATE_MULTIPLIER)
    candidate_indices = np.argsort(uncertainty_normalized)[-candidate_count:]

    current_distribution = X_current_labeled.copy()
    current_mmd = compute_mmd(X_reference, current_distribution, sigma=MMD_SIGMA)

    available_positions = list(range(len(candidate_indices)))
    selected_positions = []

    for _ in range(batch_size):
        best_position = None
        best_score = -np.inf
        best_new_mmd = None

        for position in available_positions:
            actual_index = candidate_indices[position]
            candidate = X_pool[actual_index]
            new_distribution = np.vstack([current_distribution, candidate])
            new_mmd = compute_mmd(X_reference, new_distribution, sigma=MMD_SIGMA)

            mmd_gain = current_mmd - new_mmd
            information_score = uncertainty_normalized[actual_index]

            score = alpha * information_score + (1.0 - alpha) * mmd_gain

            if score > best_score:
                best_score = score
                best_position = position
                best_new_mmd = new_mmd

        if best_position is not None:
            selected_positions.append(best_position)
            actual_index = candidate_indices[best_position]
            current_distribution = np.vstack([current_distribution, X_pool[actual_index]])
            current_mmd = best_new_mmd
            available_positions.remove(best_position)
        else:
            break

    return candidate_indices[selected_positions]


# ================================================================
# 19. ACTIVE LEARNING EXPERIMENT
# ================================================================

def run_active_learning(X_train, y_train, X_test, y_test, strategy, seed):
    rng = np.random.default_rng(seed)
    initial_indices = rng.choice(len(X_train), size=INITIAL_LABELS, replace=False)
    labeled_indices = initial_indices.tolist()

    all_indices = np.arange(len(X_train))
    mask = np.ones(len(X_train), dtype=bool)
    mask[initial_indices] = False
    unlabeled_indices = all_indices[mask].tolist()

    # Subsample reference distribution for speed
    if len(X_train) > 2000:
        X_reference = X_train[rng.choice(len(X_train), 2000, replace=False)]
    else:
        X_reference = X_train

    results = []

    for iteration in range(N_QUERIES + 1):
        X_labeled = X_train[labeled_indices]
        y_labeled = y_train[labeled_indices]

        model = create_model()
        model.fit(X_labeled, y_labeled)

        train_risk = empirical_risk(model, X_labeled, y_labeled)
        metrics = calculate_metrics(model, X_test, y_test)
        mmd = compute_mmd(X_reference, X_labeled, sigma=MMD_SIGMA)

        erm_ipm_objective = train_risk + mmd

        if len(unlabeled_indices) > 0:
            X_pool = X_train[unlabeled_indices]
            pool_uncertainty = entropy_uncertainty(model, X_pool)
            mean_pool_uncertainty = np.mean(pool_uncertainty)
        else:
            mean_pool_uncertainty = 0.0

        results.append({
            "seed": seed,
            "strategy": strategy,
            "iteration": iteration,
            "n_labeled": len(labeled_indices),
            "empirical_risk": train_risk,
            "test_risk": metrics["test_risk"],
            "accuracy": metrics["accuracy"],
            "balanced_accuracy": metrics["balanced_accuracy"],
            "f1": metrics["f1"],
            "roc_auc": metrics["roc_auc"],
            "mmd_ipm": mmd,
            "erm_plus_ipm": erm_ipm_objective,
            "mean_pool_uncertainty": mean_pool_uncertainty
        })

        if iteration == N_QUERIES or len(unlabeled_indices) == 0:
            break

        X_pool = X_train[unlabeled_indices]
        X_current_labeled = X_train[labeled_indices]

        print(f"  Iteration {iteration+1}/{N_QUERIES}: Labelling {BATCH_SIZE} samples using {strategy}...")

        if strategy == "random":
            relative_indices = query_random(X_pool, BATCH_SIZE, rng)
        elif strategy == "uncertainty":
            relative_indices = query_uncertainty(model, X_pool, BATCH_SIZE)
        elif strategy == "representative":
            relative_indices = query_representative(X_reference, X_pool, X_current_labeled, BATCH_SIZE)
        elif strategy == "ipm_hybrid":
            relative_indices = query_ipm_hybrid(model, X_reference, X_pool, X_current_labeled, BATCH_SIZE, alpha=ALPHA)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        selected_global = [unlabeled_indices[i] for i in relative_indices]
        labeled_indices.extend(selected_global)

        selected_set = set(selected_global)
        unlabeled_indices = [idx for idx in unlabeled_indices if idx not in selected_set]

    return pd.DataFrame(results)


# ================================================================
# 20. RUN COMPLETE EXPERIMENT
# ================================================================

def run_experiment(X, y):
    strategies = ["random", "uncertainty", "representative", "ipm_hybrid"]
    all_results = []

    for seed in SEEDS:
        print("\n" + "=" * 70)
        print(f"SEED {seed}")
        print("=" * 70)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, stratify=y, random_state=seed)

        for strategy in strategies:
            print(f"\nRunning: {strategy}")
            result = run_active_learning(X_train, y_train, X_test, y_test, strategy, seed)
            all_results.append(result)

    results = pd.concat(all_results, ignore_index=True)
    raw_path = os.path.join(RESULTS_DIR, "raw_results.csv")
    results.to_csv(raw_path, index=False)
    print(f"\nRaw results saved to:\n{raw_path}")
    return results


# ================================================================
# 21. AGGREGATE RESULTS
# ================================================================

def aggregate_results(results):
    summary = (
        results.groupby(["strategy", "n_labeled"])
        .agg(
            accuracy_mean=("accuracy", "mean"),
            accuracy_std=("accuracy", "std"),
            test_risk_mean=("test_risk", "mean"),
            test_risk_std=("test_risk", "std"),
            empirical_risk_mean=("empirical_risk", "mean"),
            empirical_risk_std=("empirical_risk", "std"),
            mmd_mean=("mmd_ipm", "mean"),
            mmd_std=("mmd_ipm", "std"),
            erm_ipm_mean=("erm_plus_ipm", "mean"),
            erm_ipm_std=("erm_plus_ipm", "std"),
            balanced_accuracy_mean=("balanced_accuracy", "mean"),
            f1_mean=("f1", "mean"),
            roc_auc_mean=("roc_auc", "mean")
        )
        .reset_index()
    )
    path = os.path.join(RESULTS_DIR, "summary_results.csv")
    summary.to_csv(path, index=False)
    print(f"\nSummary saved to:\n{path}")
    return summary


# ================================================================
# 22. PLOT FUNCTION
# ================================================================

def plot_metric(summary, metric_mean, metric_std, ylabel, title, filename):
    plt.figure(figsize=(11, 7))

    strategy_names = {
        "random": "Random",
        "uncertainty": "Uncertainty",
        "representative": "MMD Representativeness",
        "ipm_hybrid": "IPM + Informativeness"
    }
    markers = {"random": "o", "uncertainty": "s", "representative": "D", "ipm_hybrid": "^"}

    for strategy in strategy_names:
        data = summary[summary["strategy"] == strategy]
        plt.errorbar(
            data["n_labeled"], data[metric_mean], yerr=data[metric_std],
            marker=markers[strategy], markersize=6, linewidth=2, capsize=4,
            label=strategy_names[strategy]
        )

    plt.xlabel("Number of Labeled Samples", fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(title, fontsize=14)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()

    path = os.path.join(RESULTS_DIR, filename)
    plt.savefig(path, dpi=300, bbox_inches="tight")
    # plt.show() # Disabled to prevent blocking


# ================================================================
# 23. GENERATE ALL PLOTS
# ================================================================

def generate_plots(summary):
    plot_metric(summary, "accuracy_mean", "accuracy_std", "Test Accuracy", "Active Learning: Test Accuracy vs Labeling Budget", "accuracy_vs_labels.png")
    plot_metric(summary, "test_risk_mean", "test_risk_std", "Test Log-Loss", "Generalization Risk vs Labeling Budget", "test_risk_vs_labels.png")
    plot_metric(summary, "empirical_risk_mean", "empirical_risk_std", "Empirical Risk / Log-Loss", "ERM Empirical Risk vs Labeling Budget", "empirical_risk_vs_labels.png")
    plot_metric(summary, "mmd_mean", "mmd_std", "MMD(P_X, P_Q)", "Distributional Representativeness", "mmd_vs_labels.png")
    plot_metric(summary, "erm_ipm_mean", "erm_ipm_std", "Empirical Risk + MMD", "ERM + IPM Bound-Inspired Objective", "erm_plus_ipm.png")


# ================================================================
# 24. FINAL COMPARISON TABLE
# ================================================================

def print_final_comparison(summary):
    final_budget = INITIAL_LABELS + N_QUERIES * BATCH_SIZE
    final = summary[summary["n_labeled"] == final_budget].copy()
    final = final.sort_values("accuracy_mean", ascending=False)

    print("\n" + "=" * 90)
    print(f"FINAL COMPARISON AT {final_budget} LABELS")
    print("=" * 90)

    for _, row in final.iterrows():
        print(f"\nStrategy: {row['strategy']}")
        print(f"  Accuracy      : {row['accuracy_mean']:.4f} +/-{row['accuracy_std']:.4f}")
        print(f"  Test Risk     : {row['test_risk_mean']:.4f} +/-{row['test_risk_std']:.4f}")
        print(f"  Empirical Risk: {row['empirical_risk_mean']:.4f} +/-{row['empirical_risk_std']:.4f}")
        print(f"  MMD / IPM     : {row['mmd_mean']:.4f} +/-{row['mmd_std']:.4f}")
        print(f"  ERM + IPM     : {row['erm_ipm_mean']:.4f} +/-{row['erm_ipm_std']:.4f}")
        print(f"  F1            : {row['f1_mean']:.4f}")
        print(f"  ROC-AUC       : {row['roc_auc_mean']:.4f}")


# ================================================================
# 25. LABEL-EFFICIENCY ANALYSIS
# ================================================================

def calculate_label_efficiency(summary, target_accuracy=0.80):
    print("\n" + "=" * 90)
    print(f"LABEL EFFICIENCY\nTarget accuracy: {target_accuracy:.2f}")
    print("=" * 90)

    strategies = summary["strategy"].unique()
    for strategy in strategies:
        data = summary[summary["strategy"] == strategy].sort_values("n_labeled")
        successful = data[data["accuracy_mean"] >= target_accuracy]
        
        if len(successful) == 0:
            print(f"{strategy:20s}: Target not reached")
        else:
            first = successful.iloc[0]
            print(f"{strategy:20s}: {int(first['n_labeled'])} labels")


# ================================================================
# 26. MAIN
# ================================================================

def main():
    print("\n" + "=" * 90)
    print("REAL-DATA ACTIVE LEARNING EXPERIMENT")
    print("ERM + INFORMATIVENESS + IPM REPRESENTATIVENESS")
    print("=" * 90)

    df = load_raw_data()
    X_raw, y, numerical_columns, categorical_columns = preprocess_data(df)

    print("\nTransforming features...")
    X, y = transform_dataset(X_raw, y, numerical_columns, categorical_columns)
    print(f"Final feature dimension: {X.shape[1]}")

    results = run_experiment(X, y)
    summary = aggregate_results(results)
    print_final_comparison(summary)
    calculate_label_efficiency(summary, target_accuracy=0.80)

    print("\nGenerating plots...")
    generate_plots(summary)

    print("\n" + "=" * 90)
    print("EXPERIMENT COMPLETE")
    print("=" * 90)
    print(f"\nResults directory:\n{RESULTS_DIR}/")
    print("\nFiles generated:")
    print("  raw_results.csv\n  summary_results.csv\n  accuracy_vs_labels.png\n  test_risk_vs_labels.png")
    print("  empirical_risk_vs_labels.png\n  mmd_vs_labels.png\n  erm_plus_ipm.png")


if __name__ == "__main__":
    main()
