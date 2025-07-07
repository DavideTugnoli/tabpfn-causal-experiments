#  Copyright (c) Prior Labs GmbH 2025.
#  Licensed under the Apache License, Version 2.0

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from tabpfn_extensions import TabPFNClassifier, TabPFNRegressor, unsupervised

# Feature names for the toy example
attribute_names = ["X0", "X1", "X2", "X3"]

# Simple CPDAG adjacency matrix (4 variables)
cpdag_adj = np.array([
    [ 0, -1,  0,  0],  # X0 -> X1
    [ 1,  0, -1,  0],  # X1 --- X2 (non orientato)
    [ 0, -1,  0,  0],  # X2
    [ 0,  0,  0,  0]   # X3 isolata
])

print("CPDAG adjacency matrix:")
print(cpdag_adj)

print("\nCPDAG structure (parsed from adjacency matrix):")
for i, feature_name in enumerate(attribute_names):
    parents = [j for j in range(len(attribute_names)) if cpdag_adj[i, j] == 1]
    undirected = []
    for j in range(len(attribute_names)):
        if (cpdag_adj[i, j] == -1 and cpdag_adj[j, i] == -1 and i != j):
            undirected.append(j)
    parent_names = [attribute_names[p] for p in parents]
    undirected_names = [attribute_names[u] for u in undirected]
    print(f"{feature_name}: causal_parents={parent_names}, correlations={undirected_names}")

# Generate synthetic data for the example (just for demonstration)
np.random.seed(42)
N = 200
X = np.zeros((N, 4))
# X0: standard normal
X[:, 0] = np.random.normal(0, 1, N)
# X1: depends on X0
X[:, 1] = 2 * X[:, 0] + np.random.normal(0, 1, N)
# X2: correlated with X1 (undirected edge)
X[:, 2] = -1.5 * X[:, 1] + np.random.normal(0, 1, N)
# X3: independent
X[:, 3] = np.random.normal(0, 1, N)

y = np.zeros(N)  # Dummy target (not used)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.5,
    random_state=42,
)

# Initialize TabPFN models
clf = TabPFNClassifier(n_estimators=3)
reg = TabPFNRegressor(n_estimators=3)

# Initialize unsupervised model
model_unsupervised = unsupervised.TabPFNUnsupervisedModel(
    tabpfn_clf=clf,
    tabpfn_reg=reg,
)

# Create and run synthetic experiment
exp_synthetic = unsupervised.experiments.GenerateSyntheticDataExperiment(
    task_type="unsupervised",
)

# Convert data to torch tensors
X_tensor = torch.tensor(X_train, dtype=torch.float32)
y_tensor = torch.tensor(y_train, dtype=torch.float32)

# Run the experiment with CPDAG adjacency matrix
print("\nGenerating synthetic data with hybrid CPDAG approach...")
results = exp_synthetic.run(
    tabpfn=model_unsupervised,
    X=X_tensor,
    y=y_tensor,
    attribute_names=attribute_names,
    temp=1.0,
    n_samples=X_train.shape[0] * 2,  # Generate 2x original samples
    indices=list(range(X_train.shape[1])),  # Use all features
    n_permutations=3,
    cpdag=cpdag_adj,  # Use CPDAG adjacency matrix
)

print(f"Generated {results['synthetic_X'].shape[0]} synthetic samples")
print(f"Original data shape: {X_train.shape}")
print(f"Synthetic data shape: {results['synthetic_X'].shape}")

# Compare distributions
import matplotlib.pyplot as plt
import seaborn as sns

fig, axes = plt.subplots(2, 2, figsize=(10, 8))
fig.suptitle('Original vs Synthetic Data Distribution (Simple CPDAG Example)')

for i, feature_name in enumerate(attribute_names):
    row, col = i // 2, i % 2
    axes[row, col].hist(X_train[:, i], alpha=0.7, label='Original', bins=20, density=True)
    axes[row, col].hist(results['synthetic_X'][:, i], alpha=0.7, label='Synthetic', bins=20, density=True)
    axes[row, col].set_title(f'{feature_name}')
    axes[row, col].legend()

plt.tight_layout()
plt.show()

print("\nCPDAG Hybrid Approach Summary:")
print("- Causal relationships (parents) are strictly respected")
print("- Correlational relationships (undirected edges) are handled using vanilla approach")
print("- Mixed nodes use only causal parents for generation")
print("- Generation order: causal nodes -> mixed nodes -> correlational nodes")
print("- Uses adjacency matrix format compatible with causal-learn") 