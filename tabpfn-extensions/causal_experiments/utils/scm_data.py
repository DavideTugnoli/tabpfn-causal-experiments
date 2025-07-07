"""
Simple SCM data generation - no classes, just functions.
"""
import numpy as np

def generate_scm_data(n_samples, random_state=42, include_categorical=False):
    """Generate data from our SCM: X4 → X3 → X2 ← X1"""
    rng = np.random.default_rng(random_state)
    
    # Independent variables
    X4 = rng.normal(0, 1, n_samples)
    X1 = rng.normal(0, 1, n_samples)
    
    # Dependent variables
    X3 = 2.0 * X4 + rng.normal(0, 0.3, n_samples)
    X2 = 1.5 * X1 + 1.5 * X3 + rng.normal(0, 0.3, n_samples)
    
    if include_categorical:
        # X5 depends on X2
        X2_norm = (X2 - X2.min()) / (X2.max() - X2.min())
        X5 = np.zeros(n_samples, dtype=int)  # Ensure integer type
        for i in range(n_samples):
            if X2_norm[i] < 0.33:
                probs = [0.7, 0.2, 0.1]
            elif X2_norm[i] < 0.67:
                probs = [0.2, 0.6, 0.2]
            else:
                probs = [0.1, 0.2, 0.7]
            X5[i] = rng.choice(3, p=probs)
        
        # Create mixed data array with proper types
        data = np.column_stack([X1, X2, X3, X4])  # Continuous variables as float32
        data = data.astype(np.float32)
        
        # Add categorical column as integer
        data_with_cat = np.column_stack([data, X5.astype(int)])
        return data_with_cat
    else:
        return np.column_stack([X1, X2, X3, X4]).astype(np.float32)

def get_dag_and_config(include_categorical=False):
    """Get DAG and column info."""
    if include_categorical:
        dag = {0: [], 1: [0, 2], 2: [3], 3: [], 4: [1]}
        col_names = ["X1", "X2", "X3", "X4", "X5_cat"]
        categorical_cols = [4]
    else:
        dag = {0: [], 1: [0, 2], 2: [3], 3: []}
        col_names = ["X1", "X2", "X3", "X4"]
        categorical_cols = []
    
    return dag, col_names, categorical_cols

def get_cpdag_and_config(include_categorical=False):
    """Get CPDAG and column info. CPDAG: X1 -> X2 <- X3 - X4"""
    if include_categorical:
        # CPDAG adjacency matrix: X1 -> X2 <- X3 - X4, X5_cat depends on X2
        # Format: cpdag_adj[i, j] == -1 and cpdag_adj[j, i] == -1 means i --- j (undirected)
        #         cpdag_adj[i, j] == -1 and cpdag_adj[j, i] == 1 means i --> j (directed)
        cpdag_adj = np.array([
            [0, -1, 0, 0, 0],    # X1 -> X2
            [1, 0, 1, 0, -1],    # X2 <- X1, X2 <- X3, X2 -> X5_cat
            [0, -1, 0, -1, 0],   # X3 -> X2, X3 - X4
            [0, 0, -1, 0, 0],    # X4 - X3
            [0, 1, 0, 0, 0]      # X5_cat <- X2
        ])
        col_names = ["X1", "X2", "X3", "X4", "X5_cat"]
        categorical_cols = [4]
    else:
        # CPDAG adjacency matrix: X1 -> X2 <- X3 - X4
        cpdag_adj = np.array([
            [0, -1, 0, 0],    # X1 -> X2
            [1, 0, 1, 0],     # X2 <- X1, X2 <- X3
            [0, -1, 0, -1],   # X3 -> X2, X3 - X4
            [0, 0, -1, 0]     # X4 - X3
        ])
        col_names = ["X1", "X2", "X3", "X4"]
        categorical_cols = []
    
    return cpdag_adj, col_names, categorical_cols