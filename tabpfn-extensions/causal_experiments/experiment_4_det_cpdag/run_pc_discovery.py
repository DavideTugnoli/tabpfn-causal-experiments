"""
CAUSAL DISCOVERY EXPERIMENT WITH P-VALUE ANALYSIS

Runs causal discovery on continuous and mixed data with detailed p-value reporting
"""
import sys
import os
import numpy as np
import pydot
import io
import matplotlib.pyplot as plt
import warnings
import pandas as pd
from scipy.stats import chi2_contingency, pearsonr
import random

# Add the causal_experiments directory to the path for absolute imports
current_dir = os.path.dirname(os.path.abspath(__file__))
causal_experiments_dir = os.path.dirname(current_dir)
sys.path.insert(0, causal_experiments_dir)

from utils.dag_utils import cpdag_to_dags, convert_cpdag_to_named_dags, convert_named_dag_to_indices
from utils.scm_data import get_dag_and_config, generate_scm_data

# For causal discovery
from causallearn.search.ConstraintBased.PC import pc
from causallearn.utils.GraphUtils import GraphUtils
from causallearn.utils.cit import CIT

# For data preparation
from sklearn.preprocessing import KBinsDiscretizer

np.random.seed(42)
random.seed(42)

def get_pairwise_pvalues(data, col_names, categorical_cols, test_type='fisherz', alpha=0.05):
    """
    Compute pairwise p-values between all variables.
    
    Returns:
        DataFrame with p-values for each variable pair
    """
    n_vars = data.shape[1]
    p_values = np.ones((n_vars, n_vars))
    
    # Initialize CIT object
    cit = CIT(data, test_type)
    
    # Compute pairwise tests
    for i in range(n_vars):
        for j in range(i+1, n_vars):
            # Test independence between i and j
            p_value = cit(i, j, [])
            p_values[i, j] = p_value
            p_values[j, i] = p_value
    
    # Create DataFrame for better display
    df = pd.DataFrame(p_values, index=col_names, columns=col_names)
    return df


def analyze_edge_discovery(data, true_dag, col_names, categorical_cols, cg, alpha):
    """
    Analyze why certain edges were or weren't discovered.
    """
    print("\n=== EDGE DISCOVERY ANALYSIS ===")
    
    # Get true edges
    true_edges = []
    for child_idx, parent_indices in true_dag.items():
        for parent_idx in parent_indices:
            true_edges.append((parent_idx, child_idx))
    
    # Analyze each true edge
    print(f"\nTrue edges analysis (alpha={alpha}):")
    for parent_idx, child_idx in true_edges:
        parent_name = col_names[parent_idx]
        child_name = col_names[child_idx]
        
        # Check if edge was discovered
        discovered = False
        if cg.G.graph[child_idx, parent_idx] == -1 or cg.G.graph[parent_idx, child_idx] == -1:
            discovered = True
        
        # Get p-value for this edge
        if parent_idx in categorical_cols or child_idx in categorical_cols:
            # For mixed pairs, use chi-square test
            if parent_idx in categorical_cols:
                # Discretize continuous variable
                continuous_data = data[:, child_idx]
                bins = pd.qcut(continuous_data, q=5, duplicates='drop')
                contingency = pd.crosstab(data[:, parent_idx], bins)
            else:
                continuous_data = data[:, parent_idx]
                bins = pd.qcut(continuous_data, q=5, duplicates='drop')
                contingency = pd.crosstab(bins, data[:, child_idx])
            
            chi2, p_value, _, _ = chi2_contingency(contingency)
        else:
            # For continuous pairs, use correlation test
            r, p_value = pearsonr(data[:, parent_idx], data[:, child_idx])
        
        status = "FOUND" if discovered else "MISSED"
        print(f"  {parent_name} -> {child_name}: {status} (p-value: {p_value:.6f})")


def discover_cpdag(data, alpha, indep_test, return_pvalues=False, **kwargs):
    """
    Runs PC algorithm with optional p-value collection.
    """
    try:
        if return_pvalues:
            # Run with verbose mode to capture p-values
            cg = pc(data, alpha=alpha, indep_test=indep_test, 
                   show_progress=False, verbose=True, **kwargs)
        else:
            cg = pc(data, alpha=alpha, indep_test=indep_test, 
                   show_progress=False, **kwargs)
        return cg
    except Exception as e:
        print(f"Failed with {indep_test}: {str(e)}")
        return None


def plot_graphs(true_dag_def, discovered_cg, col_names, filename):
    """Creates comparison plot between true and discovered graphs."""
    # True Graph
    dot_true = pydot.Dot(graph_type='digraph', label="True Causal Graph", fontsize=20, labeljust="t")
    for name in col_names:
        dot_true.add_node(pydot.Node(name))
    for child_idx, parent_indices in true_dag_def.items():
        for parent_idx in parent_indices:
            dot_true.add_edge(pydot.Edge(col_names[parent_idx], col_names[child_idx]))

    # Discovered Graph
    dot_discovered = GraphUtils.to_pydot(discovered_cg.G, labels=col_names)
    dot_discovered.set_label("Discovered Graph (CPDAG)")
    dot_discovered.set_fontsize(20)
    dot_discovered.set_labeljust("t")

    # Create image
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    axes[0].imshow(plt.imread(io.BytesIO(dot_true.create_png(prog='dot'))))
    axes[0].axis('off')
    axes[1].imshow(plt.imread(io.BytesIO(dot_discovered.create_png(prog='dot'))))
    axes[1].axis('off')
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close(fig)


def run_mixed_data_discovery(data, col_names, categorical_cols, alpha):
    """
    Attempts multiple approaches for mixed data causal discovery.
    """
    n_vars = data.shape[1]
    methods_tried = []
    
    # Method 1: Mixed CI test
    try:
        node_types = np.zeros(n_vars, dtype=int)
        node_types[list(categorical_cols)] = 1
        cg = discover_cpdag(data, alpha=alpha, indep_test='mv_fisherz', node_types=node_types)
        if cg is not None:
            return cg, "Mixed Variables Fisher-Z"
    except:
        methods_tried.append("mv_fisherz failed")
    
    # Method 2: G-square test
    try:
        cg = discover_cpdag(data, alpha=alpha, indep_test='gsq')
        if cg is not None:
            return cg, "G-square test"
    except:
        methods_tried.append("gsq failed")
    
    # Method 3: Discretization with chi-square
    continuous_indices = [i for i in range(n_vars) if i not in categorical_cols]
    
    if continuous_indices:
        discretized_data = data.copy()
        discretizer = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile')
        discretized_data[:, continuous_indices] = discretizer.fit_transform(data[:, continuous_indices])
        discretized_data = discretized_data.astype(int)
    else:
        discretized_data = data.astype(int)
    
    cg = discover_cpdag(discretized_data, alpha=alpha, indep_test='chisq')
    if cg is not None:
        return cg, "Chi-square (discretized, 5 bins)"
    
    # Method 4: Treat all as continuous
    cg = discover_cpdag(data, alpha=alpha, indep_test='fisherz')
    return cg, "Fisher-Z (all continuous)"


def run_causal_discovery_experiment(include_categorical, alpha=0.05, n_samples=1000):
    """
    Runs causal discovery experiment with detailed analysis.
    """
    exp_type = "MIXED" if include_categorical else "CONTINUOUS"
    output_filename = f"{'2_mixed' if include_categorical else '1_continuous'}_data_comparison.png"

    print(f"\n{'='*50}")
    print(f"EXPERIMENT: {exp_type} DATA")
    print(f"{'='*50}")

    # Generate data
    data = generate_scm_data(n_samples=n_samples, random_state=42, include_categorical=include_categorical)
    true_dag, col_names, categorical_cols = get_dag_and_config(include_categorical=include_categorical)
    print(f"Data shape: {data.shape}")
    
    # Show data structure
    print(f"\nTrue DAG structure:")
    for child_idx, parent_indices in true_dag.items():
        if parent_indices:
            parents = [col_names[p] for p in parent_indices]
            print(f"  {col_names[child_idx]} <- {', '.join(parents)}")
    
    # Run causal discovery
    if include_categorical:
        cg, method_used = run_mixed_data_discovery(data, col_names, categorical_cols, alpha)
    else:
        cg = discover_cpdag(data, alpha=alpha, indep_test='fisherz')
        method_used = "Fisher-Z"
    
    if cg is None:
        print("ERROR: All methods failed!")
        return
    
    print(f"\nMethod used: {method_used}")
    
    # Count discovered edges
    discovered_edges = 0
    for i in range(len(col_names)):
        for j in range(len(col_names)):
            if cg.G.graph[i, j] != 0:
                discovered_edges += 1
    print(f"Discovered edges: {discovered_edges//2}")  # Divide by 2 for undirected
    
    # Enumerate all possible DAGs from the CPDAG
    possible_dags = convert_cpdag_to_named_dags(cg.G.graph, col_names)
    print(f"\nFound {len(possible_dags)} possible DAGs in the equivalence class:")
    for i, dag in enumerate(possible_dags):
        print(f"  DAG {i+1}:")
        # Example of converting a named DAG back to an index-based DAG
        # index_dag = convert_named_dag_to_indices(dag, col_names)
        # Find and print edges for better readability
        edges = []
        for child, parents in dag.items():
            if parents:
                for parent in parents:
                    edges.append(f"{parent} -> {child}")
        if edges:
            print(f"    Edges: {', '.join(edges)}")
        else:
            print("    No edges")

    # Detailed edge analysis
    analyze_edge_discovery(data, true_dag, col_names, categorical_cols, cg, alpha)
    
    # Pairwise p-values analysis
    print("\n=== PAIRWISE P-VALUES ===")
    if include_categorical:
        # For mixed data, show p-values using chi-square for all pairs
        print("\nP-values matrix (chi-square test on discretized data):")
        p_matrix = np.ones((len(col_names), len(col_names)))
        
        for i in range(len(col_names)):
            for j in range(i+1, len(col_names)):
                if i in categorical_cols and j in categorical_cols:
                    # Both categorical
                    contingency = pd.crosstab(data[:, i], data[:, j])
                elif i in categorical_cols:
                    # i categorical, j continuous
                    bins = pd.qcut(data[:, j], q=5, duplicates='drop')
                    contingency = pd.crosstab(data[:, i], bins)
                elif j in categorical_cols:
                    # i continuous, j categorical
                    bins = pd.qcut(data[:, i], q=5, duplicates='drop')
                    contingency = pd.crosstab(bins, data[:, j])
                else:
                    # Both continuous
                    bins_i = pd.qcut(data[:, i], q=5, duplicates='drop')
                    bins_j = pd.qcut(data[:, j], q=5, duplicates='drop')
                    contingency = pd.crosstab(bins_i, bins_j)
                
                chi2, p_value, _, _ = chi2_contingency(contingency)
                p_matrix[i, j] = p_value
                p_matrix[j, i] = p_value
        
        p_df = pd.DataFrame(p_matrix, index=col_names, columns=col_names)
        print(p_df.round(4))
    else:
        # For continuous data, use correlation p-values
        p_values_df = get_pairwise_pvalues(data, col_names, categorical_cols, 'fisherz', alpha)
        print("\nP-values matrix (Fisher-Z test):")
        print(p_values_df.round(4))
    
    # Save visualization
    plot_graphs(true_dag, cg, col_names, output_filename)
    print(f"\nPlot saved: {output_filename}")


def run_pc_discovery_on_dataset(dataset_name, data, true_dag, task_type="unsupervised", 
                               target_column=None, verbose=False, output_dir=None):
    """
    Run PC algorithm on a dataset and return the discovered CPDAG.
    
    Args:
        dataset_name: Name of the dataset ("mixed" or "continuous")
        data: Data array for causal discovery
        true_dag: True DAG structure (for validation/comparison)
        task_type: Type of task (unused, kept for compatibility)
        target_column: Target column name (unused, kept for compatibility)
        verbose: Whether to print detailed output
        output_dir: Output directory for plots (if None, no plots saved)
        
    Returns:
        CPDAG adjacency matrix (numpy array)
    """
    include_categorical = dataset_name == "mixed"
    _, col_names, categorical_cols = get_dag_and_config(include_categorical=include_categorical)
    
    if verbose:
        print(f"Running PC discovery on {dataset_name} data...")
        print(f"Data shape: {data.shape}")
        print(f"Categorical columns: {categorical_cols}")
    
    # Run causal discovery
    alpha = 0.05
    if include_categorical and categorical_cols:
        cg, method_used = run_mixed_data_discovery(data, col_names, categorical_cols, alpha)
    else:
        cg = discover_cpdag(data, alpha=alpha, indep_test='fisherz')
        method_used = "Fisher-Z"
    
    if cg is None:
        if verbose:
            print("ERROR: All causal discovery methods failed!")
        # Return empty CPDAG
        return np.zeros((len(col_names), len(col_names)))
    
    if verbose:
        print(f"Method used: {method_used}")
        # Count discovered edges
        discovered_edges = 0
        for i in range(len(col_names)):
            for j in range(len(col_names)):
                if cg.G.graph[i, j] != 0:
                    discovered_edges += 1
        print(f"Discovered edges: {discovered_edges//2}")
    
    # Save plot if output directory is provided
    if output_dir is not None:
        plot_filename = f"{output_dir}/{dataset_name}_discovery_result.png"
        plot_graphs(true_dag, cg, col_names, plot_filename)
        if verbose:
            print(f"Plot saved: {plot_filename}")
    
    return cg.G.graph


def main():
    """Main function."""
    # Continuous data experiment
    run_causal_discovery_experiment(include_categorical=False, alpha=0.05, n_samples=100)
    
    # Mixed data experiment - try different alphas
    print("\n\nTrying different alpha values for mixed data:")
    for alpha in [0.01, 0.05, 0.10]:
        print(f"\n--- Alpha = {alpha} ---")
        run_causal_discovery_experiment(include_categorical=True, alpha=alpha, n_samples=2000)


if __name__ == "__main__":
    main()