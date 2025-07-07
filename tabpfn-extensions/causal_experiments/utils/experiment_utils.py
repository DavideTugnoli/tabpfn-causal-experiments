import matplotlib.pyplot as plt
import sys
from io import StringIO

def generate_synthetic_data_quiet(model, n_samples, dag=None, cpdag=None, n_permutations=3):
    """Generate synthetic data with TabPFN, suppressing output."""
    plt.ioff()
    plt.close('all')
    
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    sys.stdout = StringIO()
    sys.stderr = StringIO()
    
    try:
        # Build kwargs dict with only the parameters that are not None
        kwargs = {
            'n_samples': n_samples,
            't': 1.0,
            'n_permutations': n_permutations,
        }
        if dag is not None:
            kwargs['dag'] = dag
        if cpdag is not None:
            kwargs['cpdag'] = cpdag
            
        X_synthetic = model.generate_synthetic_data(**kwargs).cpu().numpy()
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        plt.close('all')
    
    return X_synthetic

def reorder_data_and_columns(X, col_names, categorical_cols, column_order):
    """
    Reorder data and column metadata according to column_order.
    Does NOT reorder the DAG since TabPFN doesn't care about column order when using a DAG.
    
    Args:
        X: Data array to reorder
        col_names: List of column names
        categorical_cols: List of categorical column indices
        column_order: New order of columns (list of indices)
    
    Returns:
        X_reordered: Reordered data array
        col_names_reordered: Reordered column names
        categorical_cols_reordered: Reordered categorical column indices
    """
    # Reorder data
    X_reordered = X[:, column_order]
    
    # Reorder column names
    col_names_reordered = [col_names[i] for i in column_order]
    
    # Reorder categorical column indices
    categorical_cols_reordered = None
    if categorical_cols:
        old_to_new = {old_idx: new_idx for new_idx, old_idx in enumerate(column_order)}
        categorical_cols_reordered = [old_to_new[col] for col in categorical_cols if col in old_to_new]
    
    return X_reordered, col_names_reordered, categorical_cols_reordered 