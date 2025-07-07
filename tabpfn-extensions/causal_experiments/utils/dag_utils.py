"""
Generic DAG utilities that work with any DAG structure.

These functions are SCM-agnostic and can be reused across all experiments.
"""
import itertools
import numpy as np
from typing import Dict, List, Tuple
from collections import defaultdict, deque


def topological_sort(dag: Dict[int, List[int]]) -> List[int]:
    """
    Compute topological ordering of DAG nodes.
    
    This gives the "best" ordering where parents always come before children.
    
    Args:
        dag: Dictionary {node: [list_of_parents]}
        
    Returns:
        List of nodes in topological order
    """
    # Convert to adjacency list (node -> children)
    children = defaultdict(list)
    in_degree = defaultdict(int)
    
    # Get all nodes
    all_nodes = set(dag.keys())
    for parents in dag.values():
        all_nodes.update(parents)
    
    # Initialize in_degree
    for node in all_nodes:
        in_degree[node] = 0
    
    # Build children and in_degree
    for child, parents in dag.items():
        for parent in parents:
            children[parent].append(child)
            in_degree[child] += 1
    
    # Kahn's algorithm
    queue = deque([node for node in all_nodes if in_degree[node] == 0])
    result = []
    
    while queue:
        node = queue.popleft()
        result.append(node)
        
        for child in children[node]:
            in_degree[child] -= 1
            if in_degree[child] == 0:
                queue.append(child)
    
    if len(result) != len(all_nodes):
        raise ValueError("DAG contains cycles!")
    
    return result


def get_worst_ordering(dag: Dict[int, List[int]]) -> List[int]:
    """
    Get the worst possible ordering that maximizes causal violations.
    
    Uses reverse topological order, which is mathematically guaranteed to 
    produce the maximum number of causal violations possible.
    
    Mathematical guarantee: This method violates EVERY causal dependency in the DAG,
    which is the theoretical maximum number of violations achievable.
    
    Args:
        dag: Dictionary {node: [list_of_parents]}
        
    Returns:
        List of nodes in worst possible order (optimal)
    """
    # Get topological order (best ordering with 0 violations)
    topo_order = topological_sort(dag)
    
    # Reverse it to get worst ordering (maximum violations)
    worst_order = list(reversed(topo_order))
    
    return worst_order


def count_violations(dag: Dict[int, List[int]], ordering: List[int]) -> int:
    """
    Count total number of causal violations in an ordering.
    
    A violation occurs when a child node appears before its parent in the ordering.
    
    Args:
        dag: Dictionary {node: [list_of_parents]}
        ordering: List of nodes in some order
        
    Returns:
        Number of violated causal dependencies
    """
    # Create position mapping
    position = {node: i for i, node in enumerate(ordering)}
    
    violations = 0
    for child, parents in dag.items():
        child_pos = position[child]
        for parent in parents:
            parent_pos = position[parent]
            # Violation if child comes before parent
            if child_pos < parent_pos:
                violations += 1
    
    return violations


def compare_ordering_strategies(dag: Dict[int, List[int]], verbose: bool = True) -> Dict[str, int]:
    """
    Compare different ordering strategies and show violation counts.
    
    Args:
        dag: Dictionary {node: [list_of_parents]}
        verbose: If True, print comparison table
        
    Returns:
        Dictionary {strategy_name: violation_count}
    """
    strategies = {
        'topological': topological_sort(dag),
        'worst': get_worst_ordering(dag),
        'random': get_random_ordering(dag, random_state=42),
    }
    
    results = {}
    
    if verbose:
        print("Comparing ordering strategies:")
        print("-" * 50)
        
    for name, ordering in strategies.items():
        violations = count_violations(dag, ordering)
        results[name] = violations
        if verbose:
            print(f"{name:12}: {ordering} -> {violations} violations")
    
    return results


def get_random_ordering(dag: Dict[int, List[int]], random_state: int = 42) -> List[int]:
    """
    Get a random ordering of DAG nodes.
    
    Args:
        dag: Dictionary {node: [list_of_parents]}
        random_state: Random seed for reproducibility
        
    Returns:
        List of nodes in random order
    """
    all_nodes = set(dag.keys())
    for parents in dag.values():
        all_nodes.update(parents)
    
    rng = np.random.default_rng(random_state)
    nodes_list = list(all_nodes)
    rng.shuffle(nodes_list)
    
    return nodes_list


def reorder_data_and_dag(X_data: np.ndarray, original_dag: Dict[int, List[int]], 
                         new_ordering: List[int]) -> Tuple[np.ndarray, Dict[int, List[int]]]:
    """
    Reorder data columns and adjust DAG indices accordingly.
    
    This is the GENERIC function that works with any DAG and any ordering.
    
    Args:
        X_data: Original data array [n_samples, n_features]
        original_dag: DAG with original column indices
        new_ordering: New column order (e.g., from topological_sort)
    
    Returns:
        X_reordered: Data with reordered columns
        dag_reordered: DAG with updated indices to match new column positions
    """
    # Reorder data columns
    X_reordered = X_data[:, new_ordering]
    
    # Create mapping: old_index -> new_position
    old_to_new = {old_idx: new_pos for new_pos, old_idx in enumerate(new_ordering)}
    
    # Reorder DAG indices
    dag_reordered = {}
    for new_pos, old_idx in enumerate(new_ordering):
        if old_idx in original_dag:
            # Get parents in old indexing
            old_parents = original_dag[old_idx]
            # Convert to new indexing
            new_parents = []
            for parent in old_parents:
                if parent in old_to_new:  # Parent is included in reordered data
                    new_parents.append(old_to_new[parent])
            dag_reordered[new_pos] = new_parents
    
    return X_reordered, dag_reordered


def get_ordering_strategies(dag: Dict[int, List[int]]) -> Dict[str, List[int]]:
    """
    Get all available ordering strategies for a given DAG.
    
    This is the MAIN function to use - it computes orderings dynamically!
    
    Args:
        dag: DAG structure
        
    Returns:
        Dictionary of {strategy_name: ordering}
    """
    all_nodes = set(dag.keys())
    for parents in dag.values():
        all_nodes.update(parents)
    
    original_order = sorted(list(all_nodes))  # [0, 1, 2, 3, ...] - deterministic
    
    strategies = {
        'original': original_order,
        'topological': topological_sort(dag),
        'worst': get_worst_ordering(dag),
        'random': get_random_ordering(dag, random_state=42),
    }
    
    return strategies


def validate_dag(dag: Dict[int, List[int]]) -> bool:
    """
    Validate that DAG is acyclic.
    
    Args:
        dag: DAG to validate
        
    Returns:
        True if valid (acyclic), False otherwise
    """
    try:
        topological_sort(dag)
        return True
    except ValueError:
        return False


def print_dag_info(dag: Dict[int, List[int]], column_names: List[str] = None) -> None:
    """
    Print informative summary of DAG structure.
    
    Args:
        dag: DAG structure
        column_names: Optional names for columns
    """
    if column_names is None:
        column_names = [f"X{i}" for i in range(max(dag.keys()) + 1)]
    
    print("DAG Structure:")
    print("-" * 40)
    
    for child, parents in dag.items():
        child_name = column_names[child] if child < len(column_names) else f"X{child}"
        
        if not parents:
            print(f"{child_name} (independent)")
        else:
            parent_names = [column_names[p] if p < len(column_names) else f"X{p}" for p in parents]
            print(f"{child_name} ← {', '.join(parent_names)}")
    
    print("-" * 40)
    
    # Show orderings
    strategies = get_ordering_strategies(dag)
    print("Available orderings:")
    for name, ordering in strategies.items():
        ordered_names = [column_names[i] if i < len(column_names) else f"X{i}" for i in ordering]
        violations = count_violations(dag, ordering)
        print(f"  {name:12}: [{', '.join(ordered_names)}] ({violations} violations)")


def convert_named_dag_to_indices(named_dag, column_names):
    """
    Convert a DAG defined with node names to one with indices.
    
    Args:
        named_dag: Dictionary {node_name: [list_of_parent_names]}
        column_names: List of column names defining the index mapping
        
    Returns:
        Dictionary {node_index: [list_of_parent_indices]}
    """
    # Create mapping from names to indices
    name_to_idx = {name: idx for idx, name in enumerate(column_names)}
    
    # Convert DAG
    index_dag = {}
    for node_name, parent_names in named_dag.items():
        if node_name in name_to_idx:  # Skip nodes not in column_names
            node_idx = name_to_idx[node_name]
            parent_indices = [name_to_idx[p] for p in parent_names if p in name_to_idx]
            index_dag[node_idx] = parent_indices
    
    return index_dag


def convert_indices_dag_to_named(dag: Dict[int, List[int]], column_names: List[str]) -> Dict[str, List[str]]:
    """
    Convert a DAG defined with indices back to one with node names.
    
    Args:
        dag: Dictionary {node_index: [list_of_parent_indices]}
        column_names: List of column names defining the index mapping
        
    Returns:
        Dictionary {node_name: [list_of_parent_names]}
    """
    idx_to_name = {idx: name for idx, name in enumerate(column_names)}
    
    named_dag = {}
    for node_idx, parent_indices in dag.items():
        if node_idx < len(idx_to_name):  # Ensure index is valid
            node_name = idx_to_name[node_idx]
            parent_names = [idx_to_name[p] for p in parent_indices if p < len(idx_to_name)]
            named_dag[node_name] = parent_names
    
    return named_dag


def create_wrong_parents_dag(dag: Dict[int, List[int]], random_state: int = 42) -> Dict[int, List[int]]:
    """
    Create a DAG with wrong parent relationships by randomly shuffling parents.
    
    This maintains the same number of edges and ensures the result is still a valid DAG.
    
    Args:
        dag: Original DAG structure
        random_state: Random seed for reproducibility
        
    Returns:
        New DAG with shuffled parent relationships
    """
    rng = np.random.default_rng(random_state)
    
    # Get all nodes and edges
    all_nodes = set(dag.keys())
    for parents in dag.values():
        all_nodes.update(parents)
    all_nodes = sorted(list(all_nodes))
    
    # Collect all edges
    edges = []
    for child, parents in dag.items():
        for parent in parents:
            edges.append((parent, child))
    
    # Create a new random DAG with same number of edges
    new_dag = {node: [] for node in all_nodes}
    n_edges = len(edges)
    
    # Try to create a valid DAG with shuffled edges
    max_attempts = 100
    for attempt in range(max_attempts):
        # Randomly assign edges ensuring no cycles
        available_edges = []
        for i in range(len(all_nodes)):
            for j in range(i+1, len(all_nodes)):
                # Edge can go from i to j or j to i
                available_edges.append((all_nodes[i], all_nodes[j]))
                available_edges.append((all_nodes[j], all_nodes[i]))
        
        # Shuffle and select edges
        rng.shuffle(available_edges)
        selected_edges = []
        temp_dag = {node: [] for node in all_nodes}
        
        for parent, child in available_edges:
            if len(selected_edges) >= n_edges:
                break
                
            # Add edge if it doesn't create a cycle
            temp_dag[child].append(parent)
            if validate_dag(temp_dag):
                selected_edges.append((parent, child))
            else:
                temp_dag[child].remove(parent)
        
        if len(selected_edges) == n_edges:
            # Success! Return the shuffled DAG
            return temp_dag
    
    # Fallback: reverse all edges (guaranteed to be acyclic)
    reversed_dag = {node: [] for node in all_nodes}
    for child, parents in dag.items():
        for parent in parents:
            reversed_dag[parent].append(child)
    
    return reversed_dag


def create_missing_edges_dag(dag: Dict[int, List[int]], removal_fraction: float = 0.5, 
                           random_state: int = 42) -> Dict[int, List[int]]:
    """
    Create a DAG with some edges removed.
    
    Args:
        dag: Original DAG structure
        removal_fraction: Fraction of edges to remove (0.0 to 1.0)
        random_state: Random seed for reproducibility
        
    Returns:
        New DAG with some edges removed
    """
    rng = np.random.default_rng(random_state)
    
    # Count total edges
    edges = []
    for child, parents in dag.items():
        for parent in parents:
            edges.append((parent, child))
    
    n_edges = len(edges)
    n_to_remove = int(n_edges * removal_fraction)
    
    if n_to_remove == 0:
        return dag.copy()
    
    # Randomly select edges to remove
    edges_to_remove = rng.choice(edges, size=n_to_remove, replace=False)
    edges_to_remove_set = set(map(tuple, edges_to_remove))
    
    # Create new DAG without selected edges
    new_dag = {}
    for child, parents in dag.items():
        new_parents = []
        for parent in parents:
            if (parent, child) not in edges_to_remove_set:
                new_parents.append(parent)
        new_dag[child] = new_parents
    
    return new_dag


def create_extra_edges_dag(dag: Dict[int, List[int]], addition_fraction: float = 0.5,
                          random_state: int = 42) -> Dict[int, List[int]]:
    """
    Create a DAG with extra edges added (maintaining acyclicity).
    
    Args:
        dag: Original DAG structure  
        addition_fraction: Fraction of possible edges to add (0.0 to 1.0)
        random_state: Random seed for reproducibility
        
    Returns:
        New DAG with extra edges added
    """
    rng = np.random.default_rng(random_state)
    
    # Get all nodes
    all_nodes = set(dag.keys())
    for parents in dag.values():
        all_nodes.update(parents)
    all_nodes = sorted(list(all_nodes))
    
    # Start with copy of original DAG
    new_dag = {}
    for node in all_nodes:
        new_dag[node] = dag.get(node, []).copy()
    
    # Find existing edges
    existing_edges = set()
    for child, parents in new_dag.items():
        for parent in parents:
            existing_edges.add((parent, child))
    
    # Find all possible edges that don't exist yet
    possible_new_edges = []
    for i in range(len(all_nodes)):
        for j in range(len(all_nodes)):
            if i != j:
                edge = (all_nodes[i], all_nodes[j])
                if edge not in existing_edges:
                    possible_new_edges.append(edge)
    
    # Calculate how many edges to add
    n_current_edges = len(existing_edges)
    n_to_add = int(n_current_edges * addition_fraction)
    
    if n_to_add == 0 or len(possible_new_edges) == 0:
        return new_dag
    
    # Shuffle possible edges
    rng.shuffle(possible_new_edges)
    
    # Try to add edges without creating cycles
    added = 0
    for parent, child in possible_new_edges:
        if added >= n_to_add:
            break
            
        # Try adding edge
        new_dag[child].append(parent)
        
        # Check for cycle using existing validate_dag function
        if not validate_dag(new_dag):
            # Remove edge if it creates cycle
            new_dag[child].remove(parent)
        else:
            # Keep edge
            added += 1
    
    return new_dag


def create_dag_variations(dag: Dict[int, List[int]], random_state: int = 42) -> Dict[str, Dict[int, List[int]]]:
    """
    Create all standard DAG variations for robustness testing.
    
    Args:
        dag: Original DAG structure. This is treated as a read-only template.
        random_state: Random seed for reproducibility
        
    Returns:
        Dictionary with all DAG variations. The 'correct' variation is a reference
        to the original dag, while others are new, modified DAG objects.
    """
    variations = {
        # The original DAG is passed by reference, assuming it's not modified elsewhere.
        'correct': dag,
        'no_dag': None,
        'wrong_parents': create_wrong_parents_dag(dag, random_state),
        'missing_edges': create_missing_edges_dag(dag, 0.5, random_state),
        'extra_edges': create_extra_edges_dag(dag, 0.5, random_state)
    }
    
    return variations


def cpdag_to_dags(cpdag_adj: np.ndarray) -> List[Dict[int, List[int]]]:
    """
    Generates all possible DAGs from a CPDAG adjacency matrix.

    The CPDAG is represented by an adjacency matrix from the `causallearn` library,
    where the PC algorithm output has the following convention:
    - cpdag_adj[i, j] == -1 and cpdag_adj[j, i] == -1 means i --- j (undirected)
    - cpdag_adj[i, j] == -1 and cpdag_adj[j, i] == 1 means i --> j (directed)
    - cpdag_adj[i, j] == 1 and cpdag_adj[j, i] == -1 means j --> i (directed)

    Args:
        cpdag_adj: Adjacency matrix of the CPDAG.
        
    Returns:
        A list of all valid DAGs, where each DAG is a dictionary 
        {node: [list_of_parents]}.
    """
    num_nodes = cpdag_adj.shape[0]
    base_dag = {i: [] for i in range(num_nodes)}
    undirected_edges = []

    # 1. Separate directed and undirected edges from the CPDAG matrix
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if cpdag_adj[i, j] == -1 and cpdag_adj[j, i] == -1:
                undirected_edges.append((i, j))
            elif cpdag_adj[i, j] == -1 and cpdag_adj[j, i] == 1:
                # This is i -> j
                base_dag[j].append(i)
            elif cpdag_adj[i, j] == 1 and cpdag_adj[j, i] == -1:
                # This is j -> i
                base_dag[i].append(j)

    all_dags = []
    
    # 2. Iterate over all possible orientations of the undirected edges
    num_undirected = len(undirected_edges)
    
    # Generate all binary combinations for orientations
    for p in itertools.product([0, 1], repeat=num_undirected):
        temp_dag = {node: list(parents) for node, parents in base_dag.items()}
        
        for i in range(num_undirected):
            u, v = undirected_edges[i]
            if p[i] == 0:
                # Orient as u -> v
                temp_dag[v].append(u)
            else:
                # Orient as v -> u
                temp_dag[u].append(v)
        
        # 3. Check if the resulting orientation is a valid DAG (acyclic)
        if validate_dag(temp_dag):
            all_dags.append(temp_dag)
            
    return all_dags


def convert_cpdag_to_named_dags(cpdag_adj: np.ndarray, column_names: List[str]) -> List[Dict[str, List[str]]]:
    """
    Converts a CPDAG to a list of named DAGs.

    Args:
        cpdag_adj: Adjacency matrix of the CPDAG.
        column_names: List of column names for mapping.

    Returns:
        A list of all valid DAGs with node names.
    """
    # Get all possible DAGs with indices
    possible_dags = cpdag_to_dags(cpdag_adj)
    
    # Convert each DAG to its named representation
    named_dags = []
    for dag in possible_dags:
        named_dags.append(convert_indices_dag_to_named(dag, column_names))
        
    return named_dags


def dag_belongs_to_cpdag(dag, cpdag_adj: np.ndarray, column_names: List[str] = None) -> bool:
    """
    Check if a DAG belongs to the equivalence class represented by a CPDAG.
    
    This function determines whether a given DAG is one of the possible DAGs
    that can be generated from the provided CPDAG. The CPDAG represents a
    class of equivalent DAGs that encode the same conditional independence
    relationships.
    
    Args:
        dag: Dictionary representing the DAG structure. Can be either:
             - {node_index: [list_of_parent_indices]} for indexed DAGs
             - {node_name: [list_of_parent_names]} for named DAGs
        cpdag_adj: CPDAG adjacency matrix (numpy array) following causallearn format:
                  - cpdag_adj[i, j] == -1 and cpdag_adj[j, i] == -1: undirected edge i -- j
                  - cpdag_adj[i, j] == -1 and cpdag_adj[j, i] == 1: directed edge i -> j
                  - cpdag_adj[i, j] == 1 and cpdag_adj[j, i] == -1: directed edge j -> i
                  - cpdag_adj[i, j] == 0 and cpdag_adj[j, i] == 0: no edge
        column_names: Optional list of column names. Required if dag uses named nodes.
        
    Returns:
        bool: True if the DAG belongs to the CPDAG equivalence class, False otherwise.
        
    Raises:
        ValueError: If input formats are inconsistent or invalid.
        
    Examples:
        >>> # CPDAG: 0 -> 1 -- 2
        >>> cpdag = np.array([[0, -1, 0], [1, 0, -1], [0, -1, 0]])
        >>> dag1 = {0: [], 1: [0], 2: [1]}  # 0 -> 1 -> 2
        >>> dag2 = {0: [], 1: [0, 2], 2: []}  # 0 -> 1 <- 2
        >>> dag_belongs_to_cpdag(dag1, cpdag)  # True
        >>> dag_belongs_to_cpdag(dag2, cpdag)  # True
        
        >>> # Wrong DAG
        >>> wrong_dag = {0: [1], 1: [], 2: [0]}  # 1 -> 0 -> 2
        >>> dag_belongs_to_cpdag(wrong_dag, cpdag)  # False
    """
    
    # Handle empty cases
    if cpdag_adj.size == 0:
        return len(dag) == 0
    
    # Get the expected number of nodes from CPDAG
    expected_num_nodes = cpdag_adj.shape[0]
    
    # Check if DAG has the right number of nodes
    if len(dag) != expected_num_nodes:
        return False
    
    # Determine if the input DAG uses names or indices
    dag_nodes = set(dag.keys())
    uses_names = any(isinstance(node, str) for node in dag_nodes)
    
    if uses_names:
        if column_names is None:
            raise ValueError("column_names must be provided when DAG uses named nodes")
        if len(column_names) != expected_num_nodes:
            raise ValueError(f"column_names length ({len(column_names)}) must match CPDAG size ({expected_num_nodes})")
        
        # Convert named DAG to indexed DAG for comparison
        input_dag_indexed = convert_named_dag_to_indices(dag, column_names)
    else:
        # Check that DAG nodes are the expected indices
        expected_indices = set(range(expected_num_nodes))
        if dag_nodes != expected_indices:
            return False
        input_dag_indexed = dag
    
    # Generate all possible DAGs from the CPDAG
    try:
        possible_dags = cpdag_to_dags(cpdag_adj)
    except Exception:
        # If CPDAG is invalid, return False
        return False
    
    # Normalize DAGs for comparison (handle different parent list orders)
    def normalize_dag(dag_dict):
        """Normalize DAG by sorting parent lists and ensuring all nodes present."""
        normalized = {}
        for i in range(expected_num_nodes):
            parents = dag_dict.get(i, [])
            normalized[i] = sorted(parents)
        return normalized
    
    # Normalize input DAG
    normalized_input = normalize_dag(input_dag_indexed)
    
    # Check if the input DAG matches any possible DAG from CPDAG
    for possible_dag in possible_dags:
        normalized_possible = normalize_dag(possible_dag)
        if normalized_input == normalized_possible:
            return True
    
    return False


# Test DAG variation functions
def test_dag_variations():
    print("\n" + "=" * 60)
    print("TESTING DAG VARIATION FUNCTIONS")
    print("=" * 60)
    
    # Define test DAG using node names
    named_dag = {
        "X1": [],           # X1 is independent
        "X2": ["X1", "X3"], # X2 depends on X1 and X3
        "X3": ["X4"],       # X3 depends on X4
        "X4": []            # X4 is independent
    }
    
    # Column names and indices
    test_columns = ["X1", "X2", "X3", "X4"]
    
    # Convert to index-based DAG
    original_dag = convert_named_dag_to_indices(named_dag, test_columns)
    
    print("Original DAG (indices):", original_dag)
    print("Original DAG (names):", convert_indices_dag_to_named(original_dag, test_columns))
    print()
    
    # Test wrong parents DAG
    wrong_parents_dag = create_wrong_parents_dag(original_dag, random_state=42)
    print("\nWrong Parents DAG (indices):", wrong_parents_dag)
    print("Wrong Parents DAG (names):", convert_indices_dag_to_named(wrong_parents_dag, test_columns))
    print("Edge count preserved:", sum(len(p) for p in original_dag.values()) == 
                                 sum(len(p) for p in wrong_parents_dag.values()))
    print()
    
    # Test missing edges DAG
    missing_edges_dag = create_missing_edges_dag(original_dag, removal_fraction=0.5, random_state=42)
    print("\nMissing Edges DAG (indices):", missing_edges_dag)
    print("Missing Edges DAG (names):", convert_indices_dag_to_named(missing_edges_dag, test_columns))
    original_edges = sum(len(p) for p in original_dag.values())
    missing_edges = sum(len(p) for p in missing_edges_dag.values())
    print(f"Edge removal (~50%): {original_edges} -> {missing_edges} edges")
    print()
    
    # Test extra edges DAG
    extra_edges_dag = create_extra_edges_dag(original_dag, addition_fraction=0.5, random_state=42)
    print("\nExtra Edges DAG (indices):", extra_edges_dag)
    print("Extra Edges DAG (names):", convert_indices_dag_to_named(extra_edges_dag, test_columns))
    extra_edges = sum(len(p) for p in extra_edges_dag.values())
    print(f"Edge addition (~50%): {original_edges} -> {extra_edges} edges")
    print()
    
    # Test create_dag_variations (all variations at once)
    variations = create_dag_variations(original_dag, random_state=42)
    print("\nAll DAG Variations:")
    for name, dag in variations.items():
        if dag is None:
            print(f"  {name}: None")
        else:
            edge_count = sum(len(p) for p in dag.values())
            print(f"  {name}: {edge_count} edges, DAG (indices): {dag}")
            print(f"    DAG (names): {convert_indices_dag_to_named(dag, test_columns)}")
    
    # Visualize all DAGs with print_dag_info
    print("\nDetailed structure of each DAG variation:")
    for name, dag in variations.items():
        if dag is not None:
            print(f"\n{name.upper()} DAG:")
            print_dag_info(dag, test_columns)


def test_cpdag_conversion():
    """Tests the conversion from CPDAG to DAGs."""
    print("\n" + "=" * 60)
    print("TESTING CPDAG TO DAG CONVERSION")
    print("=" * 60)

    # Test case: 0 -> 1 -- 2, and 3 is isolated
    # CPDAG adjacency matrix from causallearn.pc
    # 0 -> 1: adj[0,1]=-1, adj[1,0]=1
    # 1--2: adj[1,2]=-1, adj[2,1]=-1
    cpdag_adj = np.array([
        [0, -1, 0, 0],
        [1, 0, -1, 0],
        [0, -1, 0, 0],
        [0, 0, 0, 0]
    ])
    
    col_names = ["X0", "X1", "X2", "X3"]
    
    # Expected DAGs (index format)
    # DAG 1: 0 -> 1 -> 2
    expected_dag1 = {0: [], 1: [0], 2: [1], 3: []}
    # DAG 2: 0 -> 1 <- 2
    expected_dag2 = {0: [], 1: [0, 2], 2: [], 3: []}
    
    # Get all DAGs
    all_dags = cpdag_to_dags(cpdag_adj)
    
    print(f"Found {len(all_dags)} possible DAGs from the CPDAG.")
    
    # Check that the two expected DAGs are found
    # We use sets of tuples for parents to make comparison order-independent
    def dag_to_comparable(dag):
        return {k: frozenset(v) for k, v in dag.items()}

    set_of_dags = {frozenset(dag_to_comparable(d).items()) for d in all_dags}
    
    assert frozenset(dag_to_comparable(expected_dag1).items()) in set_of_dags
    assert frozenset(dag_to_comparable(expected_dag2).items()) in set_of_dags
    assert len(all_dags) == 2

    print("Successfully verified the generated DAGs (index format).")
    
    # Test named conversion
    named_dags = convert_cpdag_to_named_dags(cpdag_adj, col_names)
    print("\nGenerated named DAGs:")
    for i, named_dag in enumerate(named_dags):
        print(f"DAG {i+1}: {named_dag}")
    
    # Expected named DAGs
    expected_named1 = {'X0': [], 'X1': ['X0'], 'X2': ['X1'], 'X3': []}
    expected_named2 = {'X0': [], 'X1': ['X0', 'X2'], 'X2': [], 'X3': []}

    def named_dag_to_comparable(dag):
        return {k: frozenset(v) for k, v in dag.items()}

    set_of_named_dags = {frozenset(named_dag_to_comparable(d).items()) for d in named_dags}
    
    assert frozenset(named_dag_to_comparable(expected_named1).items()) in set_of_named_dags
    assert frozenset(named_dag_to_comparable(expected_named2).items()) in set_of_named_dags
    assert len(named_dags) == 2
    
    print("\nSuccessfully verified the generated DAGs (named format).")


# Example usage and testing
if __name__ == "__main__":
    # Test with the example DAG: 0 -> 1 <- 2 <- 3
    print("=" * 60)
    print("TESTING WITH EXAMPLE DAG: 0 -> 1 <- 2 <- 3")
    print("=" * 60)
    
    example_dag = {
        1: [0, 2],  # 1 depends on 0 and 2
        2: [3],     # 2 depends on 3
        0: [],      # 0 is independent  
        3: []       # 3 is independent
    }
    
    print("DAG Dependencies: 0->1, 2->1, 3->2")
    print()
    
    # Compare all strategies
    compare_ordering_strategies(example_dag)
    
    print()
    print("Manual verification:")
    print("Order [0,1,2,3]: violations =", count_violations(example_dag, [0,1,2,3]))
    print("Order [1,0,2,3]: violations =", count_violations(example_dag, [1,0,2,3]))
    print("Best order [3,2,0,1]: violations =", count_violations(example_dag, [3,2,0,1]))
    
    print("\n" + "=" * 60)
    print("TESTING WITH ORIGINAL TEST CASE")
    print("=" * 60)
    
    # Define DAG using node names
    named_dag = {
        "X1": [],           # X1 is independent
        "X2": ["X1", "X3"], # X2 depends on X1 and X3
        "X3": ["X4"],       # X3 depends on X4
        "X4": []            # X4 is independent
    }

    # Convert to index-based DAG
    test_columns = ["X1", "X2", "X3", "X4"]
    test_dag = convert_named_dag_to_indices(named_dag, test_columns)

    print("Converted DAG:", test_dag)
    print()

    # Test the reverse conversion
    converted_back_dag = convert_indices_dag_to_named(test_dag, test_columns)
    print("Converted back to named DAG:", converted_back_dag)
    print()
    
    print("Testing DAG utilities:")
    print_dag_info(test_dag, test_columns)
    
    # Test with random data
    test_data = np.random.randn(10, 4)
    topo_order = topological_sort(test_dag)
    reordered_data, reordered_dag = reorder_data_and_dag(test_data, test_dag, topo_order)
    
    print(f"\nOriginal DAG: {test_dag}")
    print(f"Topological order: {topo_order}")
    print(f"Reordered DAG: {reordered_dag}")
    print(f"Data shape unchanged: {test_data.shape} -> {reordered_data.shape}")
    
    # Show that worst ordering gives maximum violations
    worst_order = get_worst_ordering(test_dag)
    max_violations = count_violations(test_dag, worst_order)
    total_dependencies = sum(len(parents) for parents in test_dag.values())
    
    print(f"\nWorst ordering: {worst_order}")
    print(f"Max violations: {max_violations}")
    print(f"Total dependencies: {total_dependencies}")
    print(f"Violates ALL dependencies: {max_violations == total_dependencies}")

    # Test these new functions: wrong parents, missing edges, extra edges, dag variation DAG creation functions
    test_dag_variations()

    # Test CPDAG conversion
    test_cpdag_conversion()