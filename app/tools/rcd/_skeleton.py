"""
Modified SkeletonDiscovery from causal-learn.

Patches applied (from RCD source):
- Added: local_skeleton_discovery() for localized F-node neighborhood discovery
- Modified: skeleton_discovery() sets data_hash_key for caching
"""

from itertools import combinations

import numpy as np
from tqdm.auto import tqdm

from causallearn.utils.cit import chisq, gsq, CIT
from causallearn.utils.PCUtils.Helper import append_value

# Use the patched CausalGraph instead of the original
from ._graph_class import CausalGraph


def _make_cit(data, indep_test):
    """Convert string method name to CIT callable (new causal-learn API)."""
    if isinstance(indep_test, str):
        return CIT(data, indep_test)
    return indep_test


def skeleton_discovery(data, alpha, indep_test, stable=True, background_knowledge=None,
                       labels={}, verbose=False, show_progress=True):
    '''
    Perform skeleton discovery

    Parameters
    ----------
    data : data set (numpy ndarray), shape (n_samples, n_features).
    alpha: float, desired significance level of independence tests in (0,1)
    indep_test : the function of the independence test being used
    stable : run stabilized skeleton discovery if True (default = True)
    background_knowledge : background knowledge
    verbose : True iff verbose output should be printed.
    show_progress : True iff the algorithm progress should be show in console.

    Returns
    -------
    cg : a CausalGraph object.
    '''

    assert type(data) == np.ndarray
    assert 0 < alpha < 1

    no_of_var = data.shape[1]
    cg = CausalGraph(no_of_var, labels=labels)
    cg.data = data
    cg.data_hash_key = hash(str(data))
    # Create proper callable from string method name (new causal-learn API)
    indep_test = _make_cit(data, indep_test)
    cg.set_ind_test(indep_test)

    depth = -1
    pbar = tqdm(total=no_of_var, disable=not show_progress)
    while cg.max_degree() - 1 > depth:
        depth += 1
        edge_removal = []
        if show_progress: pbar.reset()
        for x in range(no_of_var):
            if show_progress: pbar.update()
            if show_progress: pbar.set_description(f'Depth={depth}, working on node {x}')
            Neigh_x = cg.neighbors(x)
            if len(Neigh_x) < depth - 1:
                continue
            for y in Neigh_x:
                knowledge_ban_edge = False
                sepsets = set()
                if background_knowledge is not None and (
                        background_knowledge.is_forbidden(cg.G.nodes[x], cg.G.nodes[y])
                        and background_knowledge.is_forbidden(cg.G.nodes[y], cg.G.nodes[x])):
                    knowledge_ban_edge = True
                if knowledge_ban_edge:
                    if not stable:
                        edge1 = cg.G.get_edge(cg.G.nodes[x], cg.G.nodes[y])
                        if edge1 is not None:
                            cg.G.remove_edge(edge1)
                        edge2 = cg.G.get_edge(cg.G.nodes[y], cg.G.nodes[x])
                        if edge2 is not None:
                            cg.G.remove_edge(edge2)
                        append_value(cg.sepset, x, y, ())
                        append_value(cg.sepset, y, x, ())
                        break
                    else:
                        edge_removal.append((x, y))
                        edge_removal.append((y, x))

                Neigh_x_noy = np.delete(Neigh_x, np.where(Neigh_x == y))
                for S in combinations(Neigh_x_noy, depth):
                    p = cg.ci_test(x, y, S)
                    if p > alpha:
                        if verbose: print('%d ind %d | %s with p-value %f\n' % (x, y, S, p))
                        if not stable:
                            edge1 = cg.G.get_edge(cg.G.nodes[x], cg.G.nodes[y])
                            if edge1 is not None:
                                cg.G.remove_edge(edge1)
                            edge2 = cg.G.get_edge(cg.G.nodes[y], cg.G.nodes[x])
                            if edge2 is not None:
                                cg.G.remove_edge(edge2)
                            append_value(cg.sepset, x, y, S)
                            append_value(cg.sepset, y, x, S)
                            break
                        else:
                            edge_removal.append((x, y))
                            edge_removal.append((y, x))
                            for s in S:
                                sepsets.add(s)
                    else:
                        append_value(cg.p_values, x, y, p)
                        if verbose: print('%d dep %d | %s with p-value %f\n' % (x, y, S, p))
                append_value(cg.sepset, x, y, tuple(sepsets))
                append_value(cg.sepset, y, x, tuple(sepsets))

        if show_progress: pbar.refresh()

        for (x, y) in list(set(edge_removal)):
            edge1 = cg.G.get_edge(cg.G.nodes[x], cg.G.nodes[y])
            if edge1 is not None:
                cg.G.remove_edge(edge1)

    if show_progress: pbar.close()

    return cg


# --- RCD addition: localized skeleton discovery ---
def local_skeleton_discovery(data, local_node, alpha, indep_test, mi=[], labels={}, verbose=False):
    """
    Localized PC skeleton discovery that only discovers the neighborhood
    of the F-node. This is the core of the RCD algorithm's efficiency.

    Parameters
    ----------
    data : numpy ndarray
    local_node : int, index of the F-node
    alpha : float, significance level
    indep_test : independence test function
    mi : list, marginally independent node indices (to skip)
    labels : dict, node index to name mapping
    verbose : bool
    """
    assert type(data) == np.ndarray
    assert local_node <= data.shape[1]
    assert 0 < alpha < 1

    no_of_var = data.shape[1]
    cg = CausalGraph(no_of_var, labels=labels)
    cg.data = data
    cg.data_hash_key = hash(str(data))
    # Create proper callable from string method name (new causal-learn API)
    indep_test = _make_cit(data, indep_test)
    cg.set_ind_test(indep_test)

    depth = -1
    x = local_node
    # Remove edges between nodes in MI and F-node
    for i in mi:
        cg.remove_edge(x, i)

    while cg.max_degree() - 1 > depth:
        depth += 1

        local_neigh = np.random.permutation(cg.neighbors(x))
        for y in local_neigh:
            Neigh_y = cg.neighbors(y)
            Neigh_y = np.delete(Neigh_y, np.where(Neigh_y == x))
            Neigh_y_f = []
            if depth > 0:
                Neigh_y_f = [s for s in Neigh_y if x in cg.neighbors(s)]

            for S in combinations(Neigh_y_f, depth):
                p = cg.ci_test(x, y, S)
                if p > alpha:
                    if verbose: print(f'{cg.labels[x]} ind {cg.labels[y]} | {[cg.labels[s] for s in S]} with p-value {p}')
                    cg.remove_edge(x, y)
                    append_value(cg.sepset, x, y, S)
                    append_value(cg.sepset, y, x, S)

                    if depth == 0:
                        cg.append_to_mi(y)
                    break
                else:
                    append_value(cg.p_values, x, y, p)
                    if verbose: print(f'{cg.labels[x]} dep {cg.labels[y]} | {[cg.labels[s] for s in S]} with p-value {p}')

    return cg
# --- end RCD addition ---