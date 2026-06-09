import logging
import time
import warnings

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from causallearn.utils.cit import chisq
from sklearn.preprocessing import KBinsDiscretizer

# Use patched skeleton discovery (adds local_skeleton_discovery and CI test caching)
from ._skeleton import skeleton_discovery, local_skeleton_discovery

# Fix relative import issue
try:
    from .time_series import drop_extra
except ImportError:
    # Fallback to absolute import
    from time_series import drop_extra

warnings.filterwarnings("ignore")
plt.style.use("fivethirtyeight")

logger = logging.getLogger(__name__)


# =========== UTILS.py ====================
# Note: Some of the functions defined here are only used for data
# from sock-shop or real-world application.
CI_TEST = chisq

START_ALPHA = 0.001
ALPHA_STEP = 0.1
ALPHA_LIMIT = 1

VERBOSE = False
F_NODE = "F-node"


def drop_constant(df):
    return df.loc[:, (df != df.iloc[0]).any()]


# Only used for sock-shop and real outage datasets
def preprocess_sock_shop(n_df, a_df, per, dk_select_useful=False):
    _process = lambda df: _select_lat(_scale_down_mem(_rm_time(df)), per)

    n_df = _process(n_df)
    a_df = _process(a_df)

    n_df = drop_constant(n_df)
    a_df = drop_constant(a_df)

    n_df, a_df = _match_columns(n_df, a_df)

    df = add_fnode_and_concat(n_df, a_df)

    if dk_select_useful is True:
        df = _select_useful_cols(df)

    n_df = df[df[F_NODE] == "0"].drop(columns=[F_NODE])
    a_df = df[df[F_NODE] == "1"].drop(columns=[F_NODE])

    return (n_df, a_df)


def load_datasets(normal, anomalous):
    normal_df = pd.read_csv(normal)
    anomalous_df = pd.read_csv(anomalous)
    return (normal_df, anomalous_df)


def add_fnode_and_concat(normal_df, anomalous_df):
    normal_df[F_NODE] = "0"
    anomalous_df[F_NODE] = "1"
    return pd.concat([normal_df, anomalous_df])


# Run PC (only the skeleton phase) on the given dataset.
# The last column of the data *must* be the F-node
def run_pc(data, alpha, localized=False, labels={}, mi=[], verbose=VERBOSE):
    t0 = time.time()
    if labels == {}:
        labels = {i: name for i, name in enumerate(data.columns)}

    np_data = data.to_numpy()
    logger.info("[RCD.run_pc] Starting PC skeleton discovery: %d rows x %d cols, alpha=%.3f, localized=%s",
                np_data.shape[0], np_data.shape[1], alpha, localized)

    if localized:
        f_node = np_data.shape[1] - 1
        cg = local_skeleton_discovery(
            np_data,
            f_node,
            alpha,
            indep_test=CI_TEST,
            mi=mi,
            labels=labels,
            verbose=verbose,
        )
    else:
        cg = skeleton_discovery(
            np_data,
            alpha,
            indep_test=CI_TEST,
            background_knowledge=None,
            stable=False,
            verbose=verbose,
            labels=labels,
            show_progress=False,
        )

    cg.to_nx_graph()
    logger.info("[RCD.run_pc] PC done in %.2fs, CI tests=%d", time.time() - t0, cg.no_ci_tests)
    return cg


def get_fnode_child(G):
    return [*G.successors(F_NODE)]


def save_graph(graph, file):
    nx.draw_networkx(graph)
    plt.savefig(file)


def pc_with_fnode(normal_df, anomalous_df, alpha, bins=None, localized=False, verbose=VERBOSE):
    data = _preprocess_for_fnode(normal_df, anomalous_df, bins)
    cg = run_pc(data, alpha, localized=localized, verbose=verbose)
    return cg.nx_graph


# Equivelant to \Psi-PC from the main paper
def run_psi_pc(
    normal_df,
    anomalous_df,
    bins=None,
    mi=None,  # TODO: this is just wrong, refactor it
    localized=False,
    start_alpha=None,
    min_nodes=-1,
    verbose=VERBOSE,
):
    t0 = time.time()
    logger.info("[RCD.run_psi_pc] Starting: normal=%d rows x %d cols, anomalous=%d rows x %d cols, min_nodes=%d",
                len(normal_df), len(normal_df.columns), len(anomalous_df), len(anomalous_df.columns), min_nodes)
    """
    Run Psi-PC on the given dataset.
    The last column of the data *must* be the F-node

    Parameters
    ----------
    normal_df: pd.DataFrame
        Normal data
    anomalous_df: pd.DataFrame
        Anomalous data
    bins: int
        Number of bins to use for discretization
    mi: list
        List of tuples of mutual information
    localized: bool
        Whether to use localized PC
    start_alpha: float
        Starting alpha value
    min_nodes: int
        Minimum number of nodes to order
    verbose: bool
        Whether to print verbose output

    Returns
    -------
    # TODO: refactor this
    """

    if mi is None:
        mi = []
    if 0 in [len(normal_df.columns), len(anomalous_df.columns)]:
        return ([], None, [], 0)
    data = _preprocess_for_fnode(normal_df, anomalous_df, bins)

    if min_nodes == -1:
        # Order all nodes (if possible) except F-node
        min_nodes = len(data.columns) - 1
    assert min_nodes < len(data)

    G = None
    no_ci = 0
    i_to_labels = {i: name for i, name in enumerate(data.columns)}
    labels_to_i = {name: i for i, name in enumerate(data.columns)}

    _preprocess_mi = lambda l: [labels_to_i.get(i) for i in l]  # noqa
    _postprocess_mi = lambda l: [i_to_labels.get(i) for i in list(filter(None, l))]  # noqa
    processed_mi = _preprocess_mi(mi)
    _run_pc = lambda alpha: run_pc(
        data,
        alpha,
        localized=localized,
        mi=processed_mi,
        labels=i_to_labels,
        verbose=verbose,
    )

    rc = []
    _alpha = START_ALPHA if start_alpha is None else start_alpha
    alpha_iter = 0
    for i in np.arange(_alpha, ALPHA_LIMIT, ALPHA_STEP):
        alpha_iter += 1
        logger.info("[RCD.run_psi_pc] Alpha iteration %d: alpha=%.3f, rc_so_far=%d", alpha_iter, i, len(rc))
        cg = _run_pc(i)
        G = cg.nx_graph
        no_ci += cg.no_ci_tests

        if G is None:
            continue

        f_neigh = get_fnode_child(G)
        new_neigh = [x for x in f_neigh if x not in rc]
        if len(new_neigh) == 0:
            continue
        else:
            f_p_values = cg.p_values[-1][[labels_to_i.get(key) for key in new_neigh]]
            rc += _order_neighbors(new_neigh, f_p_values)

        if len(rc) == min_nodes:
            break

    logger.info("[RCD.run_psi_pc] Done in %.2fs: %d alpha iters, %d root causes found, %d CI tests",
                time.time() - t0, alpha_iter, len(rc), no_ci)
    return (rc, G, _postprocess_mi(cg.mi), no_ci)


def _order_neighbors(neigh, p_values):
    _neigh = neigh.copy()
    _p_values = p_values.copy()
    stack = []

    while len(_neigh) != 0:
        i = np.argmax(_p_values)
        node = _neigh[i]
        stack = [node] + stack
        _neigh.remove(node)
        _p_values = np.delete(_p_values, i)
    return stack


# ==================== Private methods =============================

_rm_time = lambda df: df.loc[:, ~df.columns.isin(["time"])]
_list_intersection = lambda l1, l2: [x for x in l1 if x in l2]


def _preprocess_for_fnode(normal_df, anomalous_df, bins):
    df = add_fnode_and_concat(normal_df, anomalous_df)
    if df is None:
        return None

    return _discretize(df, bins) if bins is not None else df


def _select_useful_cols(df):
    i = df.loc[:, df.columns != F_NODE].std() > 1
    cols = i[i].index.tolist()
    cols.append(F_NODE)
    if len(cols) == 1:
        return None
    elif len(cols) == len(df.columns):
        return df

    print(f"Drop {len(df.columns) - len(cols)} columns, left with {len(cols)}")

    return df[cols]


# Only select the metrics that are in both datasets
def _match_columns(n_df, a_df):
    cols = _list_intersection(n_df.columns, a_df.columns)
    return (n_df[cols], a_df[cols])


# Convert all memeory columns to MBs
def _scale_down_mem(df):
    def update_mem(x):
        if not x.name.endswith("_mem"):
            return x
        x /= 1e6
        x = x.astype(int)
        return x

    return df.apply(update_mem)


# Select all the non-latency columns and only select latecy columns
# with given percentaile
def _select_lat(df, per):
    return df.filter(regex=(r".*(?<!lat_\d{2})$|_lat_" + str(per) + "$"))


# NOTE: THIS FUNCTION THROWS WARNGINGS THAT ARE SILENCED!
def _discretize(data, bins):
    t0 = time.time()
    d = data.iloc[:, :-1]
    logger.info("[RCD._discretize] Fitting KBinsDiscretizer on %d rows x %d cols, bins=%d", len(d), len(d.columns), bins)
    discretizer = KBinsDiscretizer(n_bins=bins, encode="ordinal", strategy="kmeans")
    discretizer.fit(d)
    disc_d = discretizer.transform(d)
    disc_d = pd.DataFrame(disc_d, columns=d.columns.values.tolist())
    disc_d[F_NODE] = data[F_NODE].tolist()

    for c in disc_d:
        disc_d[c] = disc_d[c].astype(int)

    logger.info("[RCD._discretize] Done in %.2fs", time.time() - t0)
    return disc_d


# =========== UTILS.py ====================

# np.random.seed(0)

# LOCAL_ALPHA has an effect on execution time. Too strict alpha will produce a sparse graph
# so we might need to run phase-1 multiple times to get up to k elements. Too relaxed alpha
# will give dense graph so the size of the separating set will increase and phase-1 will
# take more time.
# We tried a few different values and found that 0.01 gives the best result in our case
# (between 0.001 and 0.1).
LOCAL_ALPHA = 0.01
DEFAULT_GAMMA = 5


# Split the dataset into multiple subsets
def create_chunks(df, gamma):
    chunks = list()
    names = np.random.permutation(df.columns)
    for i in range(df.shape[1] // gamma + 1):
        chunks.append(names[i * gamma : (i * gamma) + gamma])

    if len(chunks[-1]) == 0:
        chunks.pop()
    logger.info("[RCD.create_chunks] %d cols -> %d chunks (gamma=%d)", len(df.columns), len(chunks), gamma)
    return chunks


def run_level(normal_df, anomalous_df, gamma, localized, bins, verbose):
    """
    Run phase-1 of RCD algorithm

    Parameters
    ----------
    normal_df : pandas.DataFrame
        Normal data
    anomalous_df : pandas.DataFrame
        Anomalous data
    gamma : int
        Number of nodes in each subset
    localized : bool
        Run localized version of PSI-PC
    bins : int
        Number of bins
    verbose : bool
        Verbose mode

    Returns
    -------
    f_child_union : list
        List of nodes in the separating set
    mi_union : list
        List of mutual information values
    ci_tests : int
        Number of conditional independence tests
    """
    t0 = time.time()
    ci_tests = 0
    chunks = create_chunks(normal_df, gamma)
    logger.info("[RCD.run_level] Starting phase-1 level: %d cols -> %d chunks", len(normal_df.columns), len(chunks))

    f_child_union = []
    mi_union = []
    f_child = []
    for idx, c in enumerate(chunks):
        logger.info("[RCD.run_level] Chunk %d/%d: %d cols %s", idx + 1, len(chunks), len(c), list(c))
        tc0 = time.time()
        # Try this segment with multiple values of alpha until we find at least one node
        rc, _, mi, ci = run_psi_pc(
            normal_df.loc[:, c],
            anomalous_df.loc[:, c],
            bins=bins,
            localized=localized,
            start_alpha=LOCAL_ALPHA,
            min_nodes=1,
            verbose=verbose,
        )
        logger.info("[RCD.run_level] Chunk %d/%d done in %.2fs, found %d root causes: %s",
                    idx + 1, len(chunks), time.time() - tc0, len(rc), rc)
        f_child_union += rc
        mi_union += mi
        ci_tests += ci
        if verbose:
            f_child.append(rc)

    if verbose:
        print(f"Output of individual chunk {f_child}")
        print(f"Total nodes in mi => {len(mi_union)} | {mi_union}")

    logger.info("[RCD.run_level] Phase-1 level done in %.2fs: %d candidates total, %d CI tests",
                time.time() - t0, len(f_child_union), ci_tests)
    return f_child_union, mi_union, ci_tests


def run_multi_phase(normal_df, anomalous_df, gamma, localized, bins, verbose):
    """
    Run RCD algorithm with two phases (phase-1 and phase-2) to find the root causes of the anomaly in the data.

    Parameters
    ----------
    normal_df : pandas.DataFrame
        Normal data
    anomalous_df : pandas.DataFrame
        Anomalous data
    gamma : int
        Number of nodes in each subset
    localized : bool
        Run localized version of PSI-PC
    bins : int
        Number of bins for discretization
    verbose : bool
        Verbose mode


    Returns
    -------
    rc : list
        List of root causes
    """
    t_total = time.time()
    logger.info("[RCD.run_multi_phase] Starting: normal=%d rows x %d cols, anomalous=%d rows x %d cols, gamma=%d",
                len(normal_df), len(normal_df.columns), len(anomalous_df), len(anomalous_df.columns), gamma)

    f_child_union = normal_df.columns
    mi_union = []
    i = 0
    prev = len(f_child_union)

    # Phase-1
    logger.info("[RCD.run_multi_phase] === Phase-1 START ===")
    while True:
        start = time.time()
        logger.info("[RCD.run_multi_phase] Phase-1 Level-%d: %d variables to process", i, len(f_child_union))
        f_child_union, mi, ci_tests = run_level(
            normal_df.loc[:, f_child_union],
            anomalous_df.loc[:, f_child_union],
            gamma,
            localized,
            bins,
            verbose,
        )
        elapsed = time.time() - start
        logger.info("[RCD.run_multi_phase] Phase-1 Level-%d done in %.2fs: %d candidates remain", i, elapsed, len(f_child_union))
        if verbose:
            print(f"Level-{i}: variables {len(f_child_union)} | time {time.time() - start}")
        i += 1
        mi_union += mi
        # Phase-1 with only one level
        # break

        len_child = len(f_child_union)
        # If found gamma nodes or if running the current level did not remove any node
        if len_child <= gamma or len_child == prev:
            break
        prev = len(f_child_union)

    logger.info("[RCD.run_multi_phase] === Phase-1 END: %d levels, %d candidates ===", i, len(f_child_union))

    # Phase-2
    logger.info("[RCD.run_multi_phase] === Phase-2 START: %d variables ===", len(f_child_union))
    t_p2 = time.time()
    mi_union = []
    new_nodes = f_child_union
    rc, _, mi, ci = run_psi_pc(
        normal_df.loc[:, new_nodes],
        anomalous_df.loc[:, new_nodes],
        bins=bins,
        mi=mi_union,
        localized=localized,
        verbose=verbose,
    )
    ci_tests += ci
    logger.info("[RCD.run_multi_phase] === Phase-2 END in %.2fs: %d root causes ===", time.time() - t_p2, len(rc))

    logger.info("[RCD.run_multi_phase] TOTAL done in %.2fs. Root causes: %s", time.time() - t_total, rc)
    # return rc, ci_tests
    return rc


def rcd(
    data,
    inject_time,
    dk_select_useful=False,
    gamma=5,
    localized=True,
    bins=5,
    verbose=False,
    dataset=None,
    seed=None,
    **kwargs,
):
    t_total = time.time()
    logger.info("[RCD] ========== RCD START ==========")
    logger.info("[RCD] Input data: %d rows x %d cols", len(data), len(data.columns))

    # Remove duplicate columns (keep first occurrence)
    dup_cols = data.columns[data.columns.duplicated()].unique().tolist()
    if dup_cols:
        logger.info("[RCD] Removing %d duplicate column(s): %s", len(dup_cols), dup_cols)
        data = data.loc[:, ~data.columns.duplicated()]
    logger.info("[RCD] After dedup: %d cols", len(data.columns))

    normal_df = data[data["time"] < inject_time]
    anomal_df = data[data["time"] >= inject_time]

    # Drop time column — only metric columns should be analyzed
    if "time" in normal_df.columns:
        normal_df = normal_df.drop(columns=["time"])
        anomal_df = anomal_df.drop(columns=["time"])
    logger.info("[RCD] Split data: normal=%d rows, anomalous=%d rows, metric cols=%d",
                len(normal_df), len(anomal_df), len(normal_df.columns))

    if dk_select_useful is True:
        logger.info("[RCD] Applying drop_extra filter")
        normal_df = drop_extra(normal_df)
        anomal_df = drop_extra(anomal_df)
        logger.info("[RCD] After drop_extra: %d metric cols", len(normal_df.columns))

    # if dataset == real outages:
    if dataset == "sock-shop":
        logger.info("[RCD] Applying sock-shop preprocessing")
        normal_df, anomal_df = preprocess_sock_shop(normal_df, anomal_df, 90, dk_select_useful)
    elif dataset is not None:
        logger.info("[RCD] Applying dataset=%s preprocessing", dataset)
        from .time_series import convert_mem_mb, drop_constant, drop_time, preprocess

        normal_df = drop_constant(convert_mem_mb(drop_time(normal_df)))
        anomal_df = drop_constant(convert_mem_mb(drop_time(anomal_df)))

        normal_df, anomal_df = _match_columns(normal_df, anomal_df)

        df = add_fnode_and_concat(normal_df, anomal_df)
        if dk_select_useful is True:
            df = _select_useful_cols(df)

        normal_df = df[df[F_NODE] == "0"].drop(columns=[F_NODE])
        anomal_df = df[df[F_NODE] == "1"].drop(columns=[F_NODE])

    logger.info("[RCD] Final data for analysis: normal=%d rows x %d cols, anomalous=%d rows x %d cols",
                len(normal_df), len(normal_df.columns), len(anomal_df), len(anomal_df.columns))
    logger.info("[RCD] Metric columns: %s", list(normal_df.columns))

    # Always set seed for reproducibility (default 42)
    np.random.seed(seed if seed is not None else 42)

    rc = run_multi_phase(normal_df, anomal_df, gamma, localized, bins, verbose)
    logger.info("[RCD] ========== RCD END in %.2fs. Root causes: %s ==========", time.time() - t_total, rc)
    # return rc
    return {
        "ranks": rc,
    }