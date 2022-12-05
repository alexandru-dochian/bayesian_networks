from copy import deepcopy

import pandas as pd

from BNReasoner import BNReasoner
from BayesNet import BayesNet


def main():
    pass


# -------- Implementation --------
def marginalization(X: str, factor: pd.DataFrame) -> pd.DataFrame:
    # TODO: compute the factor in which X is summed-out.
    return factor


def maxing_out(X: str, factor: pd.DataFrame) -> pd.DataFrame:
    # TODO: compute the factor in which X is maxed-out.
    return factor


def factor_multiplication(f: pd.DataFrame, g: pd.DataFrame) -> pd.DataFrame:
    # TODO: compute the multiplied factor h = f * g.
    g_columns = f.columns.union(g.columns)
    h = pd.DataFrame(columns=g_columns, data=[])
    return h


def ordering(X: set[str], bayes_network: BayesNet) -> list[str]:
    # TODO: order X based on `min-degree` and the `min-fill` heuristics
    return []


def marginal_distribution(Q: set[str], e: set[str], bayes_network: BayesNet) -> float:
    # TODO: compute P(Q|e)
    # TODO: ??? P(Q|e) = P(Q & e) / P(e)
    return 0.5


def compute_map():
    # TODO: ???
    pass


def compute_mep():
    # TODO: ???
    pass


def variable_elimination(variables: list[str], bayes_network: BayesNet) -> list[str]:
    return []


# -------- Tests --------
def test_network_pruning():
    dog_bayes_network = BNReasoner("testing/dog_problem.BIFXML")
    dog_pruned_bayes_network = deepcopy(dog_bayes_network)

    Q = set()
    e = set()
    # TODO assert P(Q, e) == P(Q, e) for both networks


def test_d_separation():
    dog_problem = BNReasoner("testing/dog_problem.BIFXML")

    X = "family-out"
    Y = "hear-bark"
    Z = "dog-out"

    fail(dog_problem.are_nodes_connected(X, Y), True)

    Q = set()
    e = {Z}
    dog_problem.prune(Q, e)
    fail(dog_problem.are_nodes_connected(X, Y), False)

def test_implementation():
    test_network_pruning()
    test_d_separation()


def fail(actual, expected):
    assert actual == expected, "Error"


if __name__ == "__main__":
    test_implementation()
    main()
