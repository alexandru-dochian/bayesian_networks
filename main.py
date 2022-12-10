from copy import deepcopy
import numpy as np
import pandas as pd

from BNReasoner import BNReasoner


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


def fail(actual, expected):
    assert actual == expected, "Error"


def test_implementation():
    test_network_pruning()
    test_d_separation()
    test_maxing_out()


def test_maxing_out():
    """
    : BN course 4 -> pag. 11
    :return:
    """
    initial = pd.DataFrame(columns=["B", "C", "D", "f1"], data=np.array([
        [True, True, True, 0.95],
        [True, True, False, 0.05],
        [True, False, True, 0.90],
        [True, False, False, 0.10],
        [False, True, True, 0.80],
        [False, True, False, 0.20],
        [False, False, True, 0.00],
        [False, False, False, 1.00],
    ]))

    expected = pd.DataFrame(columns=["B", "C", "max_d > f1"], data=np.array([
        [True, True, 0.95],
        [True, False, 0.9],
        [False, True, 0.8],
        [False, False, 0.1],
    ]))

    actual = BNReasoner.maxing_out("D", initial)

    # assert actual.equals(expected) is True


if __name__ == "__main__":
    test_implementation()
