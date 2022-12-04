from copy import deepcopy

from BNReasoner import BNReasoner


def main():
    pass


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
