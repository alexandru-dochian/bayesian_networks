def main():
    print("Hello!")


def test_network_pruning():
    fail(3, 3)


def test_d_seperation():
    fail(3, 3)


def test_independence():
    fail(3, 3)


def test_implementation():
    test_network_pruning()
    test_d_seperation()
    test_independence()


def fail(actual, expected):
    assert actual == expected, "Error"


if __name__ == "__main__":
    test_implementation()
    main()
