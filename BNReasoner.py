from typing import Union

import pandas as pd

from BayesNet import BayesNet


class BNReasoner:
    def __init__(self, net: Union[str, BayesNet]):
        """
        :param net: either file path of the bayesian network in BIFXML format or BayesNet object
        """
        if type(net) == str:
            # constructs a BN object
            self.bn = BayesNet()
            # Loads the BN from an BIFXML file
            self.bn.load_from_bifxml(net)
        else:
            self.bn = net

    def draw(self):
        self.bn.draw_structure()

    def prune(self, Q: set, e: set):
        full_set = Q.union(e)
        while True:
            for variable in self.bn.get_all_variables():
                # If variable not in Q or E and is a leaf node => remove it
                if variable not in full_set and self.bn.is_leaf_node(variable):
                    self.bn.del_var(variable)
                    continue
                # If variable is evidence => remove outgoing edges
                if variable in e:
                    for children in self.bn.get_children(variable):
                        self.bn.del_edge((variable, children))
                    continue
            break

    def are_nodes_connected(self, start, end) -> bool:
        variables = self.bn.get_all_variables()
        if start not in variables or end not in variables:
            return False

        return self.bn.exists_path(start, end) or self.bn.exists_path(end, start)

    def ordering(self, X: set[str]) -> list[str]:
        # TODO: order X based on `min-degree` and the `min-fill` heuristics
        bayes_net = self.bn
        return []

    def marginal_distribution(self, Q: set[str], e: set[str]) -> float:
        # TODO: compute P(Q|e)
        # TODO: ??? P(Q|e) = P(Q & e) / P(e)
        bayes_net = self.bn
        return 0.5

    def variable_elimination(self, variables: list[str]) -> list[str]:
        # TODO
        bayes_net = self.bn
        return []

    @staticmethod
    def compute_map():
        # TODO: ???
        pass

    @staticmethod
    def compute_mep():
        # TODO: ???
        pass

    @staticmethod
    def marginalization(X: str, factor: pd.DataFrame) -> pd.DataFrame:
        # TODO: compute the factor in which X is summed-out.
        return factor

    @staticmethod
    def maxing_out(X: str, factor: pd.DataFrame) -> pd.DataFrame:
        # TODO: compute the factor in which X is maxed-out.

        new_columns = list(filter(lambda variable: variable != X, factor.columns))
        new_data = [[]]

        return pd.DataFrame(columns=[], data=new_data)

    @staticmethod
    def factor_multiplication(f: pd.DataFrame, g: pd.DataFrame) -> pd.DataFrame:
        # TODO: compute the multiplied factor h = f * g.
        g_columns = f.columns.union(g.columns)
        h = pd.DataFrame(columns=g_columns, data=[])
        return h
