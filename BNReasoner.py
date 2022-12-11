from typing import Union
import pandas as pd
from copy import deepcopy
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
        # If variable is evidence => remove outgoing edges
        for variable in e:
            for children in self.bn.get_children(variable):
                self.bn.del_edge((variable, children))

        full_set = Q.union(e)

        variables_to_check = self.bn.get_all_variables()
        while True:
            done = True
            for variable in variables_to_check:

                # If variable not in Q or E and is a leaf node => remove it
                if variable not in full_set and self.bn.is_leaf_node(variable):
                    done = False
                    self.bn.del_var(variable)
                    variables_to_check.remove(variable)

            if done:
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

    def d_separated(self, X: set[str], Y: set[str], Z: set[str]) -> bool:
        self_copy = deepcopy(self)
        self_copy.prune(Q=X.union(Y), e=Z)

        for variable_in_X in X:
            for variable_in_Y in Y:
                if self_copy.are_nodes_connected(variable_in_X, variable_in_Y):
                    return True
        return False

    @staticmethod
    def marginalization(X: str, factor: pd.DataFrame) -> pd.DataFrame:
        factor_name = factor.columns[-1]
        new_columns = list(filter(lambda variable: variable != X, factor.columns))
        new_variables = new_columns[:-1]

        # Sum out
        result = factor.groupby(new_variables).sum(factor_name)

        # Reverse dataframe to preserve initial ordering
        result = result.iloc[::-1]

        # Reset indexes from 0
        result = result.reset_index()

        # Remove the maxed out variable
        del result[X]

        # Rename factor
        result = result.rename(columns={factor_name: f"sum_{X} > {factor_name}"})

        return result

    @staticmethod
    def maxing_out(X: str, factor: pd.DataFrame) -> pd.DataFrame:
        factor_name = factor.columns[-1]
        new_columns = list(filter(lambda variable: variable != X, factor.columns))
        new_variables = new_columns[:-1]

        # Max out
        result = factor.groupby(new_variables).max(factor_name)

        # Reverse dataframe to preserve initial ordering
        result = result.iloc[::-1]

        # Reset indexes from 0
        result = result.reset_index()

        # Remove the maxed out variable
        del result[X]

        # Rename factor
        result = result.rename(columns={factor_name: f"max_{X} > {factor_name}"})

        return result

    @staticmethod
    def factor_multiplication(f: pd.DataFrame, g: pd.DataFrame) -> pd.DataFrame:
        f_factor_name = f.columns[-1]
        g_factor_name = g.columns[-1]

        # Merge dataframes        
        h = f.merge(g)
        
        # Create new factor column
        h[f"{f_factor_name} * {g_factor_name}"] = h[f_factor_name] * h[g_factor_name]

        # Delete old factor columns
        del h[f_factor_name]
        del h[g_factor_name]
        
        return h
