import typing
from typing import Union, Dict, List
import pandas as pd
from copy import deepcopy
from BayesNet import BayesNet
import networkx as nx


class BNReasoner:
    def __init__(self, net: Union[str, BayesNet]):
        self.instantiation = {}
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

    def compute_map(self, Q: Dict[str, bool], e: Dict[str, bool]) -> typing.Tuple[float, typing.Dict[str, bool]]:
        self_copy = deepcopy(self)
        self_copy.prune(Q=Q, e=e)

        factors = [self_copy.bn.get_cpt(variable) for variable in self_copy.bn.get_all_variables()]

        # Multiply remaining factors
        resulting_factor = BNReasoner.multiply_all_factors_together(factors)

        # First sum out V \ Q
        not_in_q = list(filter(lambda variable : variable not in Q, self.bn.get_all_variables()))
        for variable_to_remove in not_in_q:
            resulting_factor = BNReasoner.marginalization(variable_to_remove, resulting_factor)

        # Then max out variables in Q
        variables_in_q = Q.keys()
        for variable_to_remove in variables_in_q:
            resulting_factor = self_copy.maxing_out(variable_to_remove, resulting_factor)

        return resulting_factor.iloc[0][resulting_factor.columns.size - 1], self_copy.instantiation

    def compute_mpe(self, e: Dict[str, bool]) -> typing.Dict[str, bool]:
        Q = dict([(key, None) for key in list(
            filter(lambda variable: variable not in e,
                   self.bn.get_all_variables())
        )])
        self_copy = deepcopy(self)
        self_copy.prune(Q=Q, e=e)

        factors = [self_copy.bn.get_cpt(variable) for variable in self_copy.bn.get_all_variables()]

        # Multiply remaining factors
        resulting_factor = BNReasoner.multiply_all_factors_together(factors)

        variables_to_remove = list(filter(lambda variable: variable in Q, self_copy.bn.get_all_variables()))
        for variable_to_remove in variables_to_remove:
            resulting_factor = self_copy.maxing_out(variable_to_remove, resulting_factor)

        return resulting_factor.iloc[0][resulting_factor.columns.size - 1], self_copy.instantiation

    def marginal_distribution(self, Q: Dict[str, bool], e: Dict[str, bool]) -> pd.DataFrame:
        self_copy = deepcopy(self)
        self_copy.prune(Q=Q, e=e)

        factors = [self_copy.bn.get_cpt(variable) for variable in self_copy.bn.get_all_variables()]

        # Find variables to remove
        variables_to_remove = list(filter(lambda variable: variable not in Q, self_copy.bn.get_all_variables()))

        # Remove all other variables
        factors = BNReasoner.variable_elimination(variables_to_remove, factors, self.bn.get_interaction_graph())

        # Filter out possible trivial factors
        factors = list(filter(lambda factor: factor is not None, factors))

        # Multiply remaining factors
        resulting_factor = BNReasoner.multiply_all_factors_together(factors)

        return resulting_factor

    @staticmethod
    def multiply_all_factors_together(factors: typing.List[pd.DataFrame]) -> pd.DataFrame:
        resulting_factor = None
        for factor in factors:
            if resulting_factor is None:
                resulting_factor = factor
                continue
            resulting_factor = BNReasoner.factor_multiplication(factor, resulting_factor)

        return resulting_factor

    def draw(self):
        self.bn.draw_structure()

    def prune(self, Q: Dict[str, bool], e: Dict[str, bool]):
        # If variable is evidence => remove outgoing edges
        for variable in e:
            for children in self.bn.get_children(variable):
                reduced_factor = self.bn.reduce_factor(pd.Series({variable: e[variable]}), self.bn.get_cpt(children))
                self.bn.update_cpt(children, reduced_factor)
                self.bn.del_edge((variable, children))

        full_set = {}
        if len(Q.keys()) > 0:
            full_set.update(Q)
        if len(e.keys()) > 0:
            full_set.update(e)

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

        # Building a bidirectional copy of the graph
        bidirectional_graph = self.bn.structure.to_undirected()

        # Checking path using `networkx` builtin functionality
        return nx.has_path(bidirectional_graph, start, end)

    @staticmethod
    def variable_elimination(variables_to_remove: list[str], factors: List[pd.DataFrame],
                             interaction_graph: nx.Graph) -> List[typing.Union[pd.DataFrame, None]]:
        variables_to_remove = BNReasoner.ordering(variables_to_remove, interaction_graph)

        for variable_to_remove in variables_to_remove:
            factors = list(map(lambda factor: BNReasoner.marginalization(variable_to_remove, factor), factors))
            factors = list(filter(lambda factor: factor is not None, factors))

        return factors

    @staticmethod
    def ordering(variables: List[str], interaction_graph: nx.Graph) -> List[str]:
        # TODO: order X based on `min-degree` and the `min-fill` heuristics
        return variables

    def independence(self, X: Dict[str, bool], Y: Dict[str, bool], Z: Dict[str, bool]) -> bool:
        return self.d_separated(X, Y, Z)

    def d_separated(self, X: Dict[str, bool], Y: Dict[str, bool], Z: Dict[str, bool]) -> bool:
        self_copy = deepcopy(self)
        Q = deepcopy(X)
        Q.update(Y)
        self_copy.prune(Q=Q, e=Z)

        for variable_in_X in X:
            for variable_in_Y in Y:
                if self_copy.are_nodes_connected(variable_in_X, variable_in_Y):
                    return True
        return False

    @staticmethod
    def marginalization(X: str, factor: pd.DataFrame) -> typing.Union[pd.DataFrame, None]:
        if X not in factor.columns:
            return factor

        factor_name = factor.columns[-1]
        new_columns = list(filter(lambda variable: variable != X, factor.columns))
        new_variables = new_columns[:-1]

        if len(new_variables) == 0:
            return None

        # Sum out
        result = factor.groupby(new_variables).sum(factor_name)

        # Reverse dataframe to preserve initial ordering
        result = result.iloc[::-1]

        # Reset indexes from 0
        result = result.reset_index()

        # Remove the summed out variable
        del result[X]

        # Rename factor
        result = result.rename(columns={factor_name: f"sum_{X} > {factor_name}"})

        return result

    def maxing_out(self, X: str, factor: pd.DataFrame) -> pd.DataFrame:
        if X not in factor:
            return factor

        factor_name = factor.columns[-1]
        new_columns = list(filter(lambda variable: variable != X, factor.columns))
        new_variables = new_columns[:-1]

        # Max out
        if len(new_variables) == 0:
            row = factor.loc[factor[factor_name].idxmax()]
            value = row[X]
            result = pd.DataFrame(row)
        else:
            result = factor.groupby(new_variables).max(factor_name)

            # Reverse dataframe to preserve initial ordering
            result = result.iloc[::-1]

            # Reset indexes from 0
            result = result.reset_index()

            # Remove the maxed out variable
            value = result[X][0]

        if X in result:
            del result[X]

        # Rename factor
        result = result.rename(columns={factor_name: f"max_{X}={value} > {factor_name}"})
        self.instantiation[X] = value
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
        if f_factor_name in h:
            del h[f_factor_name]

        if g_factor_name in h:
            del h[g_factor_name]

        return h
