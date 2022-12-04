from typing import Union
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
                if variable not in full_set and self.bn.is_leaf_node(variable):
                    self.bn.del_var(variable)
                    continue

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

