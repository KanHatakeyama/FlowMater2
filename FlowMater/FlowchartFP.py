import copy
from typing import List, Dict,Union
import networkx as nx


class FlowchartFP:
    """
    calculate fingerprints of flowcharts
    """

    def __init__(self, g_list: List[nx.DiGraph], dict_mode: bool = False) -> None:
        """
        Parameters
        ----------
        g_list : list
            list of networkX (graph) objects
        """

        self.dict_mode = dict_mode
        # analyze graph list and set node list
        self.all_node_val_list = []
        for g in g_list:
            self.all_node_val_list.extend(analyze_node_connections(g))
        self.all_node_val_list = list(set(self.all_node_val_list))

        self.template_dict = {i: 0 for i in range(len(self.all_node_val_list))}
        self.v_to_i = {v: i for i, v in enumerate(self.all_node_val_list)}
        self.i_to_v = {i: v for i, v in enumerate(self.all_node_val_list)}

    def __call__(self, g: nx.DiGraph) -> Union[List[int], dict]:
        """
        calculate fingerprints of flowcharts

        Parameters
        ----------
        g : networkX object
            graph object

        Returns
        -------
        fp : list of integer
            fingerprint

        """

        node_val_list = analyze_node_connections(g)
        found_fp = copy.copy(self.template_dict)

        for val in node_val_list:
            found_fp[self.v_to_i[val]] = 1

        fp = list(found_fp.values())

        if self.dict_mode:
            fp = {k: v for k, v in zip(self.all_node_val_list, fp)}

        return fp


def analyze_node_connections(g: nx.DiGraph, keywords: List[str] = ["label", "type", "pressure", "temperature", "time", "frequency", "protocol", "machine"]) -> List[str]:
    """
    analyze node vals in the flowchart

    Parameters
    ----------
    g : networkX object
        graph object
    keywords : list of str
        label information list which will be considered to analyze the nodes

    Returns
    -------
    node_val_list : list of str
        list of characteristic node info.

    """

    node_val_list = []

    for node in g.nodes:
        node_val = ""
        for key in g.nodes[node]:
            if key in keywords:
                node_val += key+": "+g.nodes[node][key]+", "
        neighbor_val_list = []

        for neighbor_node in (g.predecessors(node)):
            neighbor_val_list.append(g.nodes[neighbor_node]["label"])

        for neighbor_node in (g.to_undirected().neighbors(node)):
            neighbor_val_list.append(g.nodes[neighbor_node]["label"])

        neighbor_val_list = sorted(neighbor_val_list)
        node_val += "<-->"+"--".join(neighbor_val_list)

        node_val_list.append(node_val)
    return node_val_list
