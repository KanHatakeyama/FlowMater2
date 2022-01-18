"""
******************************
process graph files
******************************
"""

import copy
import json
import os
import networkx as nx
import matplotlib.pyplot as plt
from typing import List, Dict


# --------------printing utilities------------------------


def draw_graph(g: nx.DiGraph, sx: int = 30, sy: int = 10, print_info: bool = True) -> None:
    """
    draw graph object

    Parameters
    ----------
    g : networkX object
        graph object of experiment
    sx : int
        size X
    sy : int
        size Y
    print_info : bool
        print graph node information or not
    """

    plt.figure(1, figsize=(sx, sy))

    #pos = nx.nx_pydot.pydot_layout(g, prog='dot')
    pos = nx.spring_layout(g)
    
    nx.draw_networkx_labels(g, pos, labels={k: (
        f"{k}: {v}") for k, v in nx.get_node_attributes(g, 'label').items()})
    nx.draw(g, pos, node_color="orange", node_size=50)
    plt.show()

    if print_info:
        print_graph_info(g)


def print_node_info(node: dict, node_id: int) -> None:
    """
    print node information

    Parameters
    ----------
    node : dict 
        node data for a networkX graph (nx.DiGraph.nodes[int])
    node_id : int
        node ID to be printed
    """
    count = 0
    for k, v in node.items():
        if k != "x" and k != "y":
            if count == 0:
                print(node_id, ": ", v)
            else:
                print("   ", k, ": ", v.replace("\n", " "))
            count += 1


def print_graph_info(g: nx.DiGraph) -> None:
    """
    print graph informatoin

    Parameters
    ----------
    g : networkX object
        graph object of experiment
    """

    for node_id in list(g.nodes):
        node = g.nodes[node_id]
        print_node_info(node, node_id)


# ---------- data load utilities-------------------

def parse_note(node: dict, note_info: str) -> None:
    """
    parse note information (* this function is under developement)

    Parameters
    ----------
    node : dict 
        node data of networkX object
    note_info : str
        text to be noted on the node
    """

    node["note"] = note_info


def parse_general_nodes(node: dict, lines: List[str]) -> None:
    """
    parse node information

    Parameters
    ----------
    node : dict 
        node data of networkX object
    lines : list of string
        text written on the original graphml data
    """

    if len(lines) == 1:
        return

    for line in lines[1:]:
        if line == "":
            continue

        try:
            k, v = line.split(": ")
            node[k] = v
        except:
            print(f"error parsing {line}")


def parse_graph(g: nx.DiGraph) -> dict:
    """
    parse graph object, loaded from graphml file

    Parameters
    ----------
    g : networkX object
        graph object of experiment

    Returns
    -------
    graph_dict : dict
        dict of graph data
            "type": str (type of experiment)
            "name": str (name of experiment)
            "graph": networkX object (parsed graph data)

    """

    graph_dict = {}
    graph_dict["type"] = "normal_experiment"
    graph_dict["name"] = "normal_experiment"

    # process each node
    for node_id in list(g.nodes):
        node = g.nodes[node_id]
        label = node["label"]

        # delete "None" node (e.g., photograph nodes)
        if label is None:
            g.remove_node(node_id)
            continue

        # parse node label information
        lines = label.split("\n")

        # get & set node label
        node_label = lines[0]
        node["label"] = node_label

        # in case of "Note"
        if node_label == "Note":
            parse_note(node, label)
        else:
            parse_general_nodes(node, lines)

        # specify graph types
        if node_label == "Save":
            graph_dict["type"] = "partial_experiment"
            graph_dict["name"] = node["name"]

    graph_dict["graph"] = g

    return graph_dict


# ------------- other utilities --------------------


def extract_user_selected_node_features(g: nx.DiGraph) -> dict:
    """
    search for "ID" labels in nodes and collect the corresponding data

    Parameters
    ----------
    g : networkX object
        graph object

    Returns
    out_property_dict: dict
        extracted data
    -------

    """

    special_property_dict = {}

    # obtain dict data of user-interested nodes
    for node_id in g.nodes:
        for label in g.nodes[node_id].keys():
            if label == "ID":
                special_property_dict[g.nodes[node_id]
                                      [label]] = copy.deepcopy(g.nodes[node_id])

    out_property_dict = {}

    # clean dict
    for feature_id in special_property_dict.keys():
        for key in list(special_property_dict[feature_id].keys()):
            if key not in ["x", "y", "ID", "type", "label", "note"]:
                # special_property_dict[feature_id].pop(key)
                out_property_dict[f"{feature_id}_{key}"] = special_property_dict[feature_id][key]

    return out_property_dict


def delete_note_nodes(g: nx.DiGraph) -> None:
    """
    delete "Note" nodes

    Parameters
    ----------
    g : networkX object
        graph object

    """
    for node_id in list(g.nodes):
        if g.nodes[node_id]["label"] == "Note":
            g.remove_node(node_id)


def rename_label(g: nx.DiGraph, old_label: str = "Start experiment", new_label: str = "Start experiment_") -> None:
    """
    rename node label

    Parameters
    ----------
    g : networkX object
        graph object
    old_label : str
        old label name
    new_label : str
        new label name

    """
    for node_id in list(g.nodes):
        node = g.nodes[node_id]
        if node["label"] == old_label:
            node["label"] = new_label


def find_key_and_value_in_graph(g: nx.DiGraph, key: str, value: str) -> int:
    """
    find target node ID containing specific value for node[key]

    Parameters
    ----------
    g : networkX object
        graph object
    key : str
        node key to be searched
    value : str
        value to be searched

    Returns
    -------
    node_id : int
        found node ID
    """
    for node_id in list(g.nodes):
        node = g.nodes[node_id]
        if key in node.keys():
            if node[key] == value:
                return node_id


def find_value_in_graph(g: nx.DiGraph, value: str) -> (str, int):
    """
    find target node ID containing specific value in any keys

    Parameters
    ----------
    g : networkX object
        graph object
    value : str
        value to be searched

    Returns
    -------
    key : str
        key in the found node
    node_id : int
        found node ID
    """
    for node_id in list(g.nodes):
        node = g.nodes[node_id]
        for key in node.keys():
            if node[key].find(value) >= 0:
                return key, node_id


def find_key_in_graph(g: nx.DiGraph, key: str) -> int:
    """
    find target node ID containing specific key

    Parameters
    ----------
    g : networkX object
        graph object
    key : str
        node key to be searched

    Returns
    -------
    node_id : int
        found node ID
    """
    for node_id in list(g.nodes):
        node = g.nodes[node_id]
        if key in node.keys():
            return node_id


def integrate_graphs(g: nx.DiGraph, load_name: str, experiment_dict: dict) -> nx.DiGraph:
    """
    integrate graphs

    Parameters
    ----------
    g : networkX object
        base graph object
    load_name : str
        name of experiment to be integrated with g
    experiment_dict : dict
        dict of experiment data

    Returns
    -------
    integrated_graph : networkX object
        integrated graph object
    """

    # load subgraph
    sub_graph_dict = experiment_dict[load_name]
    sub_graph = copy.deepcopy(sub_graph_dict["graph"])

    # mark start and end nodes
    rename_label(sub_graph, "Start experiment", "Start experiment_")
    rename_label(sub_graph, "Save", "Save_")

    # integrate main and sub graphs
    integrated_graph = copy.deepcopy(g)
    integrated_graph = nx.disjoint_union(sub_graph, integrated_graph)

    # node ID, which contain "load" function
    original_node_id = find_key_and_value_in_graph(
        integrated_graph, "load", load_name)

    # fragment graph, start and end node ids

    ##########
    # treat subgraphs

    # label nodes
    sub_start_node_id_ = find_key_and_value_in_graph(
        integrated_graph, "label", "Start experiment_")
    sub_end_node_id_ = find_key_and_value_in_graph(
        integrated_graph, "label", "Save_")

    # actual nodes to be connected
    sub_start_node_id = list(integrated_graph.neighbors(sub_start_node_id_))[0]
    sub_end_node_id = list(integrated_graph.predecessors(sub_end_node_id_))[0]

    # delete capping nodes in sub graph
    integrated_graph.remove_node(sub_start_node_id_)
    integrated_graph.remove_node(sub_end_node_id_)

    ##########
    # treat main graphs
    main_end_node_id = list(integrated_graph.neighbors(original_node_id))[0]
    main_start_node_id = list(
        integrated_graph.predecessors(original_node_id))[0]

    # maintain node information of the original node
    inherit_dict = {}

    for k in integrated_graph.nodes[original_node_id]:
        if k not in ["x", "y", "label", "type", "load"]:
            inherit_dict[k] = integrated_graph.nodes[original_node_id][k]

    # delete original node
    integrated_graph.remove_node(original_node_id)

    # connect
    integrated_graph.add_edge(main_start_node_id, sub_start_node_id)
    integrated_graph.add_edge(sub_end_node_id, main_end_node_id)

    # restore info of original node
    for k in inherit_dict.keys():
        integrated_graph.nodes[sub_end_node_id][k] = inherit_dict[k]

    return integrated_graph


def auto_integrate_graph(g: nx.DiGraph, experiment_dict: dict) -> nx.DiGraph:
    """
    automatically integrate graph (g)
    (if a graph has a "load" node, it loads the corresponding graph)

    Parameters
    ----------
    g : networkX object
        original graph object

    Returns
    -------
    integ_g : networkX object
        integrated graph object
    """
    integ_g = copy.deepcopy(g)
    # search for "load" function in the nodes and load subgraphs
    while True:
        load_flag = False
        for node_id in list(integ_g.nodes):
            node = integ_g.nodes[node_id]
            if "load" in node.keys():
                load_name = node["load"]
                # print(load_name)

                integ_g = integrate_graphs(integ_g, load_name, experiment_dict)
                load_flag = True
                break

        if load_flag == False:
            break

    return integ_g


def parse_json_graph(graph_dict: dict) -> None:
    """
    load json data in the corresponding graphs
    (original graphml files can have {variable} data. The corresponding "variable" can be loaded from the JSON file having the same name as the graphml file)

    Parameters
    ----------
    graph_dict : dict
        dict of graph data

    """
    # load custom experimental results
    json_path = graph_dict["path"].replace(".graphml", ".json")

    # search for json file
    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            s = f.read()

        try:
            custom_condition_list = json.loads(s)
        except:
            raise ValueError(f"error parsing {json_path}")

        # laod json data and make new graphs with custom experimental data
        for num, condition in enumerate(custom_condition_list):

            full_graph = copy.deepcopy(graph_dict["graph_integrated"])

            # load conditions
            for k, v in condition.items():
                # search for node ids for target custom condition nodes

                search_value = "{"+k+"}"
                key, target_node_id = find_value_in_graph(
                    full_graph, search_value)

                # print(k,v,key,target_node_id)

                # input value in the graph
                if target_node_id:
                    original_value = full_graph.nodes[target_node_id][key]
                    full_graph.nodes[target_node_id][key] = original_value.replace(
                        search_value, str(v))
                else:
                    print(f"caution! could not find {k} described in JSON")

            # save as new graph
            graph_dict[f"graph_integrated_json_{num}"] = full_graph


def parse_lot_keywords(graph_dict: dict, json_path: str = "database/lot_keywords.json") -> None:
    """
    load "lot" keywords in the graphs
    (If a node has a "keyword", the detailed information is loaded from the json data)

    Parameters
    ----------
    graph_dict : dict
        dict of graph data
    json_path : str
        path to the JSON file
    """

    with open(json_path, "r") as f:
        s = f.read()

    try:
        keyword_list = json.loads(s)
    except:
        raise ValueError(f"error parsing {json_path}")

    for graph_id in graph_dict.keys():
        g = graph_dict[graph_id]
        keyword_list = json.loads(s)

        for condition in keyword_list:
            # find node id
            keyword = condition["keyword"]
            res = find_value_in_graph(g, keyword)

            # add keywords to the node
            if res:
                _, node_id = res
                for key in condition:
                    g.nodes[node_id][key] = str(condition[key])
