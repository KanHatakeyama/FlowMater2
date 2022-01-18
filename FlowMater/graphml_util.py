"""
******************************
process graphml files
******************************
"""

import networkx as nx
import os
import io
from typing import List, Dict

temp_path = "temp.graphml"


def preprocess_graphml(path: str) -> List[str]:
    """
    preprocess graphml file exported by yEd

    Parameters
    ----------
        path : path to a graphml file

    Returns
    -------
        data_lines : parsed str lines
    """

    # modify graphml
    with open(path, encoding="utf-8") as f:
        data_lines = f.read()

    # replace codes
    data_lines = data_lines.replace("GenericNode", "ShapeNode")
    data_lines = data_lines.replace("<y:GroupNode>", "")
    data_lines = data_lines.replace("</y:GroupNode>", "")
    data_lines = data_lines.replace("ProxyAutoBoundsNode", "ShapeNode")
    data_lines = data_lines.replace("""<y:Realizers active="0">""", "")
    data_lines = data_lines.replace("</y:Realizers>", "")

    return data_lines


def load_graphml(path: str) -> nx.DiGraph:
    """
    load graphml file exported by yEd

    Parameters
    ----------
        path : path of graphml file

    Returns
    -------
        g : loaded graph object

    """

    data_lines = preprocess_graphml(path)
    g = nx.read_graphml(io.StringIO(data_lines))
    return g
