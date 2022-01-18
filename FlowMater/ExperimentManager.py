import glob
from graphml_util import load_graphml
from graph_util import parse_graph, auto_integrate_graph, parse_json_graph, delete_note_nodes, parse_lot_keywords, extract_user_selected_node_features, find_key_in_graph
from FlowchartFP import FlowchartFP
from FingerprintCompressor import FingerprintCompressor
import pandas as pd
import os
import joblib


class ExperimentManager:
    """
    Auutomatically load experimental data
    """

    def __init__(self, fp_numbers: int = 30,
                 base_path: str = "database/*/**",
                 experiment_bin_path: str = "temp/experiment.bin",
                 experiment_timestamp_path: str = "temp/timestamp.bin",
                 delete_notes: bool = True,
                 json_path: str = "database/lot_keywords.json"):
        """
        Parameters
        ----------
        fp_numbers : number of fingerprint types to be generawted
        base_path: path to experimental data
        experiment_bin_path: saving path to parsed data
        experiment_timestamp_path: path to timestamp data
        delete_notes: delete note information if true
        json_path: path to keyword file
        """

        self.fp_numbers = fp_numbers
        self.base_path = base_path
        self.experiment_bin_path = experiment_bin_path
        self.experiment_timestamp_path = experiment_timestamp_path
        self.delete_notes = delete_notes
        self.json_path = json_path

        if not os.path.exists("temp"):
            os.mkdir("temp")
    def auto_load(self) -> None:
        """
        Automatically load and parse graphml files recorded in the database folder 
        """

        # check timestamp for all files
        timestamp_dict = {}
        for path in glob.glob(self.base_path, recursive=True):
            timestamp_dict[path] = os.stat(path).st_mtime

        if os.path.exists(self.experiment_timestamp_path):
            old_experiment_timestamp = joblib.load(
                self.experiment_timestamp_path)
        else:
            old_experiment_timestamp = {}

        # if no change with database, load previous data
        if timestamp_dict == old_experiment_timestamp:
            print("load already parsed database")
            (self.experiment_dict, self.dataframe, self.selected_graph_dict,
             self.fp_calculator, self.fp_compressor) = joblib.load(self.experiment_bin_path)

        # parse data
        else:
            self.parse_graphml()
            self.process_graphs()

            joblib.dump((self.experiment_dict, self.dataframe, self.selected_graph_dict,
                         self.fp_calculator, self.fp_compressor), self.experiment_bin_path)
            joblib.dump(timestamp_dict, self.experiment_timestamp_path)

    def parse_graphml(self) -> None:
        """
        load all graphml files in the database folder

        Parameters
        ----------
        path : str
            path to graphml files
        """

        print("begin parsing graphml files")
        experiment_dict = {}
        count = 0
        for path in glob.glob(self.base_path, recursive=True):
            if path.find(".graphml") > 0:
                g = load_graphml(path)
                graph_dict = parse_graph(g)
                graph_dict["path"] = path
                graph_name = graph_dict["name"]

                if graph_name == "normal_experiment":
                    graph_name = graph_name+str(count)
                    count += 1

                experiment_dict[graph_name] = graph_dict

        self.experiment_dict = experiment_dict

    def process_graphs(self):
        """
        parse loaded graph objects

        Parameters
        ----------
        delete_notes : bool
            if true, delete "note" information
        json_path : str
            path to the json file of "keyword" data
        """

        experiment_dict = self.experiment_dict
        delete_notes = self.delete_notes
        json_path = self.json_path

        print("Loading graphs")

        # integrate subgraphs and parse json data
        for key in experiment_dict.keys():
            if key.find("normal_experiment") < 0:
                continue

            graph_dict = experiment_dict[key]
            # integrate subgraphs
            graph_dict["graph_integrated"] = auto_integrate_graph(
                graph_dict["graph"], experiment_dict)
            # parse json data
            try:
                parse_json_graph(graph_dict)
            except:
                path = graph_dict["path"]
                raise ValueError(f"error parsing {key}, {path}")

        # load json data of experimental results
        selected_graph_dict = {}
        for experiment_id in experiment_dict.keys():
            if experiment_id.find("normal_experiment") >= 0:
                # target_experiment_ids.append(experiment_id)

                dict_keys = list(experiment_dict[experiment_id].keys())
                if "graph_integrated_json_0" in dict_keys:
                    for key in dict_keys:
                        if key.find("graph_integrated_json_") >= 0:
                            path = experiment_dict[experiment_id]["path"]
                            selected_graph_dict[f"{experiment_id}_{key}_{path}"] = experiment_dict[experiment_id][key]
                else:
                    path = experiment_dict[experiment_id]["path"]
                    selected_graph_dict[f"{experiment_id}_{path}"] = experiment_dict[experiment_id]["graph_integrated"]

        # delete "Note nodes"
        if delete_notes:
            for key in selected_graph_dict:
                delete_note_nodes(selected_graph_dict[key])

        # load lot data in "lot_keywoeds.json"
        parse_lot_keywords(selected_graph_dict, json_path=json_path)

        self.set_path_to_experiment_data()

        # init fingerprint module
        self.fp_calculator = FlowchartFP(
            list(selected_graph_dict.values()), dict_mode=True)
        fp_label_list = self.fp_calculator.all_node_val_list

        if self.fp_numbers >= 0:
            print("Initiating FP compressor")
            self.fp_compressor = FingerprintCompressor(
                fp_label_list, n_clusters=self.fp_numbers)
            self.fp_compressor.dict_mode = True
        else:
             self.fp_compressor=None
        print("Calculating features")
        feature_dict = {}
        for k, g in selected_graph_dict.items():
            # extract use-selected features from graphs
            feature_dict[k] = extract_user_selected_node_features(g)

            # calculate fingerprints
            fp = self.fp_calculator(g)
            for label, val in fp.items():
                feature_dict[k][label] = val

            if self.fp_numbers >= 0:
                # calculate compressed fingerprints
                compressed_fp = self.fp_compressor(fp)
                for label, val in compressed_fp.items():
                    feature_dict[k]["FP: "+label] = val

        self.dataframe = pd.DataFrame.from_dict(feature_dict).T
        self.selected_graph_dict = selected_graph_dict

    def set_path_to_experiment_data(self) -> None:
        """
        set path to the experiment data noted as "load_data" keys 
        """
        load_data_key = "load_data"

        # check each experiment
        for key in self.experiment_dict:
            experiment = self.experiment_dict[key]

            # if "graph_integrated" in experiment.keys():
            for graph_key in list(experiment.keys()):
                if graph_key.find("graph_integrated") < 0:
                    continue

                g = experiment[graph_key]
                base_path = os.path.dirname(experiment['path'])+"/"
                node_id = find_key_in_graph(g, load_data_key)

                if node_id:
                    load_path = base_path + g.nodes[node_id][load_data_key]
                    #df = pd.read_csv(base_path+load_path)
                    # print(load_path)

                    # set full path
                    if os.path.exists(load_path):
                        # with open(load_path) as f:
                        #    s = f.read()
                        #experiment[graph_key+"_data_"+g.nodes[node_id]["label"]] = s
                        g.nodes[node_id][load_data_key] = load_path
                    else:
                        g.nodes[node_id][load_data_key] = "path not found"
