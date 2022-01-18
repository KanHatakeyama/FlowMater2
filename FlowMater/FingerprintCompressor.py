from transformers import BertModel, BertConfig, BertTokenizer
import torch
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import joblib
import os
from typing import List, Dict, Union


class FingerprintCompressor:
    """
    Compress fingerprint data by BERT and kNN
    """

    def __init__(self,
                 all_fp_keys: List[str],
                 bert_model_name: str = 'bert-base-uncased',
                 n_clusters: int = 30,
                 max_words: int = 4,
                 dict_mode: bool = False,
                 init_data: bool = False,
                 data_dict_path: str = "temp/bert_words.bin"
                 ):
        """
        Parameters
        ----------
        all_fp_keys : list of fingerprint keys (FlowchartFP.v_to_i.keys())
        bert_model_name: bert model name
        n_clusters: number of clusters for knn
        max_words: number of words to be extracted after compression
        dict_mode: return dict type fingerprint if true
        init_data: if true, initialize bert etc
        data_dict_path: path to the vector data made by bert

        """

        self.n_clusters = n_clusters
        self.max_words = max_words
        self.dict_mode = dict_mode
        self.init_data = init_data
        self.data_dict_path = data_dict_path
        self.bert_model_name = bert_model_name
        self.model = None

        if not os.path.exists(data_dict_path) or self.init_data == True:
            self.word_dict = {}
        else:
            self.word_dict = joblib.load(data_dict_path)

        self._prepare_conversion_dict(all_fp_keys)

    def encode_text(self, text: str) -> np.array:
        """
        encode text data by bert

        Parameters
        ----------
            text : text

        Returns
        -------
            embed : its vector
        """

        if text not in self.word_dict:
            # init bert
            if self.model is None:
                self.tokenizer = BertTokenizer.from_pretrained(
                    self.bert_model_name)
                self.model = BertModel.from_pretrained(self.bert_model_name)
                _ = self.model.eval()

            # calc vector
            with torch.no_grad():
                outputs = self.model(
                    self.tokenizer.encode(text, return_tensors="pt"))
                embed = np.array(outputs[1][0])
                self.word_dict[text] = embed
                joblib.dump(self.word_dict, self.data_dict_path)
        else:
            embed = self.word_dict[text]

        return embed

    def _prepare_conversion_dict(self, all_fp_keys: List[str]) -> None:
        """
        compress fingerprint keys by knn

        Parameters
        ----------
            all_fp_keys : original FP keys

        """

        # calc embeddinfs
        fp_embeddings = [self.encode_text(i) for i in all_fp_keys]

        self.fp_embedding_dict = {k: v for k,
                                  v in zip(all_fp_keys, fp_embeddings)}
        
        if max(self.n_clusters,len(fp_embeddings))<=30:
            self.n_clusters=len(fp_embeddings)
            pred=list(range(self.n_clusters))
        else:
            pred = KMeans(n_clusters=self.n_clusters).fit_predict(fp_embeddings)

        self.fp_class_df = pd.DataFrame([all_fp_keys, pred]).T
        self.fp_class_df.columns = ["Step", "Class"]
        self.fp_class_df = self.fp_class_df.sort_values(by="Class")
        self.fp_class_df = self.fp_class_df.reset_index()

        self.conversion_dict = prepare_rename_dict(self.fp_class_df)
        self.compressed_fp_keys = list(set(self.conversion_dict.values()))

    def __call__(self, original_fp: dict) -> Union[List[int], Dict[str, int]]:
        """
        automatically convert text data into vector

        Parameters
        ----------
            original_fp : original fingerprint data

        Returns
        -------
            compressed_fp : compressed fingerprint data
        """

        compressed_fp = {k: 0 for k in self.compressed_fp_keys}
        for k in original_fp:
            if original_fp[k] == 1:
                compressed_fp[self.conversion_dict[k]] = 1

        if not self.dict_mode:
            compressed_fp = list(compressed_fp.values())
        return compressed_fp


# utilities
def clean_text(text: str) -> str:
    """
    clean text data

    Parameters
    ----------
        text : original text

    Returns
    -------
        modif_text : modified text
    """

    modif_text = ""
    # delete space between number and unit
    for word in text.split(" "):
        modif_text += word
        try:
            float(word)
        except:
            modif_text += " "

    clean_dict = {
        "label": ",",
        ":": "",
        ",": "",
        "--": " ",
        "type": "",
        "procedure": "",
        "protocol": "",
        "time": "",
    }

    for k, v in clean_dict.items():
        modif_text = modif_text.replace(k, v)
    return modif_text


def make_new_label_name(target_texts: List[str], max_words: int = 4) -> str:
    """
    Prepare a new fingerprint name from original FP keys. Just extract freqently appearing words.

    Parameters
    ----------
        target_texts : original FP keys
        max_words: max number of words to generate the  name

    Returns
    -------
        new_label : generated key
    """

    bug_of_words = []
    bug_of_words.append({})
    bug_of_words.append({})

    # collect original text data
    for original_text in target_texts:
        # split key data
        # one step consists of "main operation" and "sub operation" (i.e., connected nodes)
        main_text, sub_text = original_text.split("<-->")
        for i, text in enumerate([main_text, sub_text]):
            c_text = clean_text(text)
            for word in c_text.split():
                bug_of_words[i][word] = bug_of_words[i].get(word, 0) + 1

    # make new label name
    new_label = ""
    for i in range(len(bug_of_words)):
        # sort by frequencies
        sorted_words = sorted(
            bug_of_words[i].items(), key=lambda x: x[1], reverse=True)

        for num, k in enumerate(bug_of_words[i]):
            new_label += sorted_words[num][0]+"-"
            if num == max_words:
                new_label = new_label[:-1]
                break
        new_label += "|"
    new_label = new_label[:-1]

    return new_label


def prepare_rename_dict(fp_class_df: pd.DataFrame) -> dict:
    """
    Prepare renaming dict of original FP keys

    Parameters
    ----------
        fp_class_df : see _prepare_conversion_dict function in FingerprintCompressor class

    Returns
    -------
        fp_rename_dict : rename dict
    """

    fp_rename_dict = {}
    # count frequency of words
    for target_class in set(fp_class_df["Class"]):
        target_texts = fp_class_df[fp_class_df["Class"]
                                   == target_class]["Step"].values
        new_label = make_new_label_name(target_texts)

        for k in target_texts:
            fp_rename_dict[k] = new_label

    return fp_rename_dict
