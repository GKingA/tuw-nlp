import pandas as pd
import networkx as nx
import json
import os
from tqdm import tqdm
from argparse import ArgumentParser, ArgumentError


def read_wikt2dict(path):
    return pd.read_csv(
        path, sep="\t", header=None, names=["lang1", "word1", "lang2", "word2", "cnt"]
    )


def graph_from_conceptnet(path, lang):
    from conceptnet_lite import connect, Language

    base_path = os.path.expanduser("~/tuw_nlp_resources")
    assert os.path.exists(
        os.path.join(base_path, "conceptnet.db")
    ), "Please download conceptnet using tuw_nlp.download_conceptnet()"
    graph = nx.Graph()

    connect(os.path.join(base_path, "conceptnet.db"))
    current_language = Language.get(name=lang)
    for label in tqdm(current_language.labels):
        label_name = " ".join(label.text.split("_"))
        for c in label.concepts:
            if c.edges_out:
                for e in c.edges_out:
                    if (
                        e.relation.name in ["synonym", "related_to"]
                        and e.end.language != current_language
                    ):
                        other = " ".join(e.end.label.text.split("_"))
                        if e.end.language is not None:
                            graph.add_edge(
                                (label_name, label.language.name),
                                (other, e.end.language.name),
                                cnt=e.etc["weight"],
                            )
    dictionary = nx.node_link_data(graph)
    with open(path, "w") as dict_file:
        json.dump(dictionary, dict_file, indent=4)
    return graph


def read_fourlang(path):
    return pd.read_csv(path, sep="\t")


def graph_from_df(df, path):
    graph = nx.Graph()
    for index, row in df.iterrows():
        n1 = (row.word1, row.lang1)
        n2 = (row.word2, row.lang2)
        graph.add_edge(n1, n2, cnt=row.cnt)
    dictionary = nx.node_link_data(graph)
    with open(path, "w") as dict_file:
        json.dump(dictionary, dict_file, indent=4)
    return graph


def graph_from_json(path):
    with open(path) as graph_path:
        dictionary_json = json.load(graph_path)
        graph_dict = nx.node_link_graph(dictionary_json)
    return graph_dict


def get_edges_to_lang(graph, word, language):
    edges = graph.edges(word, data=True)
    return {e[1][0]: e[2]["cnt"] for e in edges if e[1][1] == language}


def add_dicts(list_of_dicts):
    base_dict = list_of_dicts[0].copy()
    for dictionary in list_of_dicts[1:]:
        for element, cnt in dictionary.items():
            if element in base_dict:
                base_dict[element] *= 2
                base_dict[element] += 2 * cnt
            else:
                base_dict[element] = cnt
    return base_dict


def denumber_accent(string):
    string_acc = str(string)
    accents = {
        "a1": "á",
        "a2": "ä",
        "e1": "é",
        "i1": "í",
        "o1": "ó",
        "o2": "ö",
        "o3": "ő",
        "o12": "ő",
        "u1": "ú",
        "u2": "ü",
        "u3": "ű",
        "u12": "ű",
    }
    for numeric, acc in accents.items():
        string_acc.replace(numeric, acc).replace(numeric.upper(), acc.upper())
    return string_acc


def find_translation(df, graph, target_language):
    word_tuple = lambda x, y: (denumber_accent(x[y]).strip("/"), y)
    languages = [k for k in df.keys() if k not in ["s", "l", "num", "def", "%comment"]]
    de_nodes = []
    best_de = []
    for index, row in df.iterrows():
        edges = []
        for lang in languages:
            edges.append(
                get_edges_to_lang(graph, word_tuple(row, lang), target_language)
            )
        edges = add_dicts(edges)
        de_nodes.append(edges)
        if len(edges) > 0:
            best_match = max(edges.items(), key=lambda x: x[1])
        else:
            best_match = None, None
        best_de.append(best_match[0])
    df.insert(len(languages), target_language, best_de)
    return df


if __name__ == "__main__":
    argparse = ArgumentParser()
    argparse.add_argument(
        "--translate_file", "-tf", help="The output of the wikt2dict process"
    )
    argparse.add_argument(
        "--dictionary_json",
        "-d",
        help="Where to store or find the dictionary in a graph format",
        required=True,
    )
    argparse.add_argument(
        "--use_conceptnet",
        "-c",
        help="Whether to use conceptnet connections for the translation",
        action="store_true",
    )
    argparse.add_argument(
        "--fourlang", "-f", help="Path to the 4lang dictionary", required=True
    )
    argparse.add_argument(
        "--target_language",
        "-l",
        help="The language you want to translate to",
        default="de",
    )
    argparse.add_argument(
        "--output_file",
        "-o",
        help="Where to save the extended 4lang dictionary",
        default="fourlang_out.tsv",
    )
    args = argparse.parse_args()

    dict_graph = None
    if not os.path.exists(args.dictionary_json):
        if args.use_conceptnet:
            dict_graph = graph_from_conceptnet(
                args.dictionary_json, args.target_language
            )
        else:
            if args.translate_file is None:
                raise ArgumentError(
                    args.translate_file,
                    "If the dictionary is to be generated, you "
                    "have to use specify the translate file, or use conceptnet",
                )
            dict_graph = graph_from_df(
                read_wikt2dict(args.translate_file), args.dictionary_json
            )

    if dict_graph is None:
        dict_graph = graph_from_json(args.dictionary_json)

    fourlang_df = read_fourlang(args.fourlang)

    updated_df = find_translation(fourlang_df, dict_graph, args.target_language)

    updated_df.to_csv(args.output_file, sep="\t", index=None)
