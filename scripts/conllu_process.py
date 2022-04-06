import os
import re
import shutil

import networkx as nx

from tuw_nlp.grammar.text_to_4lang import UDTo4lang
from tuw_nlp.graph.utils import GraphFormulaMatcher
from networkx.algorithms.isomorphism import DiGraphMatcher

from stanza.utils.conll import CoNLL
from stanza.models.common.doc import Document
import graphviz
import conllu
from nltk.corpus.reader import XMLWordnet

from tuw_nlp.graph.utils import pn_to_graph


def visualize(parsed):
    dot = graphviz.Digraph()
    dot.node("0", "ROOT", shape="box")
    for sentence in parsed.sentences:
        for token in sentence.tokens:
            for word in token.words:
                dot.node(str(word.id), f"{word.lemma}_{word.pos}")
                dot.edge(str(word.head), str(word.id),
                         label=word.deprel)
    return dot


def conll_to_stanza(conll):
    lines = conll.serialize().strip().split('\n')
    tokenized = [line.strip().split("\t") for line in
                 lines if not line.startswith("#")]
    return Document(CoNLL.convert_conll([tokenized]))


def get_ud_and_4lang(tree, path=None):
    ud = conll_to_stanza(tree)
    fourlang_graph = list(UT4(ud))[0]
    if path is not None:
        fourlang = graphviz.Source(fourlang_graph.to_dot(), filename=f"{path}_4lang.gv", format="png")
        fourlang.render()

        gv = visualize(ud)
        ud_graph = graphviz.Source(gv.source, filename=f"{path}.gv", format="png")
        ud_graph.render()
    return ud, fourlang_graph


def node_matcher(node1, node2):
    regex_match = re.match(fr"\b({node2['name']})\b", node1['name'], re.IGNORECASE)
    if regex_match:
        return True
    w1 = wn.synsets(node1['name'])
    w2 = wn.synsets(node2['name'])
    for syns1 in w1:
        for syns2 in w2:
            if syns1.wup_similarity(syns2) > 0.8:
                return True
    return False


def get_concept_weight(start, end, attr):
    edge_names = [attributes["name"] for (_, attributes) in attr.items()]
    if "antonym" in edge_names or "distinct_from" in edge_names:
        return 1e12
    if "is_a" in edge_names or "synonym" in edge_names or "form_of" in edge_names:
        return 1
    return 10


def concept_distance(node1, node2):
    regex_match = re.match(fr"\b({node2['name']})\b", node1['name'], re.IGNORECASE)
    if regex_match:
        return True
    if node1["name"] in UT4.lexicon.concept_graph.nodes() and node2["name"] in UT4.lexicon.concept_graph.nodes():
        try:
            sp = nx.shortest_path(UT4.lexicon.concept_graph, node1['name'].lower(),
                                  node2['name'].lower(), weight=get_concept_weight)
            if len(sp) < 5:
                print(node1, node2)
            return len(sp) < 5
        except nx.exception.NetworkXNoPath:
            return False
    return False


def syns_graph_matcher(fourlang_graph, pattern_graph):
    subgraphs = []
    matcher = DiGraphMatcher(
        fourlang_graph, pattern_graph,
        #node_match=node_matcher,
        node_match=concept_distance,
        edge_match=GraphFormulaMatcher.edge_matcher)
    monomorphic_subgraphs = list(matcher.subgraph_monomorphisms_iter())
    if len(monomorphic_subgraphs) == 0:
        return None
    mapping = monomorphic_subgraphs[0]
    subgraph = fourlang_graph.subgraph(mapping.keys())
    nx.set_node_attributes(subgraph, mapping, name="mapping")
    subgraphs.append(subgraph)
    return subgraphs


def graph_matcher(fourlang_graph, pattern_graph):
    subgraphs = []
    matcher = DiGraphMatcher(
        fourlang_graph, pattern_graph,
        node_match=GraphFormulaMatcher.node_matcher,
        edge_match=GraphFormulaMatcher.edge_matcher)
    monomorphic_subgraphs = list(matcher.subgraph_monomorphisms_iter())
    if len(monomorphic_subgraphs) == 0:
        return None
    mapping = monomorphic_subgraphs[0]
    subgraph = fourlang_graph.subgraph(mapping.keys())
    nx.set_node_attributes(subgraph, mapping, name="mapping")
    subgraphs.append(subgraph)
    return subgraphs


if __name__ == '__main__':
    conll_file = "data/UD_German-HDT/de_hdt-ud-train.conllu"
    wn = XMLWordnet(path=None, out="/home/kinga/Documents/sexism_detection/scripts", convert=False)
    if os.path.exists("cache"):
        shutil.rmtree("cache")
    if os.path.exists("tmp"):
        shutil.rmtree("tmp")
    UT4 = UDTo4lang("de", "cache")
    with open(conll_file) as de:
        trees = conllu.parse(de.read())
    index = 0

    maximal_graph = pn_to_graph("(u_1 / .* :0 (u_2 / maximal))")[0]

    maximal = wn.synset("maximal.a.01")
    matches = []
    matches2 = []
    for i, i_tree in enumerate(trees[index:index+1000]):
        try:
            ud, fourlang = get_ud_and_4lang(i_tree)
            match = graph_matcher(fourlang.G, maximal_graph)
            match2 = syns_graph_matcher(fourlang.G, maximal_graph)
            if match is not None:
                matches.append(i+index)
                print(f"MAXIMAL: {i_tree.metadata['text']}")
            if match2 is not None:
                matches2.append(i+index)
                print(f"CONCEPT: {i_tree.metadata['text']}")
        except TypeError:
            print(i+index)
    print(len(matches))
    print(matches)
    print(len(matches2))
    print(matches2)

