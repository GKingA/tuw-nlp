import os
import re
import shutil
import functools

import networkx as nx

from tuw_nlp.grammar.text_to_4lang import UDTo4lang
from tuw_nlp.graph.utils import GraphFormulaMatcher
from tuw_nlp.graph.fourlang import FourLang
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


def visualize_concept_path(sub_nodes):
    vis_graph = graphviz.Digraph()
    subgraph_3 = UT4.lexicon.concept_graph.subgraph(sub_nodes)
    for edge in subgraph_3.edges(data=True):
        for e in edge[2].items():
            if e[1] != "external_url":
                vis_graph.edge(edge[0], edge[1], e[1])
    g = graphviz.Source(vis_graph.source, format="png")
    g.view()


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
    positive_names = {"is_a", "synonym", "form_of", "defined_as"}
    negative_names = {"antonym", "distinct_from"}
    if len(negative_names.intersection(edge_names)) > 0:
        return 1e64
    if len(positive_names.intersection(edge_names)) > 0:
        return 1
    return 10


@functools.lru_cache(maxsize=1024)
def hashable_concept_distance(node1, node2):
    try:
        sp = nx.shortest_path_length(UT4.lexicon.concept_graph, node1, node2, weight=get_concept_weight)
        # visualize_concept_path(sp)
        return sp
    except nx.exception.NetworkXNoPath:
        return None


def concept_distance(node1, node2):
    regex_match = re.match(fr"\b({node2['name']})\b", node1['name'], re.IGNORECASE)
    if regex_match:
        return True
    if node1["name"] in UT4.lexicon.concept_graph.nodes() and node2["name"] in UT4.lexicon.concept_graph.nodes():
        sp = hashable_concept_distance(node1["name"].lower(), node2["name"].lower())
        if sp is not None:
            return sp <= 11
    return False


def select_nodes_for_expand(fourlang_graph, pattern_graph):
    subgraphs = []
    matcher = DiGraphMatcher(
        fourlang_graph, pattern_graph,
        node_match=concept_distance,
        edge_match=None)
    monomorphic_subgraphs = list(matcher.subgraph_monomorphisms_iter())
    if len(monomorphic_subgraphs) == 0:
        return None
    for mapping in monomorphic_subgraphs:
        subgraph = fourlang_graph.subgraph(mapping.keys())
        sub = FourLang(fourlang_graph)
        UT4.expand(sub, depth=2, expand_set=set([n[1]["name"] for n in subgraph.nodes()._nodes.items()]),
                   use_concept_def=True)
        #graphviz.Source(sub.to_dot(), format="png").view()
        nx.set_node_attributes(subgraph, mapping, name="mapping")
        subgraphs.append(subgraph)
    return subgraphs


def syns_graph_matcher(fourlang_graph, pattern_graph):
    subgraphs = []
    matcher = DiGraphMatcher(
        fourlang_graph, pattern_graph,
        node_match=node_matcher,
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
    
    vis_graph = graphviz.Digraph()
    path = nx.shortest_path(UT4.lexicon.concept_graph, target="maximal", weight=get_concept_weight)
    path_l = nx.shortest_path_length(UT4.lexicon.concept_graph, target="maximal", weight=get_concept_weight)
    path_length_5 = set([node for p in path.items() if path_l[p[0]] <= 11 for node in p[1]])
    visualize_concept_path(path_length_5)
    try:
        target_bis = nx.shortest_path(UT4.lexicon.concept_graph, source="maximal", target="unterschreiten", weight=get_concept_weight)
    except:
        target_bis = []
    try:
        source_bis = nx.shortest_path(UT4.lexicon.concept_graph, target="maximal", source="unterschreiten", weight=get_concept_weight)
    except:
        source_bis = []
    nodes = []
    for node in target_bis + source_bis:
        if node not in nodes:
            nodes.append(node)
    visualize_concept_path(nodes)

    # maximal = wn.synset("maximal.a.01")
    matches = []
    matches2 = []
    matches3 = []
    matches4 = []
    words_of_interest = ["maximal", "maximum", "überschreit", "unterschreit", " bis ", "höchstens"]
    for i, i_tree in enumerate(trees[index:index + 1000]):
        text = i_tree.metadata['text'].lower()
        words_in = [w for w in words_of_interest if w in text]
        if len(words_in) > 0:
            matches4.append(i + index)
            print(f"TEXT: {i_tree.metadata['text']}\n{words_in}")
        try:
            ud, fourlang = get_ud_and_4lang(i_tree)
            match = graph_matcher(fourlang.G, maximal_graph)
            match2 = syns_graph_matcher(fourlang.G, maximal_graph)
            match3 = select_nodes_for_expand(fourlang.G, maximal_graph)
            if match is not None:
                matches.append(i + index)
                print(f"MAXIMAL: {i_tree.metadata['text']}")
            if match2 is not None:
                matches2.append(i + index)
                print(f"SYNSET: {i_tree.metadata['text']}")
            if match3 is not None:
                matches3.append(i + index)
                print(f"CONCEPT: {i_tree.metadata['text']}\n{[m3.nodes(data=True) for m3 in match3]}")
        except TypeError:
            print(i + index)
    print("String match in graph (just maximal):")
    print(len(matches))
    print(matches)
    print("OdeNet match:")
    print(len(matches2))
    print(matches2)
    print("Concept match:")
    print(len(matches3))
    print(matches3)
    print("String match in text (maximal, maximum, bis, überschreiten, unterschreiten, höchstens):")
    print(len(matches4))
    print(matches4)
