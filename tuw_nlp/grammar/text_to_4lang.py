import argparse
import logging
import sys
import traceback

import stanza
from tqdm import tqdm

from tuw_nlp.grammar.ud_fl import UD_FL
from tuw_nlp.graph.fourlang import FourLang
from tuw_nlp.graph.lexical import LexGraphs
from tuw_nlp.graph.utils import graph_to_pn, pn_to_graph
from tuw_nlp.text.dictionary import Dictionary
from tuw_nlp.text.pipeline import CachedStanzaPipeline, CustomStanzaPipeline
from tuw_nlp.text.preprocessor import Preprocessor


class To4lang:
    def __init__(self, lang, nlp_cache, cache_dir=None):
        if lang == 'de':
            nlp = CustomStanzaPipeline(
                processors='tokenize,mwt,pos,lemma,depparse')
        elif lang == 'en':
            nlp = stanza.Pipeline(
                'en', processors='tokenize,mwt,pos,lemma,depparse')
        elif lang == 'en_bio':
            nlp = stanza.Pipeline(
                'en', package="craft")
        assert lang, "TextTo4lang does not have lang set"

        self.lang = lang

        self.nlp = CachedStanzaPipeline(nlp, nlp_cache)

        self.ud_fl = UD_FL(cache_dir=cache_dir, lang=lang)

        self.lexicon = Dictionary(lang)

        self.graph_lexical = LexGraphs()

    def add_definition(self, graph, node, definition, substitute, strategy):
        sen = self.nlp(definition).sentences[0]
        def_graph, root = self.parse(sen)
        fourlang_graph = FourLang(def_graph, root, self.graph_lexical)
        if len(def_graph.nodes()) > 0:
            if strategy == "None":
                graph.merge_definition_graph(
                    fourlang_graph, node, substitute)
            elif strategy == "whitelisting":
                fourlang_graph.whitelisting()
                graph.merge_definition_graph(
                    fourlang_graph, node, substitute)

        return [node[1]["name"] for node in fourlang_graph.G.nodes(data=True)]

    def expand(self, graph, depth=1, substitute=False, expand_set=set(), strategy="None", use_concept_def=False):
        if depth == 0:
            return

        if not expand_set:
            nodes = [node for node in graph.G.nodes(data=True)]
        else:
            nodes = [node for node in graph.G.nodes(
                data=True) if node[1]["name"] in expand_set]
        for d_node, node_data in nodes:
            if all(
                    elem not in node_data
                    for elem in ["expanded", "substituted"]):
                node = graph.d_clean(node_data["name"]).split('_')[0]
                if (not expand_set and (node not in self.lexicon.stopwords or d_node == graph.root)) or expand_set:
                    if use_concept_def:
                        related_words = self.expand_with_concept(graph, d_node)
                        if expand_set:
                            expand_set |= set(related_words)
                    else:
                        definition = self.lexicon.get_definition(node)
                        if definition:
                            definition_nodes = self.add_definition(
                                graph, d_node, definition, substitute, strategy)
                            if expand_set:
                                expand_set |= set(definition_nodes)

        self.expand(graph, depth-1, substitute=substitute,
                    expand_set=expand_set, strategy=strategy, use_concept_def=use_concept_def)

    def expand_with_concept(self, graph, node):
        negative_names = {"antonym", "distinct_from"}
        positive_names = {"is_a", "synonym", "form_of", "derived_from"}
        word = graph.G.nodes()[node]["name"]
        index = node * 100
        words = []
        if word in self.lexicon.concept_graph.adj:
            adjacency = self.lexicon.concept_graph.adj[word]
            for adj in adjacency._atlas.items():
                edge_names = [attributes["name"] for (_, attributes) in adj[1].items()]
                if len(negative_names.intersection(edge_names)) == 0 and \
                        len(positive_names.intersection(edge_names)) > 0:
                    graph.G.nodes[node]['expanded'] = True
                    graph.G.add_node(index, name=adj[0])
                    graph.G.add_edge(node, index, color=0)
                    words.append(adj[0])
                    index += 1
        return words

    def negate(self, graph):
        node_dict = {}
        delete_list = []
        for adj in graph.G.adj.items():
            adjacents = {graph.G.nodes(data=True)[edge[0]]["name"]: edge[0]
                         for edge in adj[1].items() if edge[1]["color"] == 0}
            if "NEG" in adjacents:
                if graph.G.nodes(data=True)[adj[0]]["name"] == "PER":
                    node_dict[adj[0]] = {"name": "FOR"}
                    delete_list.append(adjacents["NEG"])
                elif graph.G.nodes(data=True)[adj[0]]["name"] == "FOR":
                    node_dict[adj[0]] = {"name": "PER"}
                    delete_list.append(adjacents["NEG"])
                else:
                    root_antonym = self.lexicon.get_antonym(graph.G.nodes(data=True)[adj[0]]["name"])
                    if root_antonym is not None:
                        node_dict[adj[0]] = {"name": root_antonym}
                        delete_list.append(adjacents["NEG"])
        graph.delete_nodes(delete_list)
        graph.update_nodes(node_dict)

    def parse(self, sen):
        fl = self.ud_fl.parse(sen, 'ud', "fl", 'amr-sgraph-src')

        graph, root = pn_to_graph(fl)

        relabeled_graph = self.graph_lexical.from_plain(graph)

        return relabeled_graph, self.graph_lexical.vocab.get_id(
            graph.nodes[root]["name"])

    def __enter__(self):
        self.nlp.__enter__()
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.nlp.__exit__(exc_type, exc_value, exc_traceback)


class TextTo4lang(To4lang):
    def __call__(self, text, depth=0, substitute=False, expand_set=set(), strategy="None"):
        for sen in self.nlp(text).sentences:
            graph, root = self.parse(sen)

            fourlang = FourLang(graph, root, self.graph_lexical)

            self.expand(fourlang, depth=depth, substitute=substitute, expand_set=expand_set, strategy=strategy)
            yield fourlang.G


class UDTo4lang(To4lang):
    def __call__(self, doc, depth=0, substitute=False, expand_set=set(), strategy="None"):
        for sen in doc.sentences:
            graph, root = self.parse(sen)

            fourlang = FourLang(graph, root, self.graph_lexical)

            self.expand(fourlang, depth=depth, substitute=substitute, expand_set=expand_set, strategy=strategy)
            fourlang.restore_accents([w.lemma for s in doc.sentences for w in s.words])
            #self.print_graph(fourlang)
            self.negate(fourlang)
            #self.print_graph(fourlang)
            yield fourlang#.G

    def print_graph(self, fourlang):
        import graphviz

        vis_graph = graphviz.Digraph()
        for node in fourlang.G.nodes(data=True):
            vis_graph.node(str(node[0]), node[1]["name"])
        for edge in fourlang.G.edges(data=True):
            vis_graph.edge(str(edge[0]), str(edge[1]), str(edge[2]["color"]))
        g = graphviz.Source(vis_graph.source)
        g.view()


def get_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-cd", "--cache-dir", default=None, type=str)
    parser.add_argument("-cn", "--nlp-cache", default=None, type=str)
    parser.add_argument("-l", "--lang", default=None, type=str)
    parser.add_argument("-d", "--depth", default=0, type=int)
    parser.add_argument("-s", "--substitute", default=False, type=bool)
    parser.add_argument("-p", "--preprocessor", default=None, type=str)
    return parser.parse_args()


def main():
    logging.basicConfig(
        format="%(asctime)s : " +
        "%(module)s (%(lineno)s) - %(levelname)s - %(message)s")
    logging.getLogger().setLevel(logging.WARNING)
    args = get_args()
    preproc = Preprocessor(args.preprocessor)
    with TextTo4lang(args.lang, args.nlp_cache, args.cache_dir) as tfl:
        for i, line in tqdm(enumerate(sys.stdin)):
            try:
                fl_graphs = list(tfl(preproc(line.strip())))
            except (TypeError, IndexError, KeyError):
                traceback.print_exc()
                sys.stderr.write(f'error on line {i}: {line}')
                print('ERROR')
                continue
                # sys.exit(-1)

            print("\t".join(graph_to_pn(fl) for fl in fl_graphs))


if __name__ == "__main__":
    main()
