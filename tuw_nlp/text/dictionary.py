import os
import re
import json
from collections import defaultdict
from nltk.corpus import stopwords as nltk_stopwords
from tqdm import tqdm

from tuw_nlp.graph.utils import preprocess_node_alto
import networkx as nx


class Dictionary:
    def __init__(self, lang):
        self.lexicon = defaultdict(list)
        self.antonyms = defaultdict(list)
        self.concept_graph = nx.MultiDiGraph()
        self.lang_map = {}
        base_fn = os.path.dirname(os.path.abspath(__file__))
        langnames_fn = os.path.join(base_fn, "langnames")

        self.lang = lang
        self.base_path = os.path.expanduser("~/tuw_nlp_resources")

        definitions_base_fn = os.path.join(self.base_path, "definitions", lang.split("_")[0])
        antonyms_base_fn = os.path.join(self.base_path, "antonyms", f"{lang.split('_')[0]}.json")
        concept_graphs_base_fn = os.path.join(self.base_path, "concept_graphs", f"{lang.split('_')[0]}.json")

        definitions_fn = None
        if os.path.isfile(definitions_base_fn):
            definitions_fn = definitions_base_fn

        assert (
            definitions_fn
        ), "Definition dictionaries are not downloaded, for setup please use tuw_nlp.download_definitions(), otherwise you will not be able to use expand functionalities"

        antonyms_fn = None
        if os.path.isfile(antonyms_base_fn):
            antonyms_fn = antonyms_base_fn

        concept_graphs_fn = None
        if os.path.isfile(concept_graphs_base_fn):
            concept_graphs_fn = concept_graphs_base_fn

        if antonyms_fn is not None and concept_graphs_fn is not None:
            with open(antonyms_base_fn, "r") as antonyms_file:
                self.antonyms = json.load(antonyms_file)
            with open(concept_graphs_base_fn, "r") as concepts_file:
                concept_json = json.load(concepts_file)
                self.concept_graph = nx.node_link_graph(concept_json)
        elif antonyms_fn is not None:
            with open(antonyms_base_fn, "r") as antonyms_file:
                self.antonyms = json.load(antonyms_file)
            create = input("The concept_graphs are not available. Do you wish to create them? (Y/n)")
            if not create.lower().startswith("n"):
                os.makedirs(os.path.dirname(concept_graphs_base_fn), exist_ok=True)
                self.create_antonyms_and_concept_graphs(antonyms=False)
                concept_json = nx.node_link_data(self.concept_graph)
                with open(concept_graphs_base_fn, "w") as concepts_file:
                    json.dump(concept_json, concepts_file, indent=4)
        elif concept_graphs_fn is not None:
            os.makedirs(os.path.dirname(antonyms_base_fn), exist_ok=True)
            with open(concept_graphs_base_fn, "r") as concepts_file:
                concept_json = json.load(concepts_file)
                self.concept_graph = nx.node_link_graph(concept_json)
            self.get_antonyms_from_concept_graph()
            with open(antonyms_base_fn, "w") as antonyms_file:
                json.dump(self.antonyms, antonyms_file, indent=4)
        else:
            create = input("The antonyms and concept_graphs are not available. Do you wish to create them? (Y/n)")
            if not create.lower().startswith("n"):
                os.makedirs(os.path.dirname(antonyms_base_fn), exist_ok=True)
                os.makedirs(os.path.dirname(concept_graphs_base_fn), exist_ok=True)
                self.create_antonyms_and_concept_graphs()
                with open(antonyms_base_fn, "w") as antonyms_file:
                    json.dump(self.antonyms, antonyms_file, indent=4)
                concept_json = nx.node_link_data(self.concept_graph)
                with open(concept_graphs_base_fn, "w") as concepts_file:
                    json.dump(concept_json, concepts_file, indent=4)

        with open(langnames_fn, "r", encoding="utf8") as f:
            for line in f:
                line = line.split("\t")
                self.lang_map[line[0]] = line[1].strip("\n")

        self.stopwords = set(nltk_stopwords.words(self.lang_map[lang]))
        self.__init_lexicons(definitions_fn)

    def __init_lexicons(self, definitions_fn):
        with open(definitions_fn, "r", encoding="utf8") as f:
            for line in f:
                line = line.split("\t")
                if len(line[2].strip().strip("\n")) > 5:
                    word = line[0].strip()

                    defi = line[2].strip().strip("\n")
                    defi = self.parse_definition(defi)
                    if defi.strip() != word:
                        def_splitted = defi.strip().split(";")
                        for def_split in def_splitted:
                            if def_split not in self.lexicon[word]:
                                self.lexicon[word].append(def_split)
                                self.lexicon[preprocess_node_alto(word)].append(
                                    def_split
                                )

    def parse_definition(self, defi):
        defi = re.sub(re.escape("#"), " ", defi).strip()

        defi = re.sub(r"^A type of", "", defi)
        defi = re.sub(r"^Something that", "", defi)
        defi = re.sub(r"^Relating to", "", defi)
        defi = re.sub(r"^Someone who", "", defi)
        defi = re.sub(r"^Of or", "", defi)
        defi = re.sub(r"^Any of", "", defi)
        defi = re.sub(r"^The act of", "", defi)
        defi = re.sub(r"^A group of", "", defi)
        defi = re.sub(r"^The part of", "", defi)
        defi = re.sub(r"^One of the", "", defi)
        defi = re.sub(r"^Used to", "", defi)
        defi = re.sub(r"^An attempt to", "", defi)

        defi = re.sub(r"^intransitive", "", defi)
        defi = re.sub(r"^ditransitive", "", defi)
        defi = re.sub(r"^ambitransitive", "", defi)
        defi = re.sub(r"^transitive", "", defi)
        defi = re.sub(r"^uncountable", "", defi)
        defi = re.sub(r"^countable", "", defi)
        defi = re.sub(r"^pulative ", "", defi)
        defi = re.sub(r"^\. ", "", defi)
        defi_words = defi.split(" ")
        first_words = defi_words[0].split(",")
        if len(first_words) > 1 and re.sub("'s", "", first_words[0].lower()) == re.sub(
            "'s", "", first_words[1].lower()
        ):
            defi = " ".join([first_words[1]] + defi_words[1:])
        return defi

    def get_definition(self, word):
        return self.lexicon[word][0] if self.lexicon[word] else None

    def get_antonyms_from_concept_graph(self):
        for root, edges in self.concept_graph.adj._atlas.items():
            for end, edge_dict in edges.items():
                edge_names = [edge["name"] for edge in edge_dict.values()]
                if "antonym" in edge_names:
                    if root not in self.antonyms:
                        self.antonyms[root] = [end]
                    elif end not in self.antonyms[root]:
                        self.antonyms[root].append(end)
                    if end not in self.antonyms:
                        self.antonyms[end] = [root]
                    elif root not in self.antonyms[end]:
                        self.antonyms[end].append(root)

    def create_antonyms_and_concept_graphs(self, antonyms=True, concept_graphs=True):
        if not antonyms and not concept_graphs:
            return

        from conceptnet_lite import connect, Language
        assert os.path.exists(os.path.join(self.base_path, "conceptnet.db")), \
            "To be able to access antonyms, please download conceptnet using tuw_nlp.download_conceptnet()"

        connect(os.path.join(self.base_path, "conceptnet.db"))
        current_language = Language.get(name=self.lang)
        for label in tqdm(current_language.labels):
            label_name = ' '.join(label.text.split('_'))
            for c in label.concepts:
                if c.edges_out:
                    for e in c.edges_out:
                        # The German conceptent is very limited without other language's wiktionary instances...
                        if (e.end.label.language is None or e.end.label.language.name == self.lang) \
                                and e.relation.name != "external_url":  # \
                                #and (("wiktionary" in e.etc["dataset"] and e.etc["dataset"].endswith(self.lang))
                                #     or "wiktionary" not in e.etc["dataset"]):
                            other = ' '.join(e.end.label.text.split('_'))
                            if concept_graphs:
                                self.concept_graph.add_edge(label_name, other, name=e.relation.name)
                            if antonyms and e.relation.name == "antonym":
                                if label_name not in self.antonyms:
                                    self.antonyms[label_name] = []
                                if other not in self.antonyms:
                                    self.antonyms[other] = []
                                if other not in self.antonyms[label_name]:
                                    self.antonyms[label_name].append(other)
                                if label_name not in self.antonyms[other]:
                                    self.antonyms[other].append(label_name)
        # Realtions in de: 'distinct_from', 'antonym', 'derived_from', 'etymologically_related_to', 'is_a', 'part_of', 'external_url', 'related_to', 'form_of', 'etymologically_derived_from', 'synonym', 'similar_to'

    def get_antonym(self, text):
        if text in self.antonyms:
            return self.antonyms[text][0]
        return None
