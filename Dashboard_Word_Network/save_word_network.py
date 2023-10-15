import streamlit as st
import requests
import networkx as nx
import tqdm
import collections
import json

def create_network_from_sentences(sentences):
    relevantPOS = ['NOUN','ADJ','PROPN']
    sentencesNN = []
    words = []
    for sen in sentences:
        lem = []
        for token in sen:
            if token.pos_ in relevantPOS:
                lem.append(token.lemma_)
        sentencesNN.append(lem)
        words.extend(lem)

    nodes = []
    curid = 1
    for word in set(words):
        node = {
            'id': curid,
            'name': word
        }
        nodes.append(node)
        curid += 1

    graph = {
        'directed': False,
        'graph': 'word_graph',
        'links': [],
        'nodes': nodes
    }

    links = []
    linkedwords = []
    for wx1, w1 in enumerate(nodes):
        for wx2, w2 in enumerate(nodes):
            if w2['id'] > w1['id']:
                for sen in sentencesNN:
                    if w1['name'] in sen and w2['name'] in sen:
                        weight = linkedwords.count(' '.join([w1['name'], w2['name']]))
                        linkedwords.append(' '.join([w1['name'], w2['name']]))
                        link_dict = {
                            'source': w1['id'],
                            'target': w2['id'],
                            'sourceWD': w1['name'],
                            'targetWD': w2['name'],
                            'weight': weight + 1
                        }
                        links.append(link_dict)
                        graph['links'].append(link_dict)

    zähler = collections.Counter(linkedwords).most_common()
    return graph

def save_to_html(graph):
    json_payload = {
        'data': graph,
        'nodecoloring': 'party',
        'nodelabel': 'name',
        'darkmode': False,
        'edgevisibility': True,
        'particles': False
    }
    result = requests.post('https://penelope.vub.be/network-components/visualiser', json=json_payload)
    with open(f"./wordnet.html", "w") as f:
        f.write(result.json()['graph'])

def save_to_gexf(graph):
    graphforgephi = nx.Graph()
    for node in graph['nodes']:
        graphforgephi.add_node(node['id'], name=node['name'])

    for link in graph['links']:
        graphforgephi.add_edge(link['source'], link['target'], weight=link['weight'])

    nx.write_gexf(graphforgephi, "graphforgephi.gexf")

# Streamlit App
st.title('Sentence Network Creator')

if st.button("Load Data"):
    with open("../../data/speeches_20.jsonl", 'r', encoding="utf8") as fp:
        data = list(fp)

    speeches = []
    for line in data:
        speeches.append(json.loads(line))

    # Assuming each entry in 'speeches' contains the 'sentence' key. 
    # This needs to be adjusted based on your data structure.
    sentences = [speech['sentence'] for speech in speeches]

    st.write("Creating network from sentences...")
    graph = create_network_from_sentences(sentences)
    st.write(f"Created network with {len(graph['nodes'])} nodes and {len(graph['links'])} links.")

    save_option = st.radio("Choose Save Option", ["Save as HTML", "Save as GEXF"])
    if save_option == "Save as HTML":
        save_to_html(graph)
        st.success("Network saved as wordnet.html!")
    elif save_option == "Save as GEXF":
        save_to_gexf(graph)
        st.success("Network saved as graphforgephi.gexf!")
