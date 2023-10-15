from wordcloud import WordCloud, STOPWORDS
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import plotly.io as pio
import matplotlib as mpl
from wordcloud import WordCloud
from PIL import Image
import nltk
from nltk.corpus import stopwords
from streamlit_tags import st_tags, st_tags_sidebar


# Let users select a DataFrame based on a model name
selected_model = st.sidebar.selectbox(
    'Select model for DataFrame source:',
    ['BERT', 'GPT']
)

# Map the user's selection to the corresponding filename
data_file_map = {
    'BERT': 'processed_dataframe.csv',
    'GPT': 'processed_output_gpt.csv'
}
DATA_URL = data_file_map[selected_model]



st.title("Sentiment Analysis Bundestag Legislatur 20")
st.sidebar.title("Sentiment Analysis")
st.markdown("Dieses Dashboard dient dem Explorieren der Sentiments der im Bundestag gehaltenen Reden der 20. Legislatur")
st.sidebar.markdown("Dieses Dashboard dient dem Explorieren der Sentiments der im Bundestag gehaltenen Reden der 20. Legislatur")

@st.cache_data(persist=False)
def load_data():
    data = pd.read_csv(DATA_URL)
    return data

data = load_data()
data.to_csv("updated_processed_dataframe_v2.csv", index=False)

unique_speakers = sorted(data['name'].unique())
st.sidebar.markdown("### Select Speaker(s) and/or Topic(s)")
selected_speakers = st.sidebar.multiselect('The filter is not applied, if no speaker is selected.', unique_speakers, default=[])

if selected_speakers:
    filtered_data = data[data['name'].isin(selected_speakers)]
else:
    filtered_data = data.copy()

unique_topics = sorted(data['topic'].unique())
selected_topics = st.sidebar.multiselect('The filter is not applied, if no topic is selected.', unique_topics, default=[])

if selected_topics:
    filtered_data = filtered_data[filtered_data['topic'].isin(selected_topics)]

filtered_data.to_csv("updated_processed_dataframe.csv", index=True)

unique_dates = sorted(data['date'].unique())
st.sidebar.subheader("Select Period for Speeches")
start_date, end_date = st.sidebar.select_slider(
    "Select a period for speeches",
    options=unique_dates,
    value=(unique_dates[0], unique_dates[-1])
)

filtered_data = filtered_data[(filtered_data['date'] >= start_date) & (filtered_data['date'] <= end_date)]
selected_sentiments = st.sidebar.multiselect('Select Sentiments', options=['positive', 'negative', 'neutral'], default=['positive', 'negative', 'neutral'])
filtered_data = filtered_data[filtered_data['Sentiment_Label'].isin(selected_sentiments)]

unique_parties = sorted(data['party'].unique())
selected_parties = st.sidebar.multiselect('Select parties', unique_parties, default=unique_parties)
filtered_data = filtered_data[filtered_data['party'].isin(selected_parties)]

import requests
import json
import spacy
import tempfile
import networkx as nx
from tqdm import tqdm
from pyvis.network import Network

nlp = spacy.load("de_core_news_sm")

def load_speeches(filepath):
    with open(filepath, 'r', encoding="utf8") as fp:
        return [json.loads(line) for line in fp]

speeches = load_speeches("data/speeches_20.jsonl")

def filter_for(what, search_terms, speeches):
    search_terms_low = [term.lower() for term in search_terms]
    if what == 'text':
        return sorted(
            [speech for speech in speeches if all(term in speech[what].lower() for term in search_terms_low)],
            key=lambda x: x['date']
        )
    else:
        return sorted(
            [speech for speech in speeches if speech[what].lower() in set(search_terms_low)],
            key=lambda x: x['date']
        )

st.sidebar.title("Filter by Focal Terms")
focal_terms = st_tags_sidebar(
    label='Enter Focal Terms:',
    text='Press enter to add',
    value=['Digitalisierung', 'Zusammenhalt', 'Demokratie'],
    maxtags=10,
    key='10000'
)


# Added this new code block to take single word input in the sidebar.
st.sidebar.title("Input a Single Word to filter for")
single_search_word = st.sidebar.text_input('Enter your word:', value='Digitalisierung')

colorize_edges_checkbox = st.sidebar.checkbox("Do you want to colorize the edges according to party affiliation?", True)

st.sidebar.markdown(f"### Number of Entries in Current Selection: {len(filtered_data)}")


def extract_sentences(subset, focus_term):
    sentences = []
    for rede in subset:
        doc = nlp(rede["satz"])
        for sent in doc.sents:
            if focus_term.lower() in sent.text.lower():
                sentences.append({"sent": sent, "name": rede['name'], "party": rede['party']})
    return sentences

def get_lemmas_from_sentences(sentences, relevant_pos):
    sentences_nn = []
    words = []
    for sen in sentences:
        lem = [token.lemma_ for token in sen["sent"] if token.pos_ in relevant_pos]
        sentences_nn.append({"lem": lem, "name": sen["name"], "party": sen["party"]})
        words.extend(lem)
    return sentences_nn, words

def build_graph(sentences_nn):
    words = [sen['lem'] for sen in sentences_nn]
    unique_words = list(set([item for sublist in words for item in sublist]))

    nodes = [{"id": idx + 1, "name": word} for idx, word in enumerate(unique_words)]
    
    links = []
    linked_words = []
    
    for idx1, w1 in enumerate(nodes):
        for idx2, w2 in enumerate(nodes):
            if w2['id'] > w1['id']:
                for sen in sentences_nn:
                    if w1['name'] in sen["lem"] and w2['name'] in sen["lem"]:
                        weight = linked_words.count(' '.join([w1['name'], w2['name']]))
                        linked_words.append(' '.join([w1['name'], w2['name']]))
                        link_dict = {
                            'source': w1['id'],
                            'target': w2['id'],
                            'sourceWD': w1['name'],
                            'targetWD': w2['name'],
                            'weight': weight + 1,
                            'name': sen["name"],
                            'party': sen["party"]
                        }
                        links.append(link_dict)
    
    graph = {
        'directed': False,
        'graph': 'word_graph',
        'links': links,
        'nodes': nodes
    }
    return graph

def save_graph_to_html(graph, url):
    json_data = {
        'data': graph,
        'nodecoloring': 'party',
        'nodelabel': 'name',
        'darkmode': False,
        'edgevisibility': True,
        'particles': False
    }
    result = requests.post(url, json=json_data)
    with open("./wordnet.html", "w") as f:
        f.write(result.json()['graph'])

def get_graph_html(graph, party_color_map, colorize_edges=True):
    graph_for_phi = nx.Graph()
    for node in tqdm(graph['nodes']):
        graph_for_phi.add_node(node['id'], name=node['name'], label=node['name'])
    for link in tqdm(graph['links']):
        graph_for_phi.add_edge(link['source'], link['target'], weight=link['weight'], party=link['party'])

    nt = Network(notebook=True, cdn_resources='in_line')
    nt.from_nx(graph_for_phi)

    options = """
    {
        "nodes": {
            "font": {
                "size": 14,
                "face": "arial"
            }
        },
        "physics": {
            "solver": "repulsion",
            "repulsion": {
                "nodeDistance": 100
            },
            "stabilization": {
                "enabled": true,
                "iterations": 500
            }
        }
    }
    """
    nt.set_options(options)

    if colorize_edges:
        for edge in nt.edges:
            edge['color'] = party_color_map.get(edge["party"], 'grey')

    with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp_file:
        temp_filename = tmp_file.name
        nt.save_graph(temp_filename)
    with open(temp_filename, 'r') as f:
        html_content = f.read()
    return html_content

def create_word_network(filtered_data, focus_terms, search_term, colorize_edges=True):
    if focal_terms:
        subset = filter_for('text', focal_terms, speeches)
    else:
        subset = speeches.copy()
    unique_ids = [speech["id"] for speech in subset]
    filtered_data = filtered_data[filtered_data['id'].isin(unique_ids)]
    satz_list = filtered_data[['satz', 'name', 'party']].to_dict('records')
    sentences = extract_sentences(satz_list, search_term)
    relevant_pos = ['NOUN', 'ADJ', 'PROPN']
    sentences_nn, words = get_lemmas_from_sentences(sentences, relevant_pos)
    graph = build_graph(sentences_nn)
    save_graph_to_html(graph, 'https://penelope.vub.be/network-components/visualiser')

    party_color_map = {
        'unknown': 'grey',
        'AfD': 'blue',
        'DIE LINKE': 'purple',
        'BÜNDNIS 90/DIE GRÜNEN': 'green',
        'CDU/CSU': 'black',
        'SPD': 'red',
        'FDP': 'yellow',
        'Fraktionslos': 'grey'
    }
    html_content = get_graph_html(graph, party_color_map, colorize_edges)
    return html_content

if st.button("Generate Word Network based on the selected filters"):
    # Only execute the following code when the button is clicked
    html_return = create_word_network(
        filtered_data=filtered_data,
        colorize_edges=colorize_edges_checkbox,
        focus_terms=focal_terms,
        search_term=single_search_word
    )
    st.components.v1.html(html_return, width=800, height=600)