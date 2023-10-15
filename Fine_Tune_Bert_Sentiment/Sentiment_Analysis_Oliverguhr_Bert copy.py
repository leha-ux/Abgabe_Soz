# Import Basic Packages
import numpy as np                 # Numpy
import pandas as pd                 #Datafrane

# Import Visualization Packages
from collections import Counter     # um worte zu zählen
import matplotlib.pyplot as plt   # Für Visualisierung
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator #Wordcloud erstellen
import plotly.express as px
import plotly.graph_objs as go
import plotly.subplots as sp

# Import NLP Packages
import nltk
pd.options.mode.chained_assignment = None
import spacy
import string
from nltk.corpus import stopwords
from collections import Counter
import plotly.graph_objects as go
from transformers import pipeline
import plotly.express as px

# Methode 9 zum Visualisierung von Wordclouds nach Sentiment
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from tqdm import tqdm




#  Methode 1: Gibt alle Reden aus dem DataFrame zurück, die dem angegebenen Datum und Tagesordnungspunkt entsprechen.

def get_reden_by_datum_tagesordnungspunkt(df, datum, tagesordnungspunkt):
    """
    Args:
        df (pandas.DataFrame): Der DataFrame mit den Reden.
        datum (str): Das gesuchte Datum im Format 'YYYY-MM-DD'.
        tagesordnungspunkt (str): Der gesuchte Tagesordnungspunkt.
        
    Returns:
        pandas.DataFrame: Ein DataFrame mit den gefundenen Reden.
    """
    filtered_df = df[(df['date'] == datum) & (df['discussion_title'] == tagesordnungspunkt)]
    return filtered_df



#Methode 2:  Tokenisiert die Sätze in der Spalte 'text' des DataFrames und speichert sie in einem neuen DataFrame.


def tokenize_and_split_sentences(df):
    """
        Args:
        df (pandas.DataFrame): Der DataFrame, in dem die Tokenisierung durchgeführt werden soll.

    Returns:
        pandas.DataFrame: Der bearbeitete DataFrame mit den tokenisierten Sätzen.
    """
    # Satztokenisierung mit NLTK
    df['tokenized_text'] = df['text'].apply(nltk.sent_tokenize)
    rows_list = []
    for _, row in df.iterrows():
        tokenized_text = row['tokenized_text']
        for satz in tokenized_text:
            dict1 = {'satz': satz, 'id': row['id'], 'party': row['party'], 'period': row['period'],
                    'date': row['date'], 'name': row['name'], 'party': row['party'],
                    'redner_id': row['redner_id'], 'discussion_title': row['discussion_title']}
            rows_list.append(dict1)
    df_token_satz = pd.DataFrame(rows_list)
    return df_token_satz




def clean_text(df, custom_stopwords=None):
    cleaned_sentences = []
    cleaned_tokens = []  # New list to store cleaned tokens
    
    # German stopwords
    stopwords_german = set(stopwords.words('german')) - {'nicht'} 
    
    # Update stopwords if custom stopwords are provided
    if custom_stopwords:
        stopwords_german.update(custom_stopwords)

    for sentence in df['satz']:
        # Tokenisierung mit Spacy
        doc = nlp(sentence)
        tokens = [token.text for token in doc if token.text not in string.punctuation]

        # Entfernung von Stoppwörtern mit NLTK
        filtered_tokens = [token for token in tokens if token.lower() not in stopwords_german]
        
        # Zusammenführen der bereinigten Tokens zu einem Satz
        cleaned_sentence = ' '.join(filtered_tokens)
        cleaned_sentences.append(cleaned_sentence)
        
        # Store cleaned tokens separately
        cleaned_tokens.append(filtered_tokens)

    # Assign the cleaned tokens to the DataFrame
    df['tokens'] = cleaned_tokens
    
    # Erstellung einer neuen Spalte 'cleaned_text' im DataFrame mit den bereinigten Sätzen
    df['cleaned_text'] = cleaned_sentences

    # Delete rows with empty 'cleaned_tokens'
    df = df[df['tokens'].map(len) > 0]
    
    return df



def plot_most_frequent_ngrams(df, num_most_common=10):
    # Get the tokens from the DataFrame
    tokens = list(df['cleaned_text'].values)

    # Count unigrams
    unigram_counts = Counter()
    for text in tokens:
        unigrams = text.split()
        unigram_counts.update(unigrams)

    # Count bigrams
    bigram_counts = Counter()
    for text in tokens:
        unigrams = text.split()
        bigrams = [",".join(bigram) for bigram in zip(unigrams[:-1], unigrams[1:])]
        bigram_counts.update(bigrams)

    # Count trigrams
    trigram_counts = Counter()
    for text in tokens:
        unigrams = text.split()
        trigrams = [",".join(trigram) for trigram in zip(unigrams[:-2], unigrams[1:-1], unigrams[2:])]
        trigram_counts.update(trigrams)

    # Get the most frequent tokens
    most_common_unigrams = unigram_counts.most_common(num_most_common)
    most_common_bigrams = bigram_counts.most_common(num_most_common)
    most_common_trigrams = trigram_counts.most_common(num_most_common)

    # Create the plot
    fig, axes = plt.subplots(3, 1, figsize=(8, 12))

    # Plot most frequent unigrams
    axes[0].barh([str(gram) for gram, count in most_common_unigrams], [count for gram, count in most_common_unigrams])
    axes[0].set_title('Most Frequent Unigrams')

    # Plot most frequent bigrams
    axes[1].barh([str(gram) for gram, count in most_common_bigrams], [count for gram, count in most_common_bigrams])
    axes[1].set_title('Most Frequent Bigrams')

    # Plot most frequent trigrams
    axes[2].barh([str(gram) for gram, count in most_common_trigrams], [count for gram, count in most_common_trigrams])
    axes[2].set_title('Most Frequent Trigrams')

    plt.tight_layout()
    plt.show()



# Methode 5 zur Modellentwicklung



def sentiment_analysis(df, text_column):
    # Define the sentiment analysis model
    nlp_sentiment = pipeline("sentiment-analysis", model='oliverguhr/german-sentiment-bert')
    
    # Apply sentiment analysis to the specified text column in the DataFrame
    tqdm.pandas()
    df['Sentiment'] = df[text_column].progress_apply(lambda x: nlp_sentiment(x))
    
    # Extract sentiment label and score
    df['Sentiment_Label'] = [x[0]['label'] for x in df['Sentiment']]
    df['Sentiment_Score'] = [x[0]['score'] for x in df['Sentiment']]
    
    # Remove the 'Sentiment' column
    df = df.drop(columns=['Sentiment'])
    
    return df





#Methoden 6 zur Visualisierung des Sentiments


def plot_sentiment_analysis(df_grundrechte_original, df_grundrechte_cleaned):
    # Count the frequency of each sentiment label
    df1_count = df_grundrechte_original['Sentiment_Label'].value_counts()
    df2_count = df_grundrechte_cleaned['Sentiment_Label'].value_counts()

    # Set the color palette
    colors = {'Positive': 'mediumseagreen', 'Negative': 'crimson', 'Neutral': 'royalblue'}

    # Create bar plots for sentiment distribution
    figure1 = px.bar(x=df1_count.index, y=df1_count.values, color=df1_count.index, color_discrete_map=colors)
    figure2 = px.bar(x=df2_count.index, y=df2_count.values, color=df2_count.index, color_discrete_map=colors)

    # Customize labels and titles
    figure1.update_layout(
        title_text='Sentiment Distribution - Original Text',
        title_font_size=24,
        xaxis_title='Sentiment',
        yaxis_title='Count',
        width=800,
        height=600
    )

    figure2.update_layout(
        title_text='Sentiment Distribution - Cleaned Text',
        title_font_size=24,
        xaxis_title='Sentiment',
        yaxis_title='Count',
        width=800,
        height=600
    )

    # Display the plots
    figure1.show()
    figure2.show()


# Methoden 7 zur Visualisierung nach Parteizugehörigkeit


def plot_sentiment_by_party (df):
    # Group the data by party and sentiment label and count the occurrences
    party_sentiment = df.groupby(['party', 'Sentiment_Label']).size().reset_index(name='Count')

    # Calculate the total count for each party
    party_count = party_sentiment.groupby('party')['Count'].sum()

    # Calculate the percentage of each sentiment category for each party
    party_sentiment['Percentage'] = party_sentiment.apply(lambda row: row['Count'] / party_count[row['party']] * 100, axis=1)

    # Create separate dataframes for each sentiment label
    positive_df = party_sentiment[party_sentiment['Sentiment_Label'] == 'positive']
    negative_df = party_sentiment[party_sentiment['Sentiment_Label'] == 'negative']
    neutral_df = party_sentiment[party_sentiment['Sentiment_Label'] == 'neutral']

    # Create the plot
    fig = go.Figure()
    fig.add_trace(go.Bar(x=positive_df['party'], y=positive_df['Count'], name='Positive', marker_color='mediumseagreen',
                         text=positive_df['Percentage'].apply(lambda x: f'{x:.2f}%'),
                         textposition='auto'))
    fig.add_trace(go.Bar(x=negative_df['party'], y=negative_df['Count'], name='Negative', marker_color='crimson',
                         text=negative_df['Percentage'].apply(lambda x: f'{x:.2f}%'),
                         textposition='auto'))
    fig.add_trace(go.Bar(x=neutral_df['party'], y=neutral_df['Count'], name='Neutral', marker_color='royalblue',
                         text=neutral_df['Percentage'].apply(lambda x: f'{x:.2f}%'),
                         textposition='auto'))

    fig.update_layout(
        barmode='group',
        xaxis_title='Partei',
        yaxis_title='Anzahl an Sätzen',
        title='Sentiment-Verteilung nach Parteizugehörigkeit'
    )

    fig.show()

def plot_sentiment_wordclouds(df):
    # Group the data by sentiment label
    sentiment_groups = df.groupby('Sentiment_Label')
    text_by_sentiment = {}

    # Combine the text for each sentiment label
    for sentiment, group in sentiment_groups:
        text_by_sentiment[sentiment] = ' '.join(group['cleaned_text'].tolist())

    # Generate a word cloud for each sentiment
    for sentiment, text in text_by_sentiment.items():
        wordcloud = WordCloud(background_color='black', width=400, height=300, max_words=150, colormap='tab20c').generate(text)

        # Plot the word cloud
        plt.figure(figsize=(12, 6))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(sentiment + ' Sentiment Word Cloud')
        plt.show()


legislatur = 20

# df Legislaturperiode 2020 (24.10.2017-26.09.21)
df20 = pd.read_json(f'../../data/speeches_{legislatur}.jsonl', lines=True)
df20['date'] = pd.to_datetime(df20['date'])
df20.sort_values(by='date')
nlp = spacy.load('de_core_news_sm')
nltk.download('punkt')


print(df20.head())
df_tokenize = tokenize_and_split_sentences(df20)
print("Tokenzied:", df_tokenize.head(15))


modell_processed = sentiment_analysis(df_tokenize, 'satz')
print("Processed:", modell_processed.head(15))


#modell_original = sentiment_analysis(df_grundrechte_cleaned, 'satz')
#modell_original.head (15)

# Save the dataframe
modell_processed.to_csv('processed_dataframe.csv', index=False)