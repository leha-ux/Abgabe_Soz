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

def plot_wordcloud(df, custom_stopwords=set(), colormap="Greys"):
    nltk.download("stopwords")
    german_stopwords = set(stopwords.words("german"))
    all_stopwords = set(STOPWORDS).union(german_stopwords).union(custom_stopwords)
    mask = np.array(Image.open("DeutscherBundestagLogo.png"))
    font = "quartzo.ttf"
    cmap = mpl.cm.get_cmap(colormap)(np.linspace(0, 1, 20))
    cmap = mpl.colors.ListedColormap(cmap[10:15])
    text = df
    wc = WordCloud(
        background_color="white",
        font_path=font,
        stopwords=all_stopwords,
        contour_width=0,
        max_words=1000,
        colormap=cmap,
        mask=mask,
        random_state=42,
        collocations=False,
        min_word_length=1,
        max_font_size=100,
    )
    wc.generate(text)
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    return fig

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

print("Filtered DATA TOPIC", filtered_data.head(100))
filtered_data.to_csv("updated_processed_dataframe.csv", index=True)

unique_dates = sorted(data['date'].unique())
st.sidebar.subheader("Select Period for Speeches")
start_date, end_date = st.sidebar.select_slider(
    "Select a period for speeches",
    options=unique_dates,
    value=(unique_dates[0], unique_dates[-1])
)

filtered_data = filtered_data[(filtered_data['date'] >= start_date) & (filtered_data['date'] <= end_date)]
select = st.sidebar.selectbox('Visualization type', ['Bar plot', 'Pie chart'], key='1')
selected_sentiments = st.sidebar.multiselect('Select Sentiments', options=['positive', 'negative', 'neutral'], default=['positive', 'negative', 'neutral'])
filtered_data = filtered_data[filtered_data['Sentiment_Label'].isin(selected_sentiments)]

unique_parties = sorted(data['party'].unique())
selected_parties = st.sidebar.multiselect('Select parties', unique_parties, default=unique_parties)
filtered_data = filtered_data[filtered_data['party'].isin(selected_parties)]
sentiment_count = filtered_data['Sentiment_Label'].value_counts()
sentiment_count = pd.DataFrame({'Sentiment': sentiment_count.index, 'Sentences': sentiment_count.values})

area_chart_data = filtered_data.groupby(['date', 'Sentiment_Label']).size().unstack().fillna(0)
for sentiment in ['positive', 'negative', 'neutral']:
    if sentiment not in area_chart_data.columns:
        area_chart_data[sentiment] = 0

area_chart_data['negative'] = area_chart_data['negative'] * -1

st.markdown("### Number of sentences by sentiment")
col1, col2 = st.columns(2)

with col1:
    if select == 'Bar plot':
        fig = px.bar(sentiment_count, x='Sentiment', y='Sentences', color='Sentences', height=500)
        st.plotly_chart(fig, use_container_width=True)
    else:
        fig = px.pie(sentiment_count, values='Sentences', names='Sentiment')
        st.plotly_chart(fig, use_container_width=True)

with col2:
    st.area_chart(area_chart_data, use_container_width=True, height=500)

st.markdown("### Sentiment by party")

bar_chart_data = filtered_data.groupby(['party', 'Sentiment_Label']).size().unstack().fillna(0)
bar_chart_data = bar_chart_data.div(bar_chart_data.sum(axis=1), axis=0) * 100
fig = px.bar(bar_chart_data, y=bar_chart_data.index, x=selected_sentiments, title='Sentiment by party (%)', labels={'value': 'Percentage (%)', 'variable': 'Sentiment'}, height=400)
st.plotly_chart(fig, use_container_width=True)

st.sidebar.header("Word Cloud")
word_sentiment = st.sidebar.radio('Display word cloud for what sentiment?', ('positive', 'neutral', 'negative'))
st.subheader('Word cloud for %s sentiment' % (word_sentiment))

custom_stopwords = st_tags(label='Add Custom Stopwords:',
                           value=['Vielen', 'Dank', 'Frau', 'Herren', 'Liebe', 'Präsidentin', 'Kollegen', 'Herr', 'Herzlichen', 'Danke', 'Damen', 'Kolleginnen', 'Kollege', 'Geehrte', 'Kollegin', 'Herzlich', 'Präsident'],
                           maxtags=50,
                           key='007')

df = filtered_data[filtered_data['Sentiment_Label'] == word_sentiment]
words = ' '.join(df['satz'])
processed_words = ' '.join([word for word in words.split()])

# Check if processed_words is empty
if not processed_words:
    st.write("No words available for the current filters.")
else:
    wordcloud = plot_wordcloud(processed_words, set(custom_stopwords))
    st.pyplot(wordcloud)



#Tables for Filtering Speech and Politician
# Exclude 'neutral' sentiment
non_neutral_data = data[data['Sentiment_Label'] != 'neutral']

# Map 'positive' and 'negative' sentiments to 1 and -1 respectively
mapped_data = non_neutral_data.copy()
mapped_data['Sentiment_Label'] = mapped_data['Sentiment_Label'].map({'positive': 1, 'negative': 0})

# Create columns for the text inputs
col1, col2 = st.columns(2)

# Create text input for name and discussion_title filtering inside the columns
name_filter = col1.text_input("Filter by Politician's Name:", "")
discussion_title_filter = col2.text_input("Filter by Discussion Title:", "")

# Filter data based on user input
filtered_data = mapped_data
if name_filter:
    filtered_data = filtered_data[filtered_data['name'].str.contains(name_filter, case=False)]

if discussion_title_filter:
    filtered_data = filtered_data[filtered_data['discussion_title'].str.contains(discussion_title_filter, case=False)]

# Calculate average sentiment for each speech using the filtered data
average_sentiment_speech = filtered_data.groupby(['name', 'date', 'discussion_title', 'party'])['Sentiment_Label'].mean().reset_index()

# Calculate average sentiment for each politician using the filtered data
average_sentiment_politician = filtered_data.groupby(['name', 'party'])['Sentiment_Label'].mean().reset_index()

# Display the dataframes
st.markdown("### Sentiment Overview")
st.dataframe(filtered_data, use_container_width=True)

st.markdown("### Average Sentiment by Speech")
st.dataframe(average_sentiment_speech, use_container_width=True)

st.markdown("### Average Sentiment by Politician")
st.dataframe(average_sentiment_politician, use_container_width=True)


# Define columns for the visualizations

# Visualization for Average Sentiment by Politician by Party
avg_sentiment_by_party_politician = average_sentiment_politician.groupby('party')['Sentiment_Label'].mean()
st.markdown("### Average Sentiment by Politician by Party")
st.bar_chart(avg_sentiment_by_party_politician)



reset_filters = st.sidebar.button('Reset Filters')

# If reset button is clicked, reset filters
if reset_filters:
    selected_speakers = []
    selected_topics = []
    select = 'Bar plot'
    selected_sentiments = ['positive', 'negative', 'neutral']
    selected_parties = sorted(data['party'].unique())
    start_date, end_date = unique_dates[0], unique_dates[-1]


