# Importing libraries

import pandas as pd
from datetime import datetime
import spacy
import spacy_transformers

# Storing docs in binary format
from spacy.tokens import DocBin

# Reading the dataset
import os
import json
import pandas as pd

# path to your folder
folder_path = "../../data/speeches_sentiment"


# create an empty DataFrame
df = pd.DataFrame(columns=['Text', 'Sentiment'])

# iterate over all files in the folder
for file_name in os.listdir(folder_path):
    # only process json files
    if file_name.endswith('.json'):
        # read the json file
        with open(os.path.join(folder_path, file_name), 'r') as f:
            data = json.load(f)
        # create a temporary DataFrame from the json data
        temp_df = pd.DataFrame(data['sentiments'])
        # rename the columns
        temp_df.columns = ['Text', 'Sentiment']
        # append the temporary DataFrame to the main DataFrame
        df = pd.concat([df, temp_df], ignore_index=True)

# replace the 'Sentiment' column with 'positive', 'neutral', or 'negative'
df.loc[df['Sentiment'].str.contains('positive|positiv', case=False), 'Sentiment'] = 'positive'
df.loc[df['Sentiment'].str.contains('neutral', case=False), 'Sentiment'] = 'neutral'
df.loc[df['Sentiment'].str.contains('negative|negativ', case=False), 'Sentiment'] = 'negative'

# remove all rows that do not have 'positive', 'neutral', or 'negative' in the 'Sentiment' column
df = df[df['Sentiment'].isin(['positive', 'neutral', 'negative'])]


#Splitting the dataset into train and test
train = df.sample(frac = 0.8, random_state = 25)
test = df.drop(train.index)

print(train.head)
print(test.head)


# Checking the shape

print(train.shape, test.shape)

import spacy
nlp=spacy.load("de_dep_news_trf")


#Creating tuples
train['tuples'] = train.apply(lambda row: (row['Text'],row['Sentiment']), axis=1)
train = train['tuples'].tolist()
test['tuples'] = test.apply(lambda row: (row['Text'],row['Sentiment']), axis=1)
test = test['tuples'].tolist()
train[0]





# User function for converting the train and test dataset into spaCy document
def document(data):
#Creating empty list called "text"
  text = []
  for doc, label in nlp.pipe(data, as_tuples = True):
    if (label=='positive'):
      doc.cats['positive'] = 1
      doc.cats['negative'] = 0
      doc.cats['neutral']  = 0
    elif (label=='negative'):
      doc.cats['positive'] = 0
      doc.cats['negative'] = 1
      doc.cats['neutral']  = 0
    else:
      doc.cats['positive'] = 0
      doc.cats['negative'] = 0
      doc.cats['neutral']  = 1
#Adding the doc into the list 'text'
      text.append(doc)
  return(text)

  # Calculate the time for converting into binary document for train dataset

start_time = datetime.now()

#passing the train dataset into function 'document'
train_docs = document(train)

#Creating binary document using DocBin function in spaCy
doc_bin = DocBin(docs = train_docs)

#Saving the binary document as train.spacy
doc_bin.to_disk("train.spacy")
end_time = datetime.now()

#Printing the time duration for train dataset
print('Duration: {}'.format(end_time - start_time))

# Calculate the time for converting into binary document for test dataset

start_time = datetime.now()

#passing the test dataset into function 'document'
test_docs = document(test)
doc_bin = DocBin(docs = test_docs)
doc_bin.to_disk("valid.spacy")
end_time = datetime.now()

#Printing the time duration for test dataset
print('Duration: {}'.format(end_time - start_time))