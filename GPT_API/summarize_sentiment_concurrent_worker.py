import jsonlines
import openai
from dotenv import load_dotenv
import os
import json
import nltk
from collections import Counter
import matplotlib.pyplot as plt
import time
import concurrent.futures

load_dotenv()

# Authentication
openai.api_key = os.getenv("OPENAI_KEY", "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")

def read_json_file(file_name):
    with open(file_name, 'r') as file:
        data = json.load(file)
    return data

def save_dict_to_json(dictionary, speech_id):
    file_name = f"../../data/speeches_sentiment/{speech_id}_data.json"
    with open(file_name, 'w') as json_file:
        json.dump(dictionary, json_file)

legislatur = 20
alleReden = []

with jsonlines.open(f'../../data/speeches_{legislatur}.jsonl') as f:
    for line in f.iter():
        alleReden.append(line)
alleReden.sort(key=lambda x: x['date'])

timeout_seconds = 5
retry_delay_seconds = 1

def call_api_with_retry(sentence):
    retry = True
    while retry:
        try:
            speech_string = f'''Hier ist ein Satz einer politischen Rede: {sentence}
            Bewerte das Sentiment (neutral, positiv, negativ).
            "Sentiment:"'''
            chat_history = [{"role": "user", "content": speech_string}]
            response = openai.ChatCompletion.create(
                model='gpt-3.5-turbo-16k-0613',
                messages=chat_history
            )
            message = response.choices[0]['message']
            print("{}: {}".format(message['role'], message['content']))
            retry = False
            return message['content']
        except Exception as e:
            if 'overloaded' in str(e) or 'not ready' in str(e):
                print("Server is overloaded or not ready yet. Retrying after 1 second.")
                time.sleep(retry_delay_seconds)
            else:
                print(f"An error occurred: {e}")
                raise

nltk.download('punkt')
from nltk.tokenize import sent_tokenize

processed_speeches_file = "processed_speeches_sentiment.txt"
processing_speeches_file = "processing_speeches_sentiment.txt"

def process_speech(speech):
    with open(processed_speeches_file, 'r') as f:
        processed_ids = f.read().splitlines()
    with open(processing_speeches_file, 'r') as f:
        processing_ids = f.read().splitlines()

    if speech["id"] not in processed_ids and speech["id"] not in processing_ids:
        # Mark this speech as being processed
        with open(processing_speeches_file, 'a') as f:
            f.write(speech["id"] + "\n")
        
        sentences = sent_tokenize(speech["text"])
        speech_sentiments = {"id": speech["id"], "sentiments": []}
        for sentence in sentences: 
            sentiment = call_api_with_retry(sentence)
            speech_sentiments["sentiments"].append({"sentence": sentence, "sentiment": sentiment})

        # Move this speech from processing to processed
        with open(processing_speeches_file, 'r') as f:
            lines = f.readlines()
        with open(processing_speeches_file, 'w') as f:
            for line in lines:
                if line.strip("\n") != speech["id"]:
                    f.write(line)
        with open(processed_speeches_file, 'a') as f:
            f.write(speech["id"] + "\n")

        save_dict_to_json(speech_sentiments, speech["id"])
    return speech["id"]

if os.path.exists(processed_speeches_file):
    with open(processed_speeches_file, 'r') as f:
        processed_speeches = f.read().splitlines()
else:
    processed_speeches = []

# Ensure the processing speeches file exists
if not os.path.exists(processing_speeches_file):
    with open(processing_speeches_file, 'w') as f:
        pass

num_workers = 15  # Adjust as needed
with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
    for result in executor.map(process_speech, alleReden):
        print(f"Processed speech with ID: {result}")
