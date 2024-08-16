if __name__ == "__main__":

    ### Topic model ###
    # This code is adapted from Bonacchi and Kryzanska (2021). The original script is found here: https://github.com/IARHeritages/HeritageTribalism_BigData/blob/main/codes/mongo/05_02_Topic_assignment.js . I also used this guide https://towardsdatascience.com/topic-modelling-in-python-with-nltk-and-gensim-4ef03213cd21 
    
    ## This code creates the LDA topic model for the unique documents in user specified datasets, for topic numbers n = 2:16, or 2:31
    ## It is designed to handle a json file of Text data, that also has a Code column. There is one type of code for this template: type a) TT.1
    ## it calculates the coherence scores for the topic and creates and intertopic-distance visualisation for the model with the highest coherence score, with the LDAvis.
    ## it then also assignes the dominant topic and topic scores to all Text documents utilised, and intergrates these with a master file in a Mongodb database
    ## thee user is asked to specify the following:
                #the working directory
                #the MongoDB connection string
                #the database name
                #the collection name
                #the maximum number of topics
                #any additional stop words, separated by commas
                #the scale of n-gram used for documents of different lengths, such as for posts, comments, or video transcripts.
                #the master collection name that the LDA results will be combined with. this requires a Code column that matches the values used in the original collection
                #the new collection name to save the results:

    ## Make sure these are installed 
    # pip install pymongo
    # pip install Scipy
    # pip install numpy
    # pip install nltk
    # pip install pyLDAvis
    # pip install gensim

    # Import relevant libraries:
    import pymongo
    import time
    import gensim
    import os
    import csv
    import re
    import operator
    import warnings
    import numpy as np
    import json
    from gensim.models import CoherenceModel, LdaModel, LsiModel, HdpModel
    from gensim.corpora import Dictionary
    from gensim.parsing.preprocessing import strip_punctuation
    from pprint import pprint
    from gensim.corpora.mmcorpus import MmCorpus
    from gensim.models import ldamulticore
    from gensim import models
    from nltk.corpus import stopwords
    from gensim.models.phrases import Phraser
    import pyLDAvis.gensim
    import gc
    import logging
    from nltk.corpus import stopwords
    import nltk
    nltk.download('stopwords')
    nltk.download('wordnet')
    from nltk.tokenize import word_tokenize
    nltk.download('punkt')
    from nltk.stem.wordnet import WordNetLemmatizer
    from gensim.models import Phrases
    from gensim.corpora import MmCorpus
    from gensim.test.utils import get_tmpfile 
    from collections import OrderedDict
    from pymongo import MongoClient
    from bson import ObjectId

    # Set up logging to console to monitor the progress of the topic model:
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    # set a format which is simpler for console use
    formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    # add the handler to the root logger
    logging.getLogger('').addHandler(console)
    logger = logging.getLogger(__name__)

    ### Set up workspace

    # Check if the script is running in an interactive environment (like Google Colab)
    def in_notebook():
        try:
            from IPython import get_ipython
            return get_ipython() is not None
        except ImportError:
            return False

    if in_notebook():
        from google.colab import drive
        drive.mount('/content/drive')
    else:
        print("Not running in an interactive environment. Skipping drive.mount.")

    # Ask the user to specify the working directory
    working_directory = input("Please specify the working directory: ")
    os.chdir(working_directory)

    # Ask the user to specify the Mongo address and connect to the MongoDB client
    mongo_address = input("Please specify the MongoDB connection string: ")
    client = MongoClient(mongo_address)

    # Ask the user to specify the database name
    database_name = input("Please specify the database name: ")
    db = client[database_name]

    # Ask the user to specify the collection name
    collection_name = input("Please specify the collection name: ")
    Textcollection = db[collection_name]

    # Load the documents from the collection into a Python list
    data_texts = list(Textcollection.find())
    
    # Function to convert id to str and remove if required
    def convert_id(entry):
        if isinstance(entry, dict):
            for key, value in list(entry.items()):
                if isinstance(value, ObjectId):
                    # Convert ObjectId to str
                    entry[key] = str(value)
                elif isinstance(value, dict) or isinstance(value, list):
                    # Recursively call convert_id to handle nested dictionaries and lists
                    convert_id(value)
        elif isinstance(entry, list):
            for item in entry:
                # Recursively call convert_id to handle nested dictionaries and lists
                convert_id(item)
    
    # Assuming data_texts is your data loaded from MongoDB
    convert_id(data_texts)
    
    # Save the data to a JSON file
    with open('Social_texts.json', 'w') as file:
         json.dump(data_texts, file, indent=4)  
    
    # Define the paths to input:
    file_path = 'Social_texts.json'

    #Creating folders for outputs
    outputs_folder = os.path.join(working_directory, 'outputs')
    os.makedirs(outputs_folder, exist_ok=True)
    topic_models_folder = os.path.join(outputs_folder, 'topic_models')
    os.makedirs(topic_models_folder, exist_ok=True)

    # Define the paths to outputs
    path2corpus= "outputs/topic_models/corpus.mm"
    path2dictionary= "outputs/topic_models/dictionary.mm"
    path2model= "outputs/topic_models/models_.mm"
    path2coherence = "outputs/03_01_01_coherenceScores.csv" # Path to model coherence scores
    path2html = "outputs/03_01_02_topic_model.html" # Path to the best model visualisation in html

    # Define language to use for the model, and the threshold for the number of topics
    language='english'

    # Ask the user to specify the max_topics
    num_docs = len(data_texts)
    recommended_topics = 16 if num_docs <= 2000 else 31
    max_topics = int(input(f"Please specify the maximum number of topics (recommended {recommended_topics} for {num_docs} documents): "))

    # Load the JSON file containing all texts
    with open(file_path, 'r', encoding='utf8') as file:
        data = json.load(file)

    # Extracting texts and ignoring the codes and dates
    texts = [item['Text'] for item in data if 'Text' in item]
    # Getting codes as well
    codes = [item['Code'] for item in data if 'Code' in item]

    # Removing links and usernames if they appear in the text/Might edit to keep links to establish linking behaviour within topics:
    i=0
    j=len(texts)
    while(i<j):
        texts[i] = re.sub(r'http\S+', '', texts[i])
        texts[i] = re.sub(r'@\S+', '', texts[i])
        texts[i] = re.sub(r'[^\w\s]', '', texts[i])  # Remove punctuation
        i=i+1    

    # Tokenize the text into words
    texts = [word_tokenize(text.lower()) for text in texts]  # Converts to lowercase and tokenizes

    # Import and define stopwords:
    nltk.download('stopwords')
    stops = set(stopwords.words('english'))

    # Ask the user to provide additional stopwords, separated by commas
    additional_stopwords = input("Please provide any additional stop words, separated by commas: ")
    new_stops = set(additional_stopwords.split(','))

    # Get rid of English stopwords and user-defined stopwords:
    texts = [[word for word in text if word not in stops] for text in texts]
    texts = [[word for word in text if word not in new_stops] for text in texts]

    # Lemmatize all the words in the document:
    nltk.download('wordnet')
    lemmatizer = WordNetLemmatizer()
    texts = [[lemmatizer.lemmatize(token) for token in text] for text in texts]

    # Create n-grams based on user input
    def create_ngrams(texts, max_n=2, min_count=20):
        """Create n-grams for the provided texts up to max_n."""
        for n in range(2, max_n + 1):
            # Create n-gram model
            ngram_model = Phrases(texts, min_count=min_count)
            ngram_phraser = Phraser(ngram_model)
            
            # Apply n-grams to texts
            for idx in range(len(texts)):
                new_tokens = [token for token in ngram_phraser[texts[idx]] if '_' in token]
                texts[idx].extend(new_tokens)

        return texts

    # User input for n-grams
    print("Please select: bigrams for shorter Text snippets such as comments, trigrams for social media posts and longer blogs, or N-Grams up to 10 for transcripts or lengthy documents. This model defaults to bigrams only. Please specify:")
    user_input = input("Enter 'bigrams', 'trigrams', or a number (2 to 10) for n-grams: ").strip().lower()

    if user_input == 'bigrams':
        max_n = 2
    elif user_input == 'trigrams':
        max_n = 3
    else:
        try:
            max_n = int(user_input)
            if max_n < 2 or max_n > 10:
                raise ValueError
        except ValueError:
            print("Invalid input. Defaulting to bigrams.")
            max_n = 2

    # User input for min_count
    num_docs = len(texts)
    suggested_min_count = max(5, num_docs // 200)
    print(f"There are {num_docs} documents. It is suggested to use a min_count of at least {suggested_min_count}.")
    min_count_input = input(f"Enter a min_count for phrases (suggested: {suggested_min_count}): ").strip()

    try:
        min_count = int(min_count_input)
    except ValueError:
        print(f"Invalid input. Defaulting to suggested min_count: {suggested_min_count}.")
        min_count = suggested_min_count

    # Create n-grams
    texts = create_ngrams(texts, max_n=max_n, min_count=min_count)
         
                         
                
    # Make dictionary and the corpus
    train_texts = texts 
    dictionary = Dictionary(train_texts)
    corpus = [dictionary.doc2bow(text) for text in train_texts]

    ### Save corpus and dictionary:  
    MmCorpus.serialize(path2corpus, corpus)
    mm = MmCorpus(path2corpus)
    dictionary.save_as_text(path2dictionary)
    dictionary = Dictionary.load_from_text(path2dictionary)

    # Set up the list to hold coherence values for each topic:
    c_v = []
    # Loop over to create models with 2 to 30 topics, and calculate coherence scores for it:
    for num_topics in range(2, max_topics):
        print(num_topics)
        lm = models.LdaMulticore(corpus=mm, num_topics=num_topics, id2word=dictionary, chunksize=9000, passes=100, eval_every=1, iterations=500, workers=4) # Create a model for num_topics topics
        print("Calculating coherence score...")
        cm = CoherenceModel(model=lm, texts=train_texts, dictionary=dictionary, coherence='c_v') # Calculate the coherence score for the topics
        print("Saving model...")
        lm.save(path2model+str(num_topics)) # Save the model
        lm.clear() # Clear the data from the model
        del lm # Delete the model
        gc.collect() # Clears data from the workspace to free up memory
        c_v.append(cm.get_coherence()) # Append the coherence score to the list

    # Save the coherence scores to the file:    
    with open(path2coherence, 'a') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow(["n_topics","coherence_score"])
        i=2
        for score in c_v:
            print(i)
            writer.writerow([i,score])
            i=i+1

    # Get the best topic model and construct the visualisation

    n=c_v.index(max(c_v))+2 # Get the number of topics with the highest coherence score
    lm = LdaModel.load(path2model+str(n)) # Load the number of topics with the highest coherence score into the workspace
    tm = pyLDAvis.gensim.prepare(lm, mm, dictionary) # Prepare the visualisation
    pyLDAvis.save_html(tm, path2html+str(n)+'.html') # Save the visualisation

    ### Assign topics to texts, using their Codes, along with Probability score:
    # Reorder topic: 

    # Change the topics order to be consistent with the to be consistent with the pyLDAvis topic model (ordered from the most
    # frequent one to the least frequent one) and assign dominant topic to each document:

    # Get the topic order
    to=tm.topic_order

    # set up writing to a file
    # List to hold the processed data for each document
    documents_data = []

    # Loop over all documents in the corpus, and assign topic and probabilities to each
    for i in range(len(corpus)):
        # Get topic probabilities for the document
        topics = lm.get_document_topics(corpus[i])
        # Reorder topics according to pyLDAvis numbering and convert topic probabilities to float
        topics = [["Corpus_Topic " + str(to.index(topic[0] + 1)), float(topic[1])] for topic in topics]
        topics = sorted(topics, key=lambda x: x[0])
        topics_dict = dict(topics)

        # Get dominant topic and its value for the documents
        topics_dict['Corpus_dominant_topic'] = max(topics_dict, key=topics_dict.get)
        topics_dict['Corpus_dominant_value'] = float(topics_dict[topics_dict['Corpus_dominant_topic']])
        topics_dict["Code"] = codes[i]
        topics_dict["text"] = texts[i]

        # Add the document's data to the list
        documents_data.append(topics_dict)

    # Write the data to a JSON file
    with open('DataTextTopics.json', 'w', encoding='utf-8') as f:
        json.dump(documents_data, f, ensure_ascii=False, indent=4)
    
    # Merge function
    def merge_Social_data(Social_data, documents_data):
        for entry in Social_data:
            # Merge with form A topics
            code_a = entry.get('Code', '')
            if code_a in documents_data:
                entry.update(documents_data[code_a])

    # Ask the user to specify the master collection name
    master_collection_name = input("Please specify the collection name that you would like to merge the LDA results with. Note: collection must have a matching 'Code' column: ")
    collection = db[master_collection_name]

    # Load the documents from the collection into a Python list
    Social_data = list(collection.find())
    
    # Social_data is already loaded from MongoDB
    merge_Social_data(Social_data, documents_data)
    
    # Function to convert ObjectId to str and remove if required
    def convert_id(entry):
        if isinstance(entry, dict):
            for key, value in list(entry.items()):
                if isinstance(value, ObjectId):
                    # Convert _id to str
                    entry[key] = str(value)
                elif isinstance(value, dict) or isinstance(value, list):
                    # Recursively call convert_id to handle nested dictionaries and lists
                    convert_id(value)
        elif isinstance(entry, list):
            for item in entry:
                # Recursively call convert_id to handle nested dictionaries and lists
                convert_id(item)

    # Assuming Social_data is your data loaded from MongoDB
    convert_id(Social_data) 

    # Save the merged data to a JSON file
    with open('Merged_Social_Data.json', 'w') as file:
         json.dump(Social_data, file, indent=4)

    # Ask the user to specify the new collection name
    new_collection_name = input("Please specify the new collection name to save the merged results: ")

    # Create a new collection (if it doesn't exist)
    new_collection = db[new_collection_name]

    # Insert the merged data
    new_collection.insert_many(Social_data)
