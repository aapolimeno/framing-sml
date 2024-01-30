import pandas as pd 
#from transformers import AutoTokenizer, AutoModel
#import torch
import spacy
from spacy.lang.nl import Dutch
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
#from gensim.models import Word2Vec
#import nltk
#import pdb

#nlp = spacy.load("nl_core_news_md")

# Run in command line if embeddings model isn't on your computer yet: 
# python -m spacy download nl_core_news_md

#nltk.download('stopwords')
#nltk.download('punkt')

def tfidf_vectorizer(documents): 
    """
    
    """
    
    # Preprocess and filter stop words

    # NLTK Dutch stop words
    stop_words = set(stopwords.words('dutch'))
    
    
    preprocessed_documents = []
    

    for doc in documents:
        tokens = word_tokenize(doc.lower())  # Tokenize and convert to lowercase
        filtered_tokens = [word for word in tokens if word not in stop_words]
        #print(filtered_tokens[:10])
        preprocessed_documents.append(" ".join(filtered_tokens))
    
    # Create an instance of CountVectorizer
    tfidf_vectorizer = TfidfVectorizer()
    # Fit the vectorizer on the documents, transform them into a Bag of Words
    tfidf_repr = tfidf_vectorizer.fit_transform(preprocessed_documents)
    
    # Save the words in the vocabulary
    feature_names = tfidf_vectorizer.get_feature_names()
    # Save the vectors as a DataFrame
    tfidf_array = tfidf_repr.toarray()
    
    tfidf = [entry for entry in tfidf_array]
    
    return tfidf



def get_sentence_embeddings(documents, input_type):
    model_name = "sentence-transformers/bert-base-nli-mean-tokens"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    encoded_documents = []

    for doc in documents:
        if input_type == "texts" or input_type == 'paragraphs':
            # Tokenize the document into sentences
            sentences = sent_tokenize(doc)
        elif input_type == "titles":
            sentences = [doc]
        else:
            raise ValueError("Invalid input_type. Please specify 'document' or 'sentence'.")
        
        sentence_embeddings = []

        for sentence in sentences:
            # Tokenize the sentence
            tokens = tokenizer(sentence, padding=True, return_tensors="pt")

            with torch.no_grad():
                model_output = model(**tokens)
                token_embeddings = model_output.last_hidden_state  # Get token embeddings

                # Calculate mean pooling over token embeddings
                sentence_embedding = torch.mean(token_embeddings, dim=1)

                sentence_embeddings.append(sentence_embedding.squeeze().numpy())

        # Calculate the document-level embedding by averaging sentence embeddings
        if sentence_embeddings:
            document_embedding = np.mean(sentence_embeddings, axis=0)
            encoded_documents.append(document_embedding)
        else:
            # Handle empty documents if necessary
            encoded_documents.append(np.zeros(768))  # You can adjust the size as needed

    # Convert the list of embeddings to a NumPy array
    sentence_embeddings = np.array(encoded_documents)

    return sentence_embeddings







def get_repr(df, method, input_type):
    
    
    if input_type == 'texts' or input_type == 'paragraphs': 
        texts = df['text'].tolist()
       
    elif input_type == 'titles': 
        texts = df['title'].tolist()
        
    
    if method == 'tfidf': 
        tfidf_vectors = tfidf_vectorizer(texts)
        df[f"{method}"] = tfidf_vectors

    if method == 'sentence_embeddings': 
        vectors = get_sentence_embeddings(texts, input_type).tolist()
        df[f"{method}"] = vectors

    if method == 'word_embeddings': 
        vectors = get_word_embeddings(texts)
        df[f'{method}'] = vectors

    if method == 'custom_embeddings': 
        vectors = get_custom_embeddings(texts, input_type)
        df[f'{method}'] = vectors
 
    
    return df


