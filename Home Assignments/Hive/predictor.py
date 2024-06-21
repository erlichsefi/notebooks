

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
class UrlRemovealTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, constant=1.0):
        self.constant = constant
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
            
        def remove_url(text):
            import re
            # Regular expression pattern to match URLs
            url_pattern = r'(https?://\S+|www\.\S+)'
            # Find all URLs in the text

            # Replace all URLs in the text with <URL>
            return re.sub(url_pattern, '', text)
        
        def get_urls(text):
            import re
            # Regular expression pattern to match URLs
            url_pattern = r'(https?://\S+|www\.\S+)'
            return re.findall(url_pattern, text)
        
        X['urls'] = X['Text'].apply(get_urls)
        X['Text'] = X['Text'].apply(remove_url)
        return X

class PunctuationRemovalTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, constant=1.0):
        self.constant = constant
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
            
        def remove_punctuation(text):
            import re
            return re.sub(r'[.,:"\'!?()]+', '',text).replace("\n"," ").replace("\t"," ")
        
        X['Text'] = X['Text'].apply(remove_punctuation)
        return X


class RetweetRemovalTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, constant=1.0):
        self.constant = constant
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
         
        def extract_retweete_user(text):
            import re
            mention_pattern = r'RT @\w+'

            # Find all words starting with @ in the text
            response =  re.findall(mention_pattern, text)

            if len(response) == 1:
                return response[0]
            return ""
        
        X['retweet'] = X['Text'].apply(extract_retweete_user)
        X['Text'] = X[['Text','retweet']].apply(lambda x:x['Text'].replace(x['retweet'],""),axis=1)
        return X


class TagsTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, constant=1.0):
        self.constant = constant
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
         
        def extract_taged_user(text):
            import re
            mention_pattern = r'@\w+'

            # Find all words starting with @ in the text
            mentions = re.findall(mention_pattern, text)

            replaced_text = re.sub(mention_pattern, '', text)

            return mentions,replaced_text
        
        X['response'] = X['Text'].apply(extract_taged_user)
        X['mentions'] = X['response'].apply(lambda x:x[0])
        X['Text'] = X['response'].apply(lambda x:x[1])
        return X.drop(columns=['response'])

class StopWordsTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, constant=1.0):
        self.constant = constant
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
 
        def get_stopwords() -> set[str]:
            """
            Returns a list of stopwords in hebrew based on pre-loaded file along with added words that are unique to this corpus.
            """
            stop_path = "Home Assignments/Hive/heb_stopwords.txt"
            with open(stop_path, encoding="utf-8") as in_file:
                lines = in_file.readlines()
                res = [l.strip() for l in lines]
                res.extend([",", ".",'-','–',"\"","\t","ה", "ל", "ב", "ו", "ש", "מ", "של", "על", "את", "או",
                                "הוא", "לא", "אם", "כל", "כ", "עם", "הם", "היא", "הן"])
            return set(res)

        def get_non_stop_words(df):
            stopwords = get_stopwords()
            return ' '.join([w for w in df.split() if w not in stopwords])
        
        X['Text'] = X['Text'].apply(get_non_stop_words)
        return X 

class TextToEmbedding(BaseEstimator, TransformerMixin):
    def __init__(self, constant=1.0):
        self.tfidf = TfidfTransformer()
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
 
        def get_stopwords() -> set[str]:
            """
            Returns a list of stopwords in hebrew based on pre-loaded file along with added words that are unique to this corpus.
            """
            stop_path = "Home Assignments/Hive/heb_stopwords.txt"
            with open(stop_path, encoding="utf-8") as in_file:
                lines = in_file.readlines()
                res = [l.strip() for l in lines]
                res.extend([",", ".",'-','–',"\"","\t","ה", "ל", "ב", "ו", "ש", "מ", "של", "על", "את", "או",
                                "הוא", "לא", "אם", "כל", "כ", "עם", "הם", "היא", "הן"])
            return set(res)

        def get_non_stop_words(df):
            stopwords = get_stopwords()
            return ' '.join([w for w in df.split() if w not in stopwords])
        
        X['Text'] = X['Text'].apply(get_non_stop_words)
        return X 
    
# Create pipeline
text_pipeline = Pipeline([
    ("url", UrlRemovealTransformer()),
    ("punctuation",PunctuationRemovalTransformer()),
    ("retweet",RetweetRemovalTransformer()),
    ("tags",TagsTransformer()),
    ("stopWords",StopWordsTransformer()),
   
])

ml_pipeline = Pipeline([
    ("counter",CountVectorizer(ngram_range=(2, 2))),
    ("tf_idf",TfidfTransformer()),
    ("classifier",RandomForestClassifier())
])

X_train = pd.read_csv("Home Assignments/Hive/data-train.csv")
X_prod = pd.read_csv("Home Assignments/Hive/data-1716191272369.csv").rename(columns={"text":"Text"})
# Split data into training and testing sets
# Fit pipeline on training data
X_processed = text_pipeline.fit_transform(X_train)
ml_pipeline.fit(X_processed.Text,X_processed.Topic)
y_predication = ml_pipeline.predict(X_processed.Text)

X_prod = text_pipeline.transform(X_prod)
X_prod['prediction']  = ml_pipeline.predict(X_prod.Text)
X_prod.to_csv("predications.csv")
