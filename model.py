import re
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
from wordcloud import WordCloud
import matplotlib.pyplot as plt

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer

from sklearn.svm import LinearSVC
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.linear_model import LogisticRegression
from xgboost.sklearn import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import f1_score, accuracy_score, plot_confusion_matrix

from textblob import TextBlob


df = pd.read_csv('cleanedTweets.csv')

# df = df.drop(['ids', 'date', 'flag', 'user'], axis = 'columns')
# df = df.drop_duplicates(keep = 'first')
# print("The size of dataset after removing duplicates is : ", len(df))

# df['sentiment'] = df['sentiment'].replace(4,1)

# pltbar = df.groupby('sentiment').count().plot(kind='bar')
# pltbar.set_xticklabels(['Negative','Positive'], rotation=0)


# #Calculating percentages of negative and positive tweets
# positiveTweets = 100*len(df.loc[df['sentiment']==1,'sentiment'])/len(df['sentiment'])
# negativeTweets = 100*len(df.loc[df['sentiment']==0,'sentiment'])/len(df['sentiment'])
# print(f'Negative Sentiment Tweets Percentage is {negativeTweets}%')
# print(f'Positive Sentiment Tweets Percentage is {positiveTweets}%')

# # Storing data in lists.
text, sentiment = list(df['text']), list(df['sentiment'])


# stop_words = stopwords.words('english')
# def cleanTweet(text):
#     cleanedTweets = []
    
#     for tweets in text:
#         #Converting text to lowercase
#         tweets = tweets.lower()

#         #Removing newline breaks
#         tweets = re.sub(r'\n', '', tweets)
        
#         #Removing URLs
#         tweets = re.sub(r"(?:\|http?\://|https?\://|www)\S+", "", tweets)
        
#         #Removing @usernames
#         tweets = re.sub('@[^\s]+','', tweets)
        
#         #Removing punctuations, numbers & special characters
#         tweets = re.sub("[^a-zA-Z]", " ", tweets)
        
#         #Removing emojis
#         tweets = re.compile("["
#                    u"U0001F600-U0001F64F"  # emoticons
#                    u"U0001F300-U0001F5FF"  # symbols & pictographs
#                    u"U0001F680-U0001F6FF"  # transport & map symbols
#                    u"U0001F1E0-U0001F1FF"  # flags (iOS)
#                    u"U00002702-U000027B0"
#                    u"U000024C2-U0001F251"
#                                "]+", flags=re.UNICODE).sub(r'', tweets)
        
#         finaltweet = ''
#         #Removing short words(with length less than 3) & stop words
#         temp = tweets.split()
#         stop_words = stopwords.words('english')
#         stop_words = stop_words + ['hi', 'im', 'amp', 'quot']
#         textwithoutstopwords = [word for word in temp if not word in stop_words and len(word)>2]
        
#         #Lemmatization
#         lem = WordNetLemmatizer()
#         lemmatizedText = [lem.lemmatize(y) for y in textwithoutstopwords]
#         finaltweet = ' '.join(lemmatizedText)
#         cleanedTweets.append(finaltweet)
#     return cleanedTweets


cleanedTweets = df;
df['Cleaned Tweets'] = cleanedTweets


# X_train,X_test,Y_train, Y_test = train_test_split(text, sentiment, test_size=0.1)
# print("Size of training set : ", len(X_train))
# print("Size of test set : ", len(X_test))

# vectorizer= TfidfVectorizer(ngram_range=(1,2), max_features=500000)
# X_train = vectorizer.fit_transform(X_train)
# X_test = vectorizer.transform(X_test)


file = open('lr.pickle','wb')



def loadModel():    
    #Load the vectoriser.
    file = open('vectorizer.pickle', 'rb')
    vectorizer = pickle.load(file)
    file.close()
    #Load the model
    file = open('svm.pickle', 'rb')
    model = pickle.load(file)
    file.close()
    return vectorizer, model



def predict(vectorizer, model, text):
    #Predict the sentiment
    inputdata = vectorizer.transform(df.text)
    sentiment = model.predict(inputdata)
    
    # Make a list of text with sentiment.
    input = []
    for text, pred in zip(text, sentiment):
        input.append((text,pred))
        
    # Convert the list into a Pandas DataFrame.
    df1 = pd.DataFrame(input, columns = ['Tweets','Sentiment'])
    df1 = df1.replace([0,1], ["Negative","Positive"])
    return df1


#Loading the saved model
vectorizer, model = loadModel()


text = ["This was the worst trip I have ever had in my life.",
            "I like pizza",
            "Heppi",
            "Cringe",
            "He wished to sleep, but he knew he would not be able to and that most happy thoughts came to him in bed",
            "lmao u got it", 
            "bit stuffing"]

df = predict(vectorizer, model, text)
print(df)

# take inputs using flask 