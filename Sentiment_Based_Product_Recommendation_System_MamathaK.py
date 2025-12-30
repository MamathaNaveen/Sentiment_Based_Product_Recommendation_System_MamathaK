#!/usr/bin/env python
# coding: utf-8

# ###### Ran from jupyter notebook
# //from google.colab import drive
# //drive.mount('/content/gdrive')
# 

# # **Problem Statment**
# 
# The e-commerce business is quite popular today. Here, you do not need to take orders by going to each customer. A company launches its website to sell the items to the end consumer, and customers can order the products that they require from the same website. Famous examples of such e-commerce companies are Amazon, Flipkart, Myntra, Paytm and Snapdeal.
# 
# Suppose you are working as a Machine Learning Engineer in an e-commerce company named 'Ebuss'. Ebuss has captured a huge market share in many fields, and it sells the products in various categories such as household essentials, books, personal care products, medicines, cosmetic items, beauty products, electrical appliances, kitchen and dining products and health care products.
# 
# With the advancement in technology, it is imperative for Ebuss to grow quickly in the e-commerce market to become a major leader in the market because it has to compete with the likes of Amazon, Flipkart, etc., which are already market leaders.
# 
# As a senior ML Engineer, you are asked to build a model that will improve the recommendations given to the users given their past reviews and ratings.
# 
# In order to do this, you planned to build a sentiment-based product recommendation system, which includes the following tasks.
# 
# 1. Data sourcing and sentiment analysis
# 2. Building a recommendation system
# 3. Improving the recommendations using the sentiment analysis model
# 4. Deploying the end-to-end project with a user interface
# 5. Steps involved in the project
#     - Exploratory data analysis
#     - Data cleaning
#     - Text preprocessing
#     - Feature extraction
#     - Training the text classification model
#     - Creating a recommedation systems (User based and Item Based choose the bestone)
#     - Evaluating the model and recommedation system using the Test data
#     - Create flask application
#     - Deploy the application to heroku platform
# 
# 

# ### **Importing Necessary Libraries**

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
#%matplotlib inline

### NLP
import re,nltk,spacy,en_core_web_sm
from nltk.corpus import stopwords
nlp=en_core_web_sm.load()

##PLotly
from plotly.offline import plot

##Recommendation System
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.metrics import classification_report,roc_auc_score,confusion_matrix,f1_score,precision_score,accuracy_score
from sklearn.metrics import pairwise_distances
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.model_selection import train_test_split,GridSearchCV,RandomizedSearchCV
import xgboost as xgb
from xgboost import XGBClassifier


# ### **Data Extraction**

# In[4]:


master_df=pd.read_csv("dataset/sample30.csv")
df_class=master_df.copy()
df_rec=master_df.copy()


# In[5]:


pd.set_option('display.max_columns',200)


# In[6]:


df_class.head()


# In[7]:


df_class.info()


# In[8]:


##Dropping reviews_userCity and reviews_userProvince
df_class.drop(['reviews_userCity','reviews_userProvince','reviews_date'],axis=1,inplace=True)


# ### **Exploratory Data Analysis AND Data Cleaning**

# In[9]:


print("Sentiment Count")
df_class.user_sentiment.value_counts()


# In[10]:


df_class.reviews_didPurchase.value_counts()
print("Found Nan Values , filling it 'No Data' ")
df_class['reviews_didPurchase'].fillna('No Data', inplace=True)
df_class.reviews_didPurchase.value_counts()


# In[11]:


df_class.reviews_doRecommend.value_counts()


# In[12]:


##No Unique Users
df_class.reviews_username.nunique()


# In[13]:


df_class.reviews_rating.value_counts()


# In[14]:


## No null values
df_class.reviews_text.info(verbose=True)


# In[15]:


## removing reviews with no username for genuine reviews
df_class.dropna(subset=['reviews_username'],inplace=True)


# In[16]:


df_class.info(verbose=True)


# In[17]:


## Final Data review
#data overivew
print('rows: ', df_class.shape[0])
print('columns: ', df_class.shape[1])
print('\nfeatures: ', df_class.columns.to_list())
print('\nmissing vlues: ', df_class.isnull().values.sum())
print('\nUnique values: \n', df_class.nunique())


# In[18]:


#sns.barplot(df_class.user_sentiment.value_counts())
sns.countplot(x='user_sentiment', data= df_class, palette="Set2")


# ***** Analysis on Sentiment: More positive reviews seen on products, good sign for platform that customers are happy.

# ***** Rating wise also highest positive rating seen by customers another indication that products are been well received

# In[19]:


df_class.columns
#df_class.reviews_didPurchase.value_counts()


# In[20]:


sns.countplot(x='reviews_rating',data=df_class,palette='Set2')
plt.show()
sns.countplot(x='reviews_didPurchase',data=df_class,palette='Set2')
plt.show()


# ###### Analysis from above graphs:
# Since reviews_didPurchase count is very low, hence i will consider reviews of other columns as well who have not purchased products for recommendation. In future when the review_didPurchase count gets increased to reasonable sum i will ignore reviews from users who have not purchased the product.

# In[21]:


sns.histplot(data=df_class,x='reviews_rating',hue='reviews_didPurchase'),plt.show()
sns.histplot(data=df_class,x='reviews_rating',hue='user_sentiment',palette='Set2'),plt.show()


# In[22]:


print("Count of rating greater than 3: ", df_class[df_class['reviews_rating'] > 3].reviews_rating.count())
print("Count of positive user_sentiment: ", df_class[df_class['user_sentiment'] == 'Positive'].user_sentiment.count())

print("Rating greater than 3 is considered positive review")


# ###### In sentiment based analysis rating 3 is considered nuetral. but since this is new e commerce platform I dont want e-comm platform to suggest average products hence i will ignore products with rating 3 along with 1 and 2. There are 300 products being missed from suggestion with rating 3 and having positive review is acceptable for me.  

# ###### In the next stages i will use NLP to lemmatize 'reviews_text' column to get hidden emotion from users on product reviews and combine it with rating
# Weight Sum ((0.x)*Combine Rating + (1-0.x)*Text Sentiment to generate final sentiment score for analysis.

# In[23]:


df_class.reviews_text.info()


# In[24]:


## Convert object to string type
df_class['reviews_text']=df_class['reviews_text'].astype(str)
type(df_class['reviews_text'].iloc[0])


# In[25]:


df_class.isna().sum()


# user_sentiment has one column with NA will remove that row!!

# In[26]:


df_class=df_class[~df_class['user_sentiment'].isna()]


# In[27]:


##checking again is any NAN values present in user_sentiment and reviews_text
df_class.isna().sum()


# ### **Feature Extraction**

# #### **Import requried modules and remove stopwords**

# In[28]:


#Common functions for cleaning the text data
import nltk
#from nltk.stem import LancasterStemmer
from nltk.stem import WordNetLemmatizer
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
#nltk.download('omw-1.4')
from nltk.corpus import stopwords
import unicodedata
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize, sent_tokenize, regexp_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
import re
import html


# In[29]:


## Combining two columns reviews_title and reviews_text for vectorizer
df_class["reviews_title"] = df_class["reviews_title"].fillna('')
df_class['reviews_text_title']=df_class['reviews_title']+" "+df_class['reviews_text']


# In[30]:


# Remove special characters and converted to lower case.
stopword_list= stopwords.words('english')

def remove_spec(text):
    pattern1 = r'[^A-Za-z0-9_\s]' ## remove anything other than mentioned
    pattern2 = r'[^\w\s]'  ## remove punctuations if any.
    text=re.sub(pattern1, '', text.lower())
    text=re.sub(pattern2, '', text.lower())
    ##remove stopwords
    new_words=[]
    for word in text.split():
        if word in stopword_list:
            continue
        else:
            new_words.append(word)
    text=' '.join(new_words)
    return text
df_class['reviews_texttitle_lemmatized']=df_class['reviews_text_title'].apply(remove_spec)
df_class[['reviews_texttitle_lemmatized','reviews_text_title']].head(5)


# #### **Lemmatizing Words**

# In[31]:


### Lemmatizing words
#Write your function to Lemmatize the texts
import nltk
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk import pos_tag

wordnet_lemmatizer = WordNetLemmatizer()


from nltk.corpus import wordnet

def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN   # default

#---------------------------------------------------
def LemmatiseText(text):
    tokens = word_tokenize(str(text).lower())
    pos_tags = pos_tag(tokens)

    lemmas = [
        wordnet_lemmatizer.lemmatize(word, get_wordnet_pos(tag))
        for word, tag in pos_tags
        if word.isalpha()
        and wordnet_lemmatizer.lemmatize(word, get_wordnet_pos(tag)) not in {"be", "have", "do"}
    ]

    return " ".join(lemmas)



#---------------------------------------------------




df_class['reviews_texttitle_lemmatized']=df_class['reviews_texttitle_lemmatized'].apply(LemmatiseText)


# In[32]:


df_class.head()


# #### **Wordnet representation**

# In[33]:


###### Word Net representation
#Using a word cloud find the top 40 words by frequency among all the articles after processing the text
full_list = ' ' . join(df_class['reviews_texttitle_lemmatized'].tolist())

from wordcloud import WordCloud
wordcloud = WordCloud(width=800, height=400, background_color='white',max_words=40).generate(full_list)

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

import pickle

with open("./pickle/wordcloud.pkl", "wb") as f:
    pickle.dump(wordcloud, f)


# #### ** N-Gram Representation**

# In[34]:


#map categorical variable user_sentiment to 0 or 1
#df_class['user_sentiment'] = df_class['user_sentiment'].map({'Positive':1,'Negative':0})
df_class['user_sentiment'].value_counts(dropna=False)


# In[35]:


df_class.head()


# ###### Find the top unigrams,bigrams and trigrams by frequency among all the reviews by users

# In[36]:


#Write your code here to find the top 30 unigram frequency among the complaints in the cleaned dataframe(df_class).
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
from collections import Counter
def get_n_grams(df_clean,n,count):
    corpus = df_clean.dropna().tolist()
    vectorizer = CountVectorizer(ngram_range=(n, n), stop_words='english',dtype=np.float32,min_df=5)
    X = vectorizer.fit_transform(corpus)
    word_counts = X.toarray().sum(axis=0)
    words_df = pd.DataFrame({
    'word': vectorizer.get_feature_names_out(),
    'count': word_counts
    })
    top_n_unigrams = words_df.sort_values(by='count', ascending=False).head(count)
    # Plot with Plotly
    fig = px.bar(top_n_unigrams, x='word', y='count', title="Top Words")
    fig.show()
    return(top_n_unigrams)



# In[37]:


from plotly.offline import plot
import plotly.graph_objects as go
import plotly.express as px


# In[38]:


#Print the top 10 words in the unigram frequency
get_n_grams(df_class['reviews_texttitle_lemmatized'],1,10)


# In[39]:


get_n_grams(df_class['reviews_texttitle_lemmatized'],2,10)


# In[40]:


get_n_grams(df_class['reviews_texttitle_lemmatized'],3,10)


# In[ ]:





# #### **Feature extraction using tfidf Vectorizer**

# In[41]:


df_class['user_sentiment_mapped'] = df_class['user_sentiment'].replace({
    'Positive': 1,
    'Negative': 0,
    'positive': 1,
    'negative': 0
})


# In[42]:


df_class['user_sentiment_mapped'].value_counts()


# Since there is class imbalance seen with ~88% of positive reviews and remaining 10% negative reviews. We will do fix with SMOTE

# In[ ]:





# In[43]:


#using TF-IDF vectorizer using the parameters to get 650 features.
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=1000, max_df=0.9, min_df=7, binary=True,
                                   ngram_range=(1,2))
#df_class
X_train_tfidf = tfidf_vectorizer.fit_transform(df_class['reviews_texttitle_lemmatized'])
y= df_class['user_sentiment_mapped']
indices=df_class.index


# In[44]:


print(tfidf_vectorizer.get_feature_names_out())

import pickle

with open("./pickle/TfIdfVectorizer.pkl", "wb") as f:
    pickle.dump(tfidf_vectorizer, f)


# #### **Train Test Split**

# In[45]:


# splitting into test and train
from sklearn.model_selection  import train_test_split

#X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
#    X_train_tfidf, y, df_clean, test_size=0.25, random_state=42, stratify=y
#)

X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
    X_train_tfidf, y, indices, test_size=0.25, random_state=42, stratify=y
)



# #### **Since we are seeing class imbalance with positive reviews more, we will use SMOTE to balance the data.**

# In[46]:


from imblearn.over_sampling import SMOTE

count = Counter(y_train)
print("Before SMOTE: ",count)
sm=SMOTE()

X_train_res,y_train_res=sm.fit_resample(X_train,y_train)
count = Counter(y_train_res)
print("After SMOTE: ",count)


# ### **Classification model on the Train set.**

# #### **Logistic Classification**

# In[47]:


from sklearn.metrics import roc_curve,roc_auc_score
from sklearn import metrics

def build_model(X_train_res,y_train_res,X_test,y_test,modelname):
  modelname.fit(X_train_res,y_train_res)
  y_prob=modelname.predict_proba(X_test)[:,1]
  y_pred = (y_prob >= 0.5).astype(int)
  roc_auc = metrics.roc_auc_score(y_test, y_prob)
  results=[]
  print("Unique probabilities:", len(np.unique(y_prob)))
  print("Class balance:", y_test.value_counts())
  print("=============================================")
  print("Classification Report: \n")
  CR=classification_report(y_test,y_pred)
  print(CR)
  print("Confusion Matrix: \n")
  CM=confusion_matrix(y_test,y_pred)
  print(CM)
  print("Accuracy: \n")
  AC=accuracy_score(y_test,y_pred)
  print(AC)
  print("Precision: \n")
  PS=precision_score(y_test,y_pred)
  print(PS)
  print("F1 Score: \n")
  F1=f1_score(y_test,y_pred)
  print(F1)
  print("AUC: \n")
  AUC=roc_auc_score(y_test,y_pred)
  print(f"Roc-Auc Score is:{AUC*100:.1f}%")
  print("AUC Graph: \n")
  fpr, tpr, thresholds = roc_curve(y_test, y_pred)
  fig = px.area(
      x=fpr, y=tpr,
      #title=f'ROC Curve (AUC={(fpr, tpr):.4f})',
      labels=dict(x='False Positive Rate', y='True Positive Rate'),
      width=700, height=500)
  fig.add_shape(
      type='line', line=dict(dash='dash'),
      x0=0, x1=1, y0=0, y1=1)
  fig.update_yaxes(scaleanchor="x", scaleratio=1)
  fig.update_xaxes(constrain='domain')
  fig.show()

  accuracy = metrics.accuracy_score(y_test, y_pred)
  precision = metrics.precision_score(y_test, y_pred)
  recall = metrics.recall_score(y_test, y_pred)
  f1score = metrics.f1_score(y_test, y_pred)
  roc_auc = metrics.roc_auc_score(y_test, y_pred)
  results.append(accuracy)
  results.append(precision)
  results.append(recall)
  results.append(f1score)
  results.append(roc_auc)
  return results


# In[48]:


LR = LogisticRegression()
lr_results=build_model(X_train_res,y_train_res,X_test,y_test,LR)
#
#y_pred


# #### **Naive Bayes**

# In[49]:


# training the NB model and making predictions
from sklearn.naive_bayes import MultinomialNB


# In[50]:


mnb = MultinomialNB(alpha=1.0)
mnb_results=build_model(X_train_res, y_train_res, X_test,y_test,mnb)


# #### **Decision Tree**

# In[51]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV

DT = DecisionTreeClassifier()
#DT_results=build_model(X_train_res, y_train_res, X_test,y_test,DT)


# #### **Random Forest Classifier**

# In[52]:


#rc=RandomForestClassifier(oob_score=True,n_jobs=-1,random_state=42,criterion='gini')
#params = {
#    'max_depth': [2,3,5,10,20],
#    'min_samples_leaf': [5,10,20,50,100,200],
#    'n_estimators': [10,25,30,50,100,200]
#}

#grid_search=GridSearchCV(rc,param_grid=params,cv=3,n_jobs=-1,verbose=1)
#grid_search.fit(X_train_res,y_train_res)


# In[53]:


#print(grid_search.best_params_)
#print(grid_search.best_score_)
#rc_best=grid_search.best_estimator_
#print(rc_best)
#RC_results=build_model(X_train_res, y_train_res, X_test,y_test,rc_best)


# #### **XGBoostClassifier**

# In[54]:


import xgboost as xgb
XG = XGBClassifier(learning_rate=0.15,random_state=42,max_depth=10)
XG_results=build_model(X_train_res, y_train_res, X_test,y_test,XG)


# #### **MODEL INFERENCE**

# In[55]:


#metrics_table = {
#    'Model': ['Logistic Regression', 'Naive Bayes', 'Decision Tree', 'Random Forest', 'XGBoost'],
#    'Accuracy': [lr_results[0], mnb_results[0],DT_results[0] ,RC_results[0], XG_results[0]],
#    'Precision':[lr_results[1],mnb_results[1],DT_results[1] ,RC_results[1],XG_results[1]],
#    'Recall': [lr_results[2], mnb_results[2],DT_results[2] ,RC_results[2], XG_results[2]],
#    'F1Score': [lr_results[3], mnb_results[3],DT_results[3] ,RC_results[3], XG_results[3]],
#    'ROC Curve': [float(lr_results[4]), float(mnb_results[4]),DT_results[4] ,float(RC_results[4]), float(XG_results[4])],
#}
#metrics_df = pd.DataFrame(metrics_table)
#metrics_df


# #### **Comparing the all the model, it is found that XGBoost better model, Saving model**

# In[56]:


import pickle

with open("./pickle/XGBBoost.pkl", "wb") as f:
    pickle.dump(XG, f)


# ### Recommendation System
# Building at least two types of recommendation systems:
# 1. user-based recommendation systems
# 2. item-based recommendation systems

# In[57]:


df_class.head()


# #### **Adding Adjusted Rating**

# In[58]:


train_sent = XG.predict_proba(X_train)[:, 1]
test_sent  = XG.predict_proba(X_test)[:, 1]
X_train.shape


# In[59]:


idx_train.shape
idx_test.shape


# In[60]:


train_sent


# In[61]:


train_df = df_class.loc[idx_train].copy()
test_df  = df_class.loc[idx_test].copy()

#final_score = 0.6 * cf_score + 0.4 * sentiment


#train_df["adjusted_rating"] = train_df["reviews_rating"].values * (0.8 + 0.2 * train_sent)
#test_df["adjusted_rating"]  = test_df["reviews_rating"].values  * (0.8 + 0.2 * test_sent)

train_df["adjusted_rating"] = (
    0.6 * train_df["reviews_rating"]
    + 0.4 * (train_sent * 5)
)
test_df["adjusted_rating"] =(
    0.6 * test_df['reviews_rating']
    + 0.4 * (test_sent * 5)
)


# In[62]:


train_df.head(100)


# In[63]:


df_class.loc[7943]


# In[64]:


# Pivot the train ratings' dataset into matrix format in which columns are product names and the rows are user names.
import pandas as pd
df_pivot = pd.pivot_table(train_df,index='reviews_username', columns = 'id', values = 'adjusted_rating').fillna(0)
df_pivot.head(10)


# In[ ]:





# ##### **Creating dummy train and test**
# 
# 

# In[65]:


# Copy the train dataset into dummy_train
dummy_train = train_df.copy()


# In[66]:


dummy_train.head()


# In[67]:


# The products not rated by user is marked as 1 for prediction.
dummy_train['adjusted_rating'] = dummy_train['adjusted_rating'].apply(lambda x: 0 if x>=1 else 1)


# In[68]:


# Pivot the train ratings' dataset into matrix format in which columns are product names and the rows are user names.
dummy_train = pd.pivot_table(dummy_train,index='reviews_username', columns = 'id', values = 'adjusted_rating').fillna(1)
dummy_train.head(10)


# In[ ]:





# ##### Using Similarity Matrix

# In[69]:


df_pivot.index.nunique()


# In[70]:


from sklearn.metrics.pairwise import pairwise_distances, cosine_similarity


# In[71]:


####computing similarity matrix in df_pivot is crashing the system hence saving it in disk chunk wise and loading it


# In[72]:


#using cosine_similarity function to compute the distance.
user_correlation = cosine_similarity(df_pivot)
user_correlation[np.isnan(user_correlation)] = 0
print(user_correlation)
print(user_correlation.shape)


# #### **Prediction User-User**

# In[73]:


#filtering out the user_correlation that are negatively correlated
user_correlation[user_correlation<0]=0
user_correlation


# In[74]:


user_predicted_ratings = np.dot(user_correlation, df_pivot.fillna(0))
user_predicted_ratings


# In[75]:


#since we are interested in products that are not rated by the user, we multiply with dummy train to make it zero
user_final_rating = np.multiply(user_predicted_ratings,dummy_train)
user_final_rating.head()

### This will used for final recommender model saved in pickle file


# ##### **Find 20 recommendation for the user**

# In[76]:


user_input = "01impala" 
print(user_input)


# In[77]:


recommendations = user_final_rating.loc[user_input].sort_values(ascending=False)[0:20]
recommendations


# In[78]:


#display the top 20 product id, name and similarity_score 
#train_df.head()
final_recommendations = pd.DataFrame({'product_id': recommendations.index, 'similarity_score' : recommendations})
final_recommendations.reset_index(drop=True)
pd.merge(final_recommendations, train_df, on="id")[["id", "name", "similarity_score"]].drop_duplicates()


# #### **Pickle file for User Recommender**
# 
# Saving this model since user-user recommendation showed better results than item-item based. 
# 

# In[79]:


import pickle

with open("./pickle/user_final_rating.pkl", "wb") as f:
    pickle.dump(user_final_rating, f)


# #### **Evaluation User-User**
# 
# 
# 

# In[80]:


# Find out the common users of test and train dataset.
common = test_df[test_df.reviews_username.isin(train_df.reviews_username)]
common.shape


# In[81]:


common.head()


# In[82]:


# convert into the user-movie matrix.
common_user_based_matrix = pd.pivot_table(common,index='reviews_username', columns = 'id', values = 'adjusted_rating')
common_user_based_matrix.head()


# In[83]:


# Convert the user_correlation matrix into dataframe.
user_correlation_df = pd.DataFrame(user_correlation)
user_correlation_df.head()


# In[84]:


user_correlation_df['reviews_username'] = df_pivot.index
user_correlation_df.set_index('reviews_username',inplace=True)
user_correlation_df.head()


# In[85]:


list_name = common.reviews_username.tolist()

user_correlation_df.columns = df_pivot.index.tolist()
user_correlation_df_1 =  user_correlation_df[user_correlation_df.index.isin(list_name)]


# In[86]:


user_correlation_df_1.shape


# In[87]:


user_correlation_df_2 = user_correlation_df_1.T[user_correlation_df_1.T.index.isin(list_name)]


# In[88]:


user_correlation_df_3 = user_correlation_df_2.T


# In[89]:


user_correlation_df_3[user_correlation_df_3<0]=0

common_user_predicted_ratings = np.dot(user_correlation_df_3, common_user_based_matrix.fillna(0))
common_user_predicted_ratings


# In[ ]:





# In[90]:


dummy_test=common.copy()
dummy_test['adjusted_rating'] = dummy_test['adjusted_rating'].apply(lambda x: 1 if x >= 1 else 0)
dummy_test = pd.pivot_table(dummy_test,index='reviews_username', columns = 'id', values = 'adjusted_rating').fillna(0)
common.head()


# In[91]:


dummy_test.shape


# In[92]:


common_user_based_matrix.head()


# In[93]:


dummy_test.head()


# In[94]:


common_user_predicted_ratings = np.multiply(common_user_predicted_ratings,dummy_test)


# In[95]:


common_user_predicted_ratings.head()


# #### **Calculate RMSE User-User based recommendation system**

# In[96]:


#calculate RMSE

from sklearn.preprocessing import MinMaxScaler
from numpy import *

X  = common_user_predicted_ratings.copy() 
X = X[X>0]

scaler = MinMaxScaler(feature_range=(1, 5))
print(scaler.fit(X))
y = (scaler.transform(X))

print(y)


# In[97]:


common_ = pd.pivot_table(common,index='reviews_username', columns = 'id', values = 'adjusted_rating')


# In[98]:


# Finding total non-NaN value
total_non_nan = np.count_nonzero(~np.isnan(y))


# In[99]:


rmse = (sum(sum((common_ - y )**2))/total_non_nan)**0.5
print(rmse)


# #### **Item Item based recommendation**

# In[100]:


df_pivot = pd.pivot_table(train_df,
    index='id',
    columns='reviews_username',
    values='adjusted_rating'
)

df_pivot.head()


# In[101]:


mean = np.nanmean(df_pivot, axis=1)
df_subtracted = (df_pivot.T-mean).T


# In[102]:


df_subtracted.head()


# In[103]:


# Item Similarity Matrix
item_correlation = 1 - pairwise_distances(df_subtracted.fillna(0), metric='cosine')
item_correlation[np.isnan(item_correlation)] = 0
print(item_correlation)


# In[104]:


item_correlation[item_correlation<0]=0
item_correlation


# #### **Prediction - item-item**

# In[105]:


item_predicted_ratings = np.dot((df_pivot.fillna(0).T),item_correlation)
item_predicted_ratings


# ##### **Filtering the rating only for the products not rated by the user for recommendation**

# In[106]:


item_final_rating = np.multiply(item_predicted_ratings,dummy_train)
item_final_rating.head()


# ##### **Finding the top 20 recommendation for the user**

# In[107]:


# Take the user ID as input
user_input = '01impala'
print(user_input)


# In[108]:


# Recommending the Top 5 products to the user.
item_recommendations = item_final_rating.loc[user_input].sort_values(ascending=False)[0:20]
item_recommendations


# In[109]:


item_final_recommendations = pd.DataFrame({'product_id': item_recommendations.index, 'similarity_score' : item_recommendations})
item_final_recommendations.reset_index(drop=True)
#final_recommendations.drop(['id'], axis=1)
pd.merge(item_final_recommendations, train_df, on="id")[["id", "name", "similarity_score"]].drop_duplicates()


# #### **Evaluation - item-item**

# In[110]:


common =  test_df[test_df.id.isin(train_df.id)]
common.shape


# In[111]:


common.head(4)


# In[112]:


common_item_based_matrix = common.pivot_table(index='id', columns='reviews_username', values='adjusted_rating')


# In[113]:


item_correlation_df = pd.DataFrame(item_correlation)
item_correlation_df.head(1)


# In[114]:


item_correlation_df['id'] = df_subtracted.index
item_correlation_df.set_index('id',inplace=True)
item_correlation_df.head()


# In[115]:


list_name = common.id.tolist()


# In[116]:


item_correlation_df.columns = df_subtracted.index.tolist()

item_correlation_df_1 =  item_correlation_df[item_correlation_df.index.isin(list_name)]


# In[117]:


item_correlation_df_2 = item_correlation_df_1.T[item_correlation_df_1.T.index.isin(list_name)]

item_correlation_df_3 = item_correlation_df_2.T


# In[118]:


df_subtracted


# In[119]:


item_correlation_df_3[item_correlation_df_3<0]=0

common_item_predicted_ratings = np.dot(item_correlation_df_3, common_item_based_matrix.fillna(0))
common_item_predicted_ratings


# In[120]:


dummy_test = common.copy()
dummy_test['adjusted_rating'] = dummy_test['adjusted_rating'].apply(lambda x: 1 if x>=1 else 0)
dummy_test = pd.pivot_table(dummy_test, index='id', columns='reviews_username', values='adjusted_rating').fillna(0)
common_item_predicted_ratings = np.multiply(common_item_predicted_ratings,dummy_test)


# In[121]:


common_ = pd.pivot_table(common,index='id', columns='reviews_username', values='adjusted_rating')


# In[122]:


from sklearn.preprocessing import MinMaxScaler
from numpy import *

X  = common_item_predicted_ratings.copy() 
X = X[X>0]

scaler = MinMaxScaler(feature_range=(1, 5))
print(scaler.fit(X))
y = (scaler.transform(X))

print(y)


# In[123]:


# Finding total non-NaN value
total_non_nan = np.count_nonzero(~np.isnan(y))


# ##### **Computing RMSE for ITEM ITEM based recommendation system**

# In[124]:


rmse = (sum(sum((common_ - y )**2))/total_non_nan)**0.5
print(rmse)


# On comparing the RMSE values of User Based Recommender and Item Based Recommender, User based recommendation model seems to be better in this case, as it has a lower RMSE value (~2)

# #### **Top Product Recommendations - Recommendation of 20 products and filtering by sentiment model**
# 
# Get the top 20 product recommendations using the recommender system and get the top 5 using the sentiment ML model.. the similar method would be used in model.py

# In[125]:


def get_sentiment_recommendations(user):
    if (user in user_final_rating.index):
        # get the product recommedation using the trained ML model
        recommendations = list(user_final_rating.loc[user].sort_values(ascending=False)[0:20].index)
        temp = df_class[df_class.id.isin(recommendations)]
        #temp["reviews_text_cleaned"] = temp["reviews_text"].apply(lambda x: self.preprocess_text(x))
        #transfor the input data using saved tf-idf vectorizer
        X =  tfidf_vectorizer.transform(temp["reviews_texttitle_lemmatized"].values.astype(str))
        temp["predicted_sentiment"]= XG.predict(X)
        temp = temp[['name','predicted_sentiment']]
        temp_grouped = temp.groupby('name', as_index=False).count()
        temp_grouped["pos_review_count"] = temp_grouped.name.apply(lambda x: temp[(temp.name==x) & (temp.predicted_sentiment==1)]["predicted_sentiment"].count())
        temp_grouped["total_review_count"] = temp_grouped['predicted_sentiment']
        temp_grouped['pos_sentiment_percent'] = np.round(temp_grouped["pos_review_count"]/temp_grouped["total_review_count"]*100,2)
        return temp_grouped.sort_values('pos_sentiment_percent', ascending=False)
    else:
        print(f"User name {user} doesn't exist")


# In[126]:


#testing the above fuction using one of the users that's trained on.
get_sentiment_recommendations("01impala")


# In[127]:


#get the top 5
get_sentiment_recommendations("joshua")[:5]


# In[128]:


#testing the above fuction on the user that doesn't exists or a new user
get_sentiment_recommendations("mamathak")


# In[129]:


X_sample = tfidf_vectorizer.transform(["Awesome product, will recommend"])
y_pred_sample = XG.predict(X_sample)
y_pred_sample


# In[130]:


X_sample = tfidf_vectorizer.transform(["worst product, quality is poor"])
y_pred_sample = XG.predict(X_sample)
y_pred_sample


# In[131]:


with open("./pickle/cleaned_data.pkl", "wb") as f:
    pickle.dump(df_class, f)


# In[132]:


import pickle
df = pickle.load(open("pickle/user_final_rating.pkl", "rb"))

df.iloc[0].equals(df.iloc[1])


# In[ ]:




