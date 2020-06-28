import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

data = pd.read_csv('/kaggle/input/amazon-music-reviews/Musical_instruments_reviews.csv')
print(data)

data.isna().sum()

data.reviewText.fillna("",inplace = True)data.drop(['reviewerID', 'asin', 'reviewerName', 'helpful', 'unixReviewTime', 'reviewTime'], axis = 1, inplace = True)
print(data)

print(data['reviewText'][5])

data['review'] = data['reviewText'] + ' ' + data['summary']
data.drop(['reviewText', 'summary'], axis = 1, inplace = True)
print(data)

for i in range(0, data['overall'].size):
    if data['overall'][i] > 3.0:
        data['overall'][i] = 1
    else:
        data['overall'][i] = 0

print(data['overall'].value_counts())

#Cleaning the text
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
for i in range(0, data.shape[0]):
    review = re.sub('[^a-zA-Z]',' ', str(data['review'][i]))
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)

#Creating the bag of words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)

#Independent Vector
X = cv.fit_transform(corpus).toarray()
#Dependent Vector
y = data['overall']

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#Creating ANN
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
classifier = Sequential()
classifier.add(Dense(units = 75 , activation = 'relu' , input_dim = X_train.shape[1]))
classifier.add(Dense(units = 50 , activation = 'relu'))
classifier.add(Dense(units = 25 , activation = 'relu'))
classifier.add(Dense(units = 10 , activation = 'relu')) 
classifier.add(Dense(units = 1 , activation = 'sigmoid'))
classifier.compile(optimizer = 'adam' , loss = 'binary_crossentropy' , metrics = ['accuracy'])

classifier.fit(X_train,y_train , epochs = 10)

# Evaluate the model on the test data using `evaluate`
print("Evaluate on test data")
results = classifier.evaluate(X_test,y_test)
print("test loss, test acc:", results)

