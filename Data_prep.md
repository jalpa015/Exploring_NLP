```python
from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import decomposition, ensemble

import pandas, xgboost, numpy, textblob, string
from keras.preprocessing import text, sequence
from keras import layers, models, optimizers
```


```python
# Loading the DataSet

data = open('corpus', encoding="utf8").read()
labels, texts = [], []
for i, line in enumerate(data.split("\n")):
    content = line.split()
    labels.append(content[0])
    texts.append(" ".join(content[1:]))

# create a dataframe using texts and labels
trainDF = pandas.DataFrame()
trainDF['text'] = texts
trainDF['label'] = labels

```


```python
trainDF.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>text</th>
      <th>label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Stuning even for the non-gamer: This sound tra...</td>
      <td>__label__2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>The best soundtrack ever to anything.: I'm rea...</td>
      <td>__label__2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Amazing!: This soundtrack is my favorite music...</td>
      <td>__label__2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Excellent Soundtrack: I truly like this soundt...</td>
      <td>__label__2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Remember, Pull Your Jaw Off The Floor After He...</td>
      <td>__label__2</td>
    </tr>
  </tbody>
</table>
</div>




```python
# split the dataset into training and validation datasets 
train_x, valid_x, train_y, valid_y = model_selection.train_test_split(trainDF['text'], trainDF['label'])

# label encode the target variable 
encoder = preprocessing.LabelEncoder()
train_y = encoder.fit_transform(train_y)
valid_y = encoder.fit_transform(valid_y)
```

# Feature Engineering

Raw text data transofrmed into feature vectors and new features will be created using existing dataset. 

-> Count vectors as features
-> TF-IDF Vectors
    Word Level
    N-Gram Level
    Character Level
    
-> Word Embeddings as features
-> Text/NLP based features
-> Topic models as features

# Count Vectors


```python
count_vect = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}')
count_vect.fit(trainDF['text'])

# transform the training and validation data using count vectorizer object
xtrain_count =  count_vect.transform(train_x)
xvalid_count =  count_vect.transform(valid_x)
```

# TF-IDF Vectors


```python
# word level tf-idf
tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=5000)
tfidf_vect.fit(trainDF['text'])
xtrain_tfidf =  tfidf_vect.transform(train_x)
xvalid_tfidf =  tfidf_vect.transform(valid_x)

# ngram level tf-idf 
tfidf_vect_ngram = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', ngram_range=(2,3), max_features=5000)
tfidf_vect_ngram.fit(trainDF['text'])
xtrain_tfidf_ngram =  tfidf_vect_ngram.transform(train_x)
xvalid_tfidf_ngram =  tfidf_vect_ngram.transform(valid_x)

# characters level tf-idf
tfidf_vect_ngram_chars = TfidfVectorizer(analyzer='char', token_pattern=r'\w{1,}', ngram_range=(2,3), max_features=5000)
tfidf_vect_ngram_chars.fit(trainDF['text'])
xtrain_tfidf_ngram_chars =  tfidf_vect_ngram_chars.transform(train_x) 
xvalid_tfidf_ngram_chars =  tfidf_vect_ngram_chars.transform(valid_x)
```

    C:\ProgramData\Anaconda3\lib\site-packages\sklearn\feature_extraction\text.py:524: UserWarning: The parameter 'token_pattern' will not be used since 'analyzer' != 'word'
      warnings.warn("The parameter 'token_pattern' will not be used"
    

# Embeddings


```python
embeddings_index = {}
for i, line in enumerate(open('data/embeddings/wiki-news-300d-1M.vec', 'r', encoding='utf-8', newline='\n', errors='ignore')):
    values = line.split()
    embeddings_index[values[0]] = numpy.asarray(values[1:], dtype='float32')
```


```python
# Tokenizer 
token = text.Tokenizer()
token.fit_on_texts(trainDF['text'])
word_index = token.word_index
```


```python
# convert text to sequence of tokens and pad them to ensure equal length vectors 
train_seq_x = sequence.pad_sequences(token.texts_to_sequences(train_x), maxlen=70)
valid_seq_x = sequence.pad_sequences(token.texts_to_sequences(valid_x), maxlen=70)

# create token-embedding mapping
embedding_matrix = numpy.zeros((len(word_index) + 1, 300))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector
```

# Train Model


```python
def train_model(classifier, feature_vector_train, label, feature_vector_valid, is_neural_net=False):
    
    if is_neural_net:
        classifier.fit(feature_vector_train, label, epochs=10)
    else:
        # fit the training dataset on the classifier
        classifier.fit(feature_vector_train, label)
    
    # predict the labels on validation dataset
    predictions = classifier.predict(feature_vector_valid)
    
    if is_neural_net:
        predictions = predictions.argmax(axis=-1)
    
    return metrics.accuracy_score(predictions, valid_y)
```

# Naive Bayes


```python
# Naive Bayes on Count Vectors
accuracy = train_model(naive_bayes.MultinomialNB(), xtrain_count, train_y, xvalid_count)
print("NB, Count Vectors: ", accuracy*100)

# Naive Bayes on Word Level TF IDF Vectors
accuracy = train_model(naive_bayes.MultinomialNB(), xtrain_tfidf, train_y, xvalid_tfidf)
print("NB, WordLevel TF-IDF: ", accuracy*100)

# Naive Bayes on Ngram Level TF IDF Vectors
accuracy = train_model(naive_bayes.MultinomialNB(), xtrain_tfidf_ngram, train_y, xvalid_tfidf_ngram)
print("NB, N-Gram Vectors: ", accuracy*100)

# Naive Bayes on Character Level TF IDF Vectors
accuracy = train_model(naive_bayes.MultinomialNB(), xtrain_tfidf_ngram_chars, train_y, xvalid_tfidf_ngram_chars)
print("NB, CharLevel Vectors: ", accuracy*100)
```

    NB, Count Vectors:  81.92
    NB, WordLevel TF-IDF:  83.56
    NB, N-Gram Vectors:  82.88
    NB, CharLevel Vectors:  80.4
    

# Linear Classifier


```python
# Linear Classifier on Count Vectors
accuracy = train_model(linear_model.LogisticRegression(), xtrain_count, train_y, xvalid_count)
print("LR, Count Vectors: ", accuracy*100)

# Linear Classifier on Word Level TF IDF Vectors
accuracy = train_model(linear_model.LogisticRegression(), xtrain_tfidf, train_y, xvalid_tfidf)
print("LR, WordLevel TF-IDF: ", accuracy*100)

# Linear Classifier on Ngram Level TF IDF Vectors
accuracy = train_model(linear_model.LogisticRegression(), xtrain_tfidf_ngram, train_y, xvalid_tfidf_ngram)
print("LR, N-Gram Vectors: ", accuracy*100)

# Linear Classifier on Character Level TF IDF Vectors
accuracy = train_model(linear_model.LogisticRegression(), xtrain_tfidf_ngram_chars, train_y, xvalid_tfidf_ngram_chars)
print("LR, CharLevel Vectors: ", accuracy*100)
```

    C:\ProgramData\Anaconda3\lib\site-packages\sklearn\linear_model\_logistic.py:940: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      extra_warning_msg=_LOGISTIC_SOLVER_CONVERGENCE_MSG)
    

    LR, Count Vectors:  84.84
    LR, WordLevel TF-IDF:  86.56
    LR, N-Gram Vectors:  82.48
    LR, CharLevel Vectors:  83.88
    

# SVM


```python
# SVM on Ngram Level Count Vectors
accuracy = train_model(svm.SVC(), xtrain_count, train_y, xvalid_count)
print("SVM, Count Vectors: ", accuracy*100)

# Linear Classifier on Word Level TF IDF Vectors
accuracy = train_model(svm.SVC(), xtrain_tfidf, train_y, xvalid_tfidf)
print("SVM, WordLevel TF-IDF: ", accuracy*100)

# SVM on Ngram Level N-Gram Vectors
accuracy = train_model(svm.SVC(), xtrain_tfidf_ngram, train_y, xvalid_tfidf_ngram)
print("SVM, N-Gram Vectors: ", accuracy*100)
```

    SVM, Count Vectors:  84.72
    SVM, WordLevel TF-IDF:  86.68
    SVM, N-Gram Vectors:  83.2
    

# Bagging


```python
# RF on Count Vectors
accuracy = train_model(ensemble.RandomForestClassifier(), xtrain_count, train_y, xvalid_count)
print("RF, Count Vectors: ", accuracy*100)

# RF on Word Level TF IDF Vectors
accuracy = train_model(ensemble.RandomForestClassifier(), xtrain_tfidf, train_y, xvalid_tfidf)
print("RF, WordLevel TF-IDF: ", accuracy*100)
```

    RF, Count Vectors:  82.6
    RF, WordLevel TF-IDF:  82.96
    

# Neural Networks


```python
def create_model_architecture(input_size):
    # create input layer 
    input_layer = layers.Input((input_size, ), sparse=True)
    
    # create hidden layer
    hidden_layer = layers.Dense(100, activation="relu")(input_layer)
    
    # create output layer
    output_layer = layers.Dense(1, activation="sigmoid")(hidden_layer)

    classifier = models.Model(inputs = input_layer, outputs = output_layer)
    classifier.compile(optimizer=optimizers.Adam(), loss='binary_crossentropy',  metrics=['accuracy'])
    return classifier 

classifier = create_model_architecture(xtrain_tfidf_ngram.shape[1])
classifier.fit(xtrain_tfidf_ngram, train_y, epochs=10)
classifier.predict(xvalid_tfidf_ngram)
print("NN, Ngram Level TF IDF Vectors",  classifier.evaluate(xtrain_tfidf_ngram, train_y)[1]*100)
```

    Epoch 1/10
    235/235 [==============================] - 1s 2ms/step - loss: 0.6156 - accuracy: 0.7079
    Epoch 2/10
    235/235 [==============================] - 1s 3ms/step - loss: 0.2769 - accuracy: 0.9072
    Epoch 3/10
    235/235 [==============================] - 1s 2ms/step - loss: 0.1858 - accuracy: 0.9424
    Epoch 4/10
    235/235 [==============================] - 1s 2ms/step - loss: 0.1279 - accuracy: 0.9653
    Epoch 5/10
    235/235 [==============================] - 1s 2ms/step - loss: 0.0939 - accuracy: 0.9795
    Epoch 6/10
    235/235 [==============================] - 1s 2ms/step - loss: 0.0678 - accuracy: 0.9871
    Epoch 7/10
    235/235 [==============================] - 1s 2ms/step - loss: 0.0499 - accuracy: 0.9927
    Epoch 8/10
    235/235 [==============================] - 1s 2ms/step - loss: 0.0331 - accuracy: 0.9979
    Epoch 9/10
    235/235 [==============================] - 1s 2ms/step - loss: 0.0238 - accuracy: 0.9992
    Epoch 10/10
    235/235 [==============================] - 1s 2ms/step - loss: 0.0172 - accuracy: 0.9987
    235/235 [==============================] - 0s 788us/step - loss: 0.0126 - accuracy: 0.9987
    235/235 [==============================] - 0s 754us/step - loss: 0.0126 - accuracy: 0.9987
    NN, Ngram Level TF IDF Vectors 99.86666440963745
    

# Deep Neural Networks

CNN


```python
def create_cnn():
    # Add an Input Layer
    input_layer = layers.Input((70, ))

    # Add the word embedding Layer
    embedding_layer = layers.Embedding(len(word_index) + 1, 300, weights=[embedding_matrix], trainable=False)(input_layer)
    embedding_layer = layers.SpatialDropout1D(0.3)(embedding_layer)

    # Add the convolutional Layer
    conv_layer = layers.Convolution1D(100, 3, activation="relu")(embedding_layer)

    # Add the pooling Layer
    pooling_layer = layers.GlobalMaxPool1D()(conv_layer)

    # Add the output Layers
    output_layer1 = layers.Dense(50, activation="relu")(pooling_layer)
    output_layer1 = layers.Dropout(0.25)(output_layer1)
    output_layer2 = layers.Dense(1, activation="sigmoid")(output_layer1)

    # Compile the model
    model = models.Model(inputs=input_layer, outputs=output_layer2)
    model.compile(optimizer=optimizers.Adam(), loss='binary_crossentropy', metrics=['accuracy'])
    
    return model

classifier = create_cnn()
classifier.fit(train_seq_x, train_y, epochs=10)
classifier.predict(valid_seq_x)
accuracy=classifier.evaluate(train_seq_x, train_y)
print(accuracy)
print("CNN, Word Embeddings",  accuracy[1]*100)
```

    Epoch 1/10
    235/235 [==============================] - 2s 8ms/step - loss: 0.6491 - accuracy: 0.5972
    Epoch 2/10
    235/235 [==============================] - 2s 8ms/step - loss: 0.3863 - accuracy: 0.8273
    Epoch 3/10
    235/235 [==============================] - 2s 9ms/step - loss: 0.3269 - accuracy: 0.8603
    Epoch 4/10
    235/235 [==============================] - 2s 9ms/step - loss: 0.2650 - accuracy: 0.8930
    Epoch 5/10
    235/235 [==============================] - 2s 9ms/step - loss: 0.2309 - accuracy: 0.9073
    Epoch 6/10
    235/235 [==============================] - 2s 9ms/step - loss: 0.1930 - accuracy: 0.9268
    Epoch 7/10
    235/235 [==============================] - 2s 9ms/step - loss: 0.1562 - accuracy: 0.9407
    Epoch 8/10
    235/235 [==============================] - 2s 9ms/step - loss: 0.1283 - accuracy: 0.9539
    Epoch 9/10
    235/235 [==============================] - 2s 9ms/step - loss: 0.0922 - accuracy: 0.9686
    Epoch 10/10
    235/235 [==============================] - 2s 9ms/step - loss: 0.0819 - accuracy: 0.9694
    235/235 [==============================] - 1s 3ms/step - loss: 0.0129 - accuracy: 0.9995
    [0.01294676586985588, 0.9994666576385498]
    CNN, Word Embeddings 99.94666576385498
    

LSTM


```python
def create_rnn_lstm():
    # Add an Input Layer
    input_layer = layers.Input((70, ))

    # Add the word embedding Layer
    embedding_layer = layers.Embedding(len(word_index) + 1, 300, weights=[embedding_matrix], trainable=False)(input_layer)
    embedding_layer = layers.SpatialDropout1D(0.3)(embedding_layer)

    # Add the LSTM Layer
    lstm_layer = layers.LSTM(100)(embedding_layer)

    # Add the output Layers
    output_layer1 = layers.Dense(50, activation="relu")(lstm_layer)
    output_layer1 = layers.Dropout(0.25)(output_layer1)
    output_layer2 = layers.Dense(1, activation="sigmoid")(output_layer1)

    # Compile the model
    model = models.Model(inputs=input_layer, outputs=output_layer2)
    model.compile(optimizer=optimizers.Adam(), loss='binary_crossentropy', metrics=['accuracy'])
    
    return model

classifier = create_rnn_lstm()
classifier.fit(train_seq_x, train_y, epochs=10)
classifier.predict(valid_seq_x)
accuracy=classifier.evaluate(train_seq_x, train_y)
print("RNN-LSTM, Word Embeddings",  accuracy[1]*100)
```

    Epoch 1/10
    235/235 [==============================] - 8s 29ms/step - loss: 0.6350 - accuracy: 0.6201
    Epoch 2/10
    235/235 [==============================] - 7s 29ms/step - loss: 0.4874 - accuracy: 0.7712
    Epoch 3/10
    235/235 [==============================] - 7s 29ms/step - loss: 0.4712 - accuracy: 0.7792
    Epoch 4/10
    235/235 [==============================] - 7s 30ms/step - loss: 0.4481 - accuracy: 0.7919
    Epoch 5/10
    235/235 [==============================] - 7s 30ms/step - loss: 0.4154 - accuracy: 0.8183
    Epoch 6/10
    235/235 [==============================] - 7s 29ms/step - loss: 0.3967 - accuracy: 0.8172
    Epoch 7/10
    235/235 [==============================] - ETA: 0s - loss: 0.3674 - accuracy: 0.83 - 7s 29ms/step - loss: 0.3674 - accuracy: 0.8315
    Epoch 8/10
    235/235 [==============================] - 7s 30ms/step - loss: 0.3483 - accuracy: 0.8460
    Epoch 9/10
    235/235 [==============================] - 7s 29ms/step - loss: 0.3402 - accuracy: 0.8474
    Epoch 10/10
    235/235 [==============================] - 7s 29ms/step - loss: 0.3227 - accuracy: 0.8592
    235/235 [==============================] - 4s 14ms/step - loss: 0.2787 - accuracy: 0.8749
    RNN-LSTM, Word Embeddings 87.49333620071411
    

GRU


```python
def create_rnn_gru():
    # Add an Input Layer
    input_layer = layers.Input((70, ))

    # Add the word embedding Layer
    embedding_layer = layers.Embedding(len(word_index) + 1, 300, weights=[embedding_matrix], trainable=False)(input_layer)
    embedding_layer = layers.SpatialDropout1D(0.3)(embedding_layer)

    # Add the GRU Layer
    lstm_layer = layers.GRU(100)(embedding_layer)

    # Add the output Layers
    output_layer1 = layers.Dense(50, activation="relu")(lstm_layer)
    output_layer1 = layers.Dropout(0.25)(output_layer1)
    output_layer2 = layers.Dense(1, activation="sigmoid")(output_layer1)

    # Compile the model
    model = models.Model(inputs=input_layer, outputs=output_layer2)
    model.compile(optimizer=optimizers.Adam(), loss='binary_crossentropy', metrics=['accuracy'])
    
    return model

classifier = create_rnn_gru()
classifier.fit(train_seq_x, train_y, epochs=10)
classifier.predict(valid_seq_x)
accuracy=classifier.evaluate(train_seq_x, train_y)
print("RNN-GRU, Word Embeddings",  accuracy[1]*100)
```

    Epoch 1/10
    235/235 [==============================] - 8s 27ms/step - loss: 0.6635 - accuracy: 0.5928
    Epoch 2/10
    235/235 [==============================] - 6s 26ms/step - loss: 0.4855 - accuracy: 0.7631
    Epoch 3/10
    235/235 [==============================] - 6s 26ms/step - loss: 0.3981 - accuracy: 0.8205
    Epoch 4/10
    235/235 [==============================] - 6s 27ms/step - loss: 0.3566 - accuracy: 0.8450
    Epoch 5/10
    235/235 [==============================] - 8s 32ms/step - loss: 0.3499 - accuracy: 0.8471
    Epoch 6/10
    235/235 [==============================] - 6s 27ms/step - loss: 0.3280 - accuracy: 0.8569
    Epoch 7/10
    235/235 [==============================] - 6s 27ms/step - loss: 0.3134 - accuracy: 0.8600
    Epoch 8/10
    235/235 [==============================] - 6s 27ms/step - loss: 0.2981 - accuracy: 0.8717
    Epoch 9/10
    235/235 [==============================] - 6s 26ms/step - loss: 0.2902 - accuracy: 0.8717
    Epoch 10/10
    235/235 [==============================] - 6s 27ms/step - loss: 0.2831 - accuracy: 0.8811
    235/235 [==============================] - 3s 10ms/step - loss: 0.2286 - accuracy: 0.9004
    RNN-GRU, Word Embeddings 90.03999829292297
    


