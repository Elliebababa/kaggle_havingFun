
# coding: utf-8

# In[1]:


# coding: utf-8
import pickle
from keras.optimizers import SGD, Adam, Nadam, RMSprop
from keras.models import Sequential,Model,load_model
from keras.layers import Embedding,Conv1D,MaxPooling1D,Input
from keras.layers.core import Dense, Activation,Dropout ,Flatten
from keras.layers.recurrent import LSTM
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import sequence
from keras.preprocessing.text import text_to_word_sequence,one_hot,Tokenizer
from keras.constraints import maxnorm
from keras.callbacks import ModelCheckpoint,TensorBoard, ReduceLROnPlateau,EarlyStopping
from keras.applications import Xception
from keras import regularizers
from keras import backend as K
import keras
import numpy as np
import pandas as pd
from keras.preprocessing import text
import os
import glob
import math
seed = 7
np.random.seed(seed)


def shuffle_2(a, b): # Shuffles 2 arrays with the same order
    s = np.arange(a.shape[0])
    np.random.shuffle(s)
    return a[s], b[s]

MAX_NUM_WORDS = 20000

DATA_DIR = '.\\data'
GLOVE_DIR = os.path.join(DATA_DIR, 'glove6B')
EMBEDDING_DIM = 100
# # first, build index mapping words in the embeddings set
# # to their embedding vector

print('Indexing word vectors...')

embeddings_index = {}
with open(os.path.join(GLOVE_DIR, 'glove.6B.100d.txt'), encoding= 'utf-8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs

print('Found %s word vectors.' % len(embeddings_index))

train_df = pd.read_csv('.\\data\\train.tsv',header=0,delimiter='\t')
test_df = pd.read_csv('.\\data\\test.tsv',header=0,delimiter='\t')

raw_docs_train = train_df['Phrase'].values


# In[7]:


X_train = train_df['Phrase']
Y_train = train_df['Sentiment']
feature_names = train_df.columns.values
X_test = test_df['Phrase']
X_test_PhraseID = test_df['PhraseId']


# In[8]:


tokenizer = Tokenizer()
tokenizer.fit_on_texts(np.concatenate((X_train, X_test), axis=0))
Tokenizer_vocab_size = len(tokenizer.word_index) + 1
print(Tokenizer_vocab_size)
word_index = tokenizer.word_index

# saving
#with open('tokenizer.pickle', 'wb') as handle:
#    pickle.dump(Tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

#======================================== splitting data to train set and test set =====================================================

num_test = 32000

Y_Val = Y_train[:num_test]
Y_Val2 = Y_train[:num_test]
X_Val = X_train[:num_test]

X_train = X_train[num_test:]
Y_train = Y_train[num_test:]

maxWordCount = 60
maxDictionary_size = Tokenizer_vocab_size

#======================================== to_sequences =====================================================

encoded_words = tokenizer.texts_to_sequences(X_train)
encoded_words2 = tokenizer.texts_to_sequences(X_Val)
encoded_words3 = tokenizer.texts_to_sequences(X_test)

#======================================== padding =====================================================

X_Train_encodedPadded_words = sequence.pad_sequences(encoded_words,maxlen = maxWordCount)
X_Val_encodedPadded_words = sequence.pad_sequences(encoded_words2, maxlen=maxWordCount)
X_test_encodedPadded_words = sequence.pad_sequences(encoded_words3, maxlen=maxWordCount)

#======================================== one-hot labeling =====================================================

Y_train = keras.utils.to_categorical(Y_train,5)
Y_Val = keras.utils.to_categorical(Y_Val,5)

#======================================== shuffling =====================================================

shuffle_2(X_Train_encodedPadded_words,Y_train)

#======================================== Embedding =============================================================
# prepare embedding matrix
num_words = min(MAX_NUM_WORDS, len(word_index) + 1)
embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
for word, i in word_index.items():
    if i >= MAX_NUM_WORDS:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector

# load pre-trained word embeddings into an Embedding layer
# note that we set trainable = False so as to keep the embeddings fixed
embedding_layer = Embedding(num_words,EMBEDDING_DIM,weights=[embedding_matrix],input_length=maxWordCount,trainable=True)


#==================================================Attention Layer===========================================
from keras import backend as K
from keras.engine.topology import Layer
#from keras import initializations
from keras import initializers, regularizers, constraints


class Att(Layer):
    def __init__(self, step_dim,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        """
        Keras Layer that implements an Attention mechanism for temporal data.
        Supports Masking.
        Follows the work of Raffel et al. [https://arxiv.org/abs/1512.08756]
        # Input shape
            3D tensor with shape: `(samples, steps, features)`.
        # Output shape
            2D tensor with shape: `(samples, features)`.
        :param kwargs:
        Just put it on top of an RNN Layer (GRU/LSTM/SimpleRNN) with return_sequences=True.
        The dimensions are inferred based on the output shape of the RNN.
        Example:
            model.add(LSTM(64, return_sequences=True))
            model.add(Attention())
        """
        self.supports_masking = True
        #self.init = initializations.get('glorot_uniform')
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.step_dim = step_dim
        self.features_dim = 0
        super(Att, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight((input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        self.features_dim = input_shape[-1]

        if self.bias:
            self.b = self.add_weight((input_shape[1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None

        self.built = True

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):
        # eij = K.dot(x, self.W) TF backend doesn't support it

        # features_dim = self.W.shape[0]
        # step_dim = x._keras_shape[1]

        features_dim = self.features_dim
        step_dim = self.step_dim

        eij = K.reshape(K.dot(K.reshape(x, (-1, features_dim)), K.reshape(self.W, (features_dim, 1))), (-1, step_dim))

        if self.bias:
            eij += self.b

        eij = K.tanh(eij)

        a = K.exp(eij)

        # apply mask after the exp. will be re-normalized next
        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            a *= K.cast(mask, K.floatx())

        # in some cases especially in the early stages of training the sum may be almost zero
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a
    #print weigthted_input.shape
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        #return input_shape[0], input_shape[-1]
        return input_shape[0],  self.features_dim
       


learning_rate = 0.0001
epochs = 30
batch_size = 32



#========================================= model LSTM ================================================================

tensorboard1 = keras.callbacks.TensorBoard(log_dir='./logs/log_1',histogram_freq=0,write_graph=True,write_images=False)
checkpointer1 = ModelCheckpoint(filepath = "./weights/weights_1",verbose = 1, save_best_only = True, monitor = "val_loss")
reducer_lr1 = ReduceLROnPlateau(monitor='val_loss',factor=0.8,patience=0,verbose = 1, mode = 'auto', cooldown = 0, min_lr = 1e-6)
earlyStopping1 = EarlyStopping(monitor = 'val_loss',min_delta=0,patience=4,verbose=1)


review_input = Input(shape = (60,),dtype = 'int32')
embedded_sequence = embedding_layer(review_input)
lstmLayer  = LSTM(128, dropout_W=0.2, dropout_U=0.2)
x = lstmLayer(embedded_sequence)
denseLayer = Dense(5, activation='softmax')(x) 

model1 = Model(inputs=[review_input],outputs=denseLayer)
model1.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics = ['accuracy'])
model1.summary()

#====================================================training====================================================

history1  = model1.fit(X_Train_encodedPadded_words, Y_train, epochs = epochs, batch_size=batch_size, verbose=1, validation_data=(X_Val_encodedPadded_words, Y_Val), callbacks=[tensorboard1, reducer_lr1,checkpointer1,earlyStopping1])
scores1 = model1.evaluate(X_Val_encodedPadded_words, Y_Val, verbose=0)

model1.save('lstm_model.h5')

from keras.utils import plot_model
plot_model(model1,to_file='model1.png')

#predicted_classes1 = np.argmax(model1.predict(X_test_encodedPadded_words,batch_size = batch_size,verbose =1),axis=1)
#submission1=pd.DataFrame({'PhraseId':X_test_PhraseID,'Sentiment':predicted_classes1})
#submission1.to_csv('./submission1.csv',index=False)




#========================================= model BiLSTM================================================================

from keras.layers.wrappers import Bidirectional

tensorboard2 = keras.callbacks.TensorBoard(log_dir='./logs/log_2',histogram_freq=0,write_graph=True,write_images=False)
checkpointer2 = ModelCheckpoint(filepath = "./weights/weights_2",verbose = 1, save_best_only = True, monitor = "val_loss")
reducer_lr2 = ReduceLROnPlateau(monitor='val_loss',factor=0.8,patience=0,verbose = 1, mode = 'auto', cooldown = 0, min_lr = 1e-6)
earlyStopping2 = EarlyStopping(monitor = 'val_loss',min_delta=0,patience=4,verbose=1)


review_input = Input(shape = (60,),dtype = 'int32')
embedded_sequence = embedding_layer(review_input)
bilstmLayer  = Bidirectional(LSTM(128, dropout_W=0.2, dropout_U=0.2))
x = bilstmLayer(embedded_sequence)
denseLayer2 = Dense(5, activation='softmax')(x) 

model2 = Model(inputs=[review_input],outputs=denseLayer2)
model2.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics = ['accuracy'])
model2.summary()

#====================================================training====================================================

history2  = model2.fit(X_Train_encodedPadded_words, Y_train, epochs = epochs, batch_size=batch_size, verbose=1, validation_data=(X_Val_encodedPadded_words, Y_Val), callbacks=[tensorboard2, reducer_lr2,checkpointer2,earlyStopping2])
scores2 = model2.evaluate(X_Val_encodedPadded_words, Y_Val, verbose=0)

model2.save('bilstm_model.h5')

from keras.utils import plot_model
plot_model(model1,to_file='model2.png')


#predicted_classes2 = np.argmax(model2.predict(X_test_encodedPadded_words,batch_size = batch_size,verbose =1),axis=1)
#submission2=pd.DataFrame({'PhraseId':X_test_PhraseID,'Sentiment':predicted_classes2})
#submission2.to_csv('./submission2.csv',index=False)



#========================================= model LSTM+ATTENTION================================================================

tensorboard3 = keras.callbacks.TensorBoard(log_dir='./logs/log_3',histogram_freq=0,write_graph=True,write_images=False)
checkpointer3 = ModelCheckpoint(filepath = "./weights/weights_3",verbose = 1, save_best_only = True, monitor = "val_loss")
reducer_lr3 = ReduceLROnPlateau(monitor='val_loss',factor=0.8,patience=0,verbose = 1, mode = 'auto', cooldown = 0, min_lr = 1e-6)
earlyStopping3 = EarlyStopping(monitor = 'val_loss',min_delta=0,patience=4,verbose=1)


review_input = Input(shape = (60,),dtype = 'int32')
embedded_sequence = embedding_layer(review_input)
lstmLayer  = LSTM(128, dropout_W=0.2, dropout_U=0.2,return_sequences = True)
x = lstmLayer(embedded_sequence)
att = Att(60)(x)  
denseLayer3 = Dense(5, activation='softmax')(att) 

model3 = Model(inputs=[review_input],outputs=denseLayer3)
model3.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics = ['accuracy'])
model3.summary()

#====================================================training====================================================

history3  = model3.fit(X_Train_encodedPadded_words, Y_train, epochs = epochs, batch_size=batch_size, verbose=1, validation_data=(X_Val_encodedPadded_words, Y_Val), callbacks=[tensorboard3, reducer_lr3,checkpointer3,earlyStopping3])
scores3 = model3.evaluate(X_Val_encodedPadded_words, Y_Val, verbose=0)

model3.save('lstm+attention_model.h5')

from keras.utils import plot_model
plot_model(model1,to_file='model3.png')


#predicted_classes3 = np.argmax(model3.predict(X_test_encodedPadded_words,batch_size = batch_size,verbose =1),axis=1)
#submission3=pd.DataFrame({'PhraseId':X_test_PhraseID,'Sentiment':predicted_classes3})
#submission3.to_csv('./submission3.csv',index=False)




