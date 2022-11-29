import keras.backend as K
import keras
import tensorflow as tf
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import io
from nltk import tokenize
from keras.utils.np_utils import to_categorical
from keras.layers import Dense, Flatten, Input,Reshape,GRU,AveragePooling1D,Conv2D
from keras.models import Model

import sys
reload(sys)
sys.setdefaultencoding('utf-8')


# evaluation
def f1_score(y_true, y_pred):
    # calculate tp tn fp fn
    tp = K.sum(K.cast(y_true * y_pred, 'float'), axis=0)
    tn = K.sum(K.cast((1 - y_true) * (1 - y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1 - y_true) * y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true * (1 - y_pred), 'float'), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2 * p * r / (p + r + K.epsilon())
    f1 = tf.where(tf.math.is_nan(f1), tf.zeros_like(f1), f1)
    return f1


def precision(y_true, y_pred):
    # calculate tp tn fp fn
    tp = K.sum(K.cast(y_true * y_pred, 'float'), axis=0)
    tn = K.sum(K.cast((1 - y_true) * (1 - y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1 - y_true) * y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true * (1 - y_pred), 'float'), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())
    # calculate f1
    # f1 = 2*p*r / (p+r+K.epsilon())
    # f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
    return p


def recall(y_true, y_pred):
    # calculate tp tn fp fn
    tp = K.sum(K.cast(y_true * y_pred, 'float'), axis=0)
    tn = K.sum(K.cast((1 - y_true) * (1 - y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1 - y_true) * y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true * (1 - y_pred), 'float'), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())
    # calculate f1
    # f1 = 2*p*r / (p+r+K.epsilon())
    # f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
    return r


# ==============================================================================================================#


VALIDATION_SPLIT = 0.10
platform = "politifact"

data_train = pd.read_csv('data/' + platform + '/' + platform + '_content_no_ignore.tsv', sep='\t', encoding="utf-8")
contents_vector = io.open('data/' + platform + '/' + platform + "_content_vector_200_50_1.txt", "r", encoding="utf-8")
VALIDATION_SPLIT = 0.15
contents = []
labels = []
texts = []
ids = []
content_vec_dic = {}
for i in contents_vector:
    i = i.strip("\n")
    e = i.split("$:$:")
    v = e[1].split(" ")
    for j in range(len(v)):
        v[j] = v[j].lstrip(":")
        v[j] = float(v[j])
    temp = np.empty((1, 200))
    for i in range(200):
        temp[0][i] = v[i]
    v = temp
    content_vec_dic[str(e[0])] = v

for idx in range(data_train.content.shape[0]):
    text = data_train.content[idx]
    texts.append(text)
    sentences = tokenize.sent_tokenize(text)
    tmp_contents_vector = np.empty((0, 200))
    sentences_count = 0
    if len(sentences) > 50:
        # continue
        for s in sentences[:50]:
            tmp_contents_vector = np.append(tmp_contents_vector, content_vec_dic[str(s)], axis=0)
    else:
        for s in sentences:
            # print(s)
            sentences_count += 1
            tmp_contents_vector = np.append(tmp_contents_vector, content_vec_dic[str(s)], axis=0)
        while sentences_count < 50:
            tmp_contents_vector = np.append(tmp_contents_vector, np.zeros((1, 200)),
                                            axis=0)
            sentences_count += 1
    contents.append(tmp_contents_vector)
    ids.append(data_train.id[idx])
data_user = pd.read_csv('data/' + platform + '/' + platform + '_comment_id_no_ignore.tsv', sep='\t', encoding="utf-8")
f_w_r_average_leaning = io.open("data/leaning/"+ platform + "_user_average_leaning.txt", "r", encoding="utf-8")
user_leaning = {}
for line in f_w_r_average_leaning:
    line = line.rstrip('\n')
    e = line.split('\t')
    user_leaning[e[0]] = e[1]
users_leaning = []
for idx in range(data_user.id.shape[0]):
    u_ids = data_user.comment[idx].split("::")
    temp = []
    for i in range(0,200):
        if i < len(u_ids):
            temp.append(user_leaning[u_ids[0]])
        else:
            temp.append(0)
    users_leaning.append(temp)
    labels.append(data_train.label[idx])
labels = np.asarray(labels)
labels = to_categorical(labels)
del content_vec_dic
print("content finished!")


id_train, id_test, x_train, x_val, x_u_train, x_u_val, y_train, y_val = train_test_split(
    ids,
    contents,
    users_leaning,
    labels,
    # comment_info,
    test_size=VALIDATION_SPLIT,
    random_state=42,
    stratify=labels)
print(len(x_u_train))
print(len(x_u_train[0]))

print(np.array(x_train).shape, np.array(x_u_train).reshape(len(x_u_train),1,200))

content_input = Input(shape=(50, 200),dtype='float32')
user_input = Input(shape=(1,200),dtype='float32')

co = Flatten()(content_input)
uo = Flatten()(user_input)

output_dim=32
# merged_vector = keras.layers.concatenate([co, uo])
merged_vector = uo
x = Dense(output_dim, activation="relu")(merged_vector)
x = Dense(output_dim, activation="relu")(x)
x = Dense(output_dim, activation="relu")(x)
prediction = Dense(2, activation="softmax")(x)  # Here is the final output:class{0,1}

model = Model([content_input, user_input], prediction)
model.summary()

# keras 2.0.8
# tensorflow   1.14.0
from keras import optimizers

Adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
model.compile(optimizer=Adam, loss="categorical_crossentropy", metrics=['accuracy', f1_score, precision, recall])
from keras.callbacks import EarlyStopping

early_stopping = EarlyStopping(monitor='val_loss', patience=15, verbose=2)
history = model.fit([np.array(x_train), np.array(x_u_train).reshape(len(x_u_train),1,200)], np.array(y_train), epochs=50,
                    validation_split=0.1, callbacks=[early_stopping])
scores = model.evaluate([np.array(x_val), np.array(x_u_val).reshape(len(x_u_val),1,200)], np.array(y_val), verbose=0)

# history = model.fit([np.array(x_train), np.array(x_u_train)], np.array(y_train), epochs=50,
#                     validation_split=0.1, callbacks=[early_stopping])
# scores = model.evaluate([np.array(x_val), np.array(x_u_val)], np.array(y_val), verbose=0)
print(scores)
