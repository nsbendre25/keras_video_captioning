from __future__ import print_function
import numpy as np
import pandas as pd
import cv2
import pickle
from time import time

from keras.layers import Input, Dense, Conv2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D, Dropout, Flatten, merge, concatenate, Reshape
from keras.models import Model
from keras.regularizers import l2
from keras.optimizers import SGD, Adadelta, Nadam
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten, Reshape, Permute
from keras.layers.convolutional import Conv3D, MaxPooling3D, ZeroPadding3D, Cropping2D, Cropping3D
from keras.models import Sequential
from keras.layers.wrappers import TimeDistributed
from keras.layers.normalization import BatchNormalization
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
from sklearn.feature_extraction.text import TfidfVectorizer

# cut texts after this number of words (among top max_features most common words)
vector_size = 100
batch_size = 10
epochs = 150
frames_in_each_sample = 66
frame_shape = (75, 175)  # width, height
channels = 3

global v
train = pd.read_csv('train_list_final_UPDATED.csv')['description'].values
valid = pd.read_csv('valid_list_final_UPDATED.csv')['description'].values
v = TfidfVectorizer(stop_words='english',
                        analyzer=u'word', max_features=vector_size)
v.fit(np.concatenate((train, valid)))
with open("Vectorizor.pkl", 'wb') as f:
    pickle.dump(v, f)
def preprocessing(video_path):
    """Get Video from video_path"""
    cap = cv2.VideoCapture(video_path)
    vid = []
    while True:
        ret, img = cap.read()
        if not ret:
            break
        vid.append(cv2.resize(img, frame_shape, cv2.INTER_AREA))

    vid = np.array(vid, dtype=np.float32).reshape(-1,
                                                  frame_shape[1],
                                                  frame_shape[0],
                                                  channels)
    new_vid = vid[:frames_in_each_sample]
    return new_vid


def get_x_batch(bdf):
    """Get a batch of videos from locations in the data frame"""
    x_train = []
    for index, row in bdf.iterrows():
        new_vid = preprocessing(row['video path'])
        x_train.append(new_vid)

    x_train = np.asarray(x_train).reshape(-1,
                                          frames_in_each_sample,
                                          frame_shape[1],
                                          frame_shape[0],
                                          channels)
    return x_train


def generator(filepath='train_list_final_UPDATED.csv', batch_size=batch_size):
    """Generate Batches with randomness"""
    global v
    df = pd.read_csv(filepath)
    TOTAL_SAMPLES = df.shape[0]
    #v = TfidfVectorizer(stop_words='english',
                        #analyzer=u'word', max_features=vector_size)
    Y_values = v.transform(df['description']).toarray().reshape(-1, vector_size)
    while True:
        start = np.random.randint(TOTAL_SAMPLES - batch_size)
        end = start + batch_size
        bdf = df.iloc[start:end]
        x_train = get_x_batch(bdf)
        y_train = Y_values[start:end, :]
        yield x_train, y_train


model = Sequential()
# 1st layer group
model.add(Conv3D(64, (3, 3, 3), activation="relu", padding="same",
                 name='conv1', strides=(1, 1, 1), input_shape=(frames_in_each_sample, frame_shape[1],
                                                               frame_shape[0], channels)))
model.add(MaxPooling3D(pool_size=(1, 2, 2), strides=(
    1, 2, 2), padding="valid", name='pool1'))
# 2nd layer group
model.add(Conv3D(128, (3, 3, 3), activation="relu",
                 padding="same", name='conv2', strides=(1, 1, 1)))
model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(
    2, 2, 2), padding="valid", name='pool2'))
# 3rd layer group
model.add(Conv3D(256, (3, 3, 3), activation="relu",
                 padding="same", name='conv3a', strides=(1, 1, 1)))
model.add(Conv3D(256, (3, 3, 3), activation="relu",
                 padding="same", name='conv3b', strides=(1, 1, 1)))
model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(
    2, 2, 2), padding="valid", name='pool3'))
# 4th layer group
model.add(Conv3D(512, (3, 3, 3), activation="relu",
                 padding="same", name='conv4a', strides=(1, 1, 1)))
model.add(Conv3D(512, (3, 3, 3), activation="relu",
                 padding="same", name='conv4b', strides=(1, 1, 1)))
model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(
    2, 2, 2), padding="valid", name='pool4'))
# 5th layer group
model.add(Conv3D(512, (3, 3, 3), activation="relu",
                 padding="same", name='conv5a', strides=(1, 1, 1)))
model.add(Conv3D(512, (3, 3, 3), activation="relu",
                 padding="same", name='conv5b', strides=(1, 1, 1)))
model.add(ZeroPadding3D(padding=(0, 1, 1)))
model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(
    2, 2, 2), padding="valid", name='pool5'))

model.add(TimeDistributed(Flatten(), name="Flatten"))
# LSTM Layer
model.add(LSTM(units=vector_size*4, dropout=0.5, return_sequences=True, name="LSTM_1"))
model.add(LSTM(units=vector_size*3, dropout=0.5, return_sequences=True, name="LSTM_2"))
model.add(LSTM(units=vector_size*2, dropout=0.5, return_sequences=True, name="LSTM_3"))
model.add(LSTM(units=vector_size, dropout=0.5, name="LSTM_4"))
print(model.summary())

sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
adadelta = Adadelta(lr=1.0, rho=0.95, epsilon=1e-08, decay=0.0)
nadam = Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004)

model.compile(loss='binary_crossentropy',
              optimizer=sgd, metrics=['accuracy'])
tensorboard = TensorBoard(log_dir="./logs/{}".format(time()), batch_size=batch_size,
                          write_graph=True, write_grads=True, write_images=True)
checkpoint = ModelCheckpoint(filepath='./checkpoints/video_captioning'+'{epoch:02d}'+'{val_loss:.2f}.hdf5', monitor='val_loss',
                             verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=5)
early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='auto')


t0 = time()
model.fit_generator(generator(), steps_per_epoch=3700, callbacks=[tensorboard, checkpoint, early_stop], epochs=epochs, max_queue_size=50, validation_data=generator(
    filepath='valid_list_final_UPDATED.csv'), validation_steps=500)
t1 = time()
print("Training completed in " + str((t1 - t0) / 3600) + " hours")

model.save_weights('video_captioning_weights.h5')
print("Weights Saved Successfully...")
model.save('video_captioning_model.h5')
print("Model Saved Successfully...")
