import os
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from skimage.transform import resize
from keras.models import Model, load_model
from keras.layers import *
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K
from keras import regularizers
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import img_to_array, load_img,ImageDataGenerator
from skimage.io import imread, imshow, imread_collection, concatenate_images

# Set some parameters
im_width = 128
im_height = 128
im_chan = 3

# Build U-Net model
inputs = Input((im_height, im_width, im_chan))
s = Lambda(lambda x: x / 255)(inputs)
c1 = Conv2D(16, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(s)
c1 = Conv2D(16, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(c1)
p1 = MaxPooling2D((2, 2))(c1)

c2 = Conv2D(32, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(p1)
c2 = Conv2D(32, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(c2)
p2 = MaxPooling2D((2, 2))(c2)

c3 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(p2)
c3 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(c3)
p3 = MaxPooling2D((2, 2))(c3)

c4 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(p3)
c4 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(c4)
p4 = MaxPooling2D(pool_size=(2, 2))(c4)

c5 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(p4)
c5 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(c5)

u6 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c5)
u6 = concatenate([u6, c4])
c6 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(u6)
c6 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(c6)


u7 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c6)
u7 = concatenate([u7, c3])
c7 = Conv2D(64, (3, 3), activation='relu', padding='same',  kernel_initializer='he_normal')(u7)
c7 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(c7)


u8 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c7)
u8 = concatenate([u8, c2])
c8 = Conv2D(32, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(u8)
c8 = Conv2D(32, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(c8)


u9 = Conv2DTranspose(8, (2, 2), strides=(2, 2), padding='same', kernel_initializer='he_normal')(c8)
u9 = concatenate([u9, c1], axis=3)
c9 = Conv2D(16, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(u9)
c9 = Conv2D(16, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(c9)

outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)

model = Model(inputs=[inputs], outputs=[outputs])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[mean_iou])
model.summary() 
