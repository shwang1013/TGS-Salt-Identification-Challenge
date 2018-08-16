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
im_chan = 1
path_train = '../data/train/'

# Remove those black files with file size =107 bytes.
train_ids = next(os.walk(path_train+"/Auge_NoColor_image/"))[2]
i = 0
for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):
    statinfo = os.stat(path_train + '/Auge_NoColor_image/' + id_)
    if statinfo.st_size < 108:
        continue
    i += 1
i += 1

# Get and resize train images and masks
X_train = np.zeros((i, im_height, im_width, im_chan), dtype=np.uint8)
Y_train = np.zeros((i, im_height, im_width, 1), dtype=np.bool)

print('Getting and resizing train images and masks ... ')
i = 0
for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):
    path = path_train
    # Remove black data
    statinfo = os.stat(path + '/Auge_NoColor_image/' + id_)
    # delete the black image
    # os.st_size: os can get the information of the file, the black_pic's size is 107kb, so delete them
    if statinfo.st_size < 108:
        continue
    i += 1
    print(path + '/Auge_NoColor_image/' + id_, ":", statinfo.st_size)
    img = load_img(path + '/Auge_NoColor_image/' + id_)
    x = img_to_array(img)[:, :, 1]
    x = resize(x, (128, 128, 1), mode='constant', preserve_range=True)
    X_train[i] = x
    mask = img_to_array(load_img(path + '/Auge_mask_data/' + id_))[:, :, 1]
    Y_train[i] = resize(mask, (128, 128, 1), mode='constant', preserve_range=True)

print("Total training records are ", len(X_train))
print('Done!')

# Define IoU metric
def mean_iou(y_true, y_pred):
    prec = []
    for t in np.arange(0.5, 1.0, 0.05):
        y_pred_ = tf.to_int32(y_pred > t)
        score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 2)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([up_opt]):
            score = tf.identity(score)
        prec.append(score)
    return K.mean(K.stack(prec), axis=0)

# Build U-Net model
inputs = Input((im_height, im_width, im_chan))
s = Lambda(lambda x: x / 255)(inputs)
c1 = Conv2D(16, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(s)
c1 = BatchNormalization(epsilon=1e-06, axis=3, momentum=0.9)(s)
c1 = Conv2D(16, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(c1)
c1 = BatchNormalization(epsilon=1e-06, axis=3, momentum=0.9)(c1)
p1 = MaxPooling2D((2, 2))(c1)

c2 = Conv2D(32, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(p1)
c2 = BatchNormalization(epsilon=1e-06, axis=3, momentum=0.9)(c2)
c2 = Conv2D(32, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(c2)
c2 = BatchNormalization(epsilon=1e-06, axis=3, momentum=0.9)(c2)
p2 = MaxPooling2D((2, 2))(c2)

c3 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(p2)
c3 = BatchNormalization(epsilon=1e-06, axis=3, momentum=0.9)(c3)
c3 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal',kernel_regularizer=regularizers.l2(0.0001))(c3)
c3 = BatchNormalization(epsilon=1e-06, axis=3, momentum=0.9)(c3)
p3 = MaxPooling2D((2, 2))(c3)

c4 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(p3)
c4 = BatchNormalization(epsilon=1e-06, axis=3, momentum=0.9)(c4)
c4 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal',kernel_regularizer=regularizers.l2(0.0001))(c4)
c4 = BatchNormalization(epsilon=1e-06, axis=3, momentum=0.9)(c4)
p4 = MaxPooling2D(pool_size=(2, 2))(c4)

c5 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(p4)
c5 = BatchNormalization(epsilon=1e-06, axis=3, momentum=0.9)(c5)
c5 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal',kernel_regularizer=regularizers.l2(0.0001))(c5)
c5 = BatchNormalization(epsilon=1e-06, axis=3, momentum=0.9)(c5)

u6 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c5)
u6 = concatenate([u6, c4])
c6 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(u6)
c6 = BatchNormalization(epsilon=1e-06, axis=3, momentum=0.9)(c6)
c6 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal',kernel_regularizer=regularizers.l2(0.0001))(c6)
c6 = BatchNormalization(epsilon=1e-06, axis=3, momentum=0.9)(c6)

u7 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c6)
u7 = concatenate([u7, c3])
c7 = Conv2D(64, (3, 3), activation='relu', padding='same',  kernel_initializer='he_normal' ,kernel_regularizer=regularizers.l2(0.0001))(u7)
c7 = BatchNormalization(epsilon=1e-06, axis=3, momentum=0.9)(c7)
c7 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(c7)
c7 = BatchNormalization(epsilon=1e-06, axis=3, momentum=0.9)(c7)

u8 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c7)
u8 = concatenate([u8, c2])
c8 = Conv2D(32, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(u8)
c8 = BatchNormalization(epsilon=1e-06, axis=3, momentum=0.9)(c8)
c8 = Conv2D(32, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(c8)
c8 = BatchNormalization(epsilon=1e-06, axis=3, momentum=0.9)(c8)

u9 = Conv2DTranspose(8, (2, 2), strides=(2, 2), padding='same', kernel_initializer='he_normal')(c8)
u9 = concatenate([u9, c1], axis=3)
c9 = Conv2D(16, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(u9)
c9 = BatchNormalization(epsilon=1e-06, axis=3, momentum=0.9)(c9)
c9 = Conv2D(16, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(c9)
c9 = BatchNormalization(epsilon=1e-06, axis=3, momentum=0.9)(c9)
c9 = Dropout(0.5)(c9)  

outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)

model = Model(inputs=[inputs], outputs=[outputs])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[mean_iou])
model.summary()       

earlystopper = EarlyStopping(patience=5, verbose=1)
checkpointer = ModelCheckpoint('5.h5', verbose=1, save_best_only=True)
results = model.fit(X_train, Y_train, validation_split=0.1, batch_size=32, epochs=30, callbacks=[earlystopper, checkpointer])
