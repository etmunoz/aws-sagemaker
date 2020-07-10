import argparse, os
import numpy as np

import tensorflow as tf
import keras
from keras import backend as K
from keras.models import Sequential
#from keras.optimizers import SGD
#from keras.callbacks import Callback, EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import multi_gpu_model



class MnistModel(tf.keras.Model):
    def __init__(self, input_shape=None, batch_norm_axis=None, num_classes=None):
        super(MnistModel, self).__init__()
        
        
        # 1st convolution block
        self.l1_conv = tf.keras.layers.Conv2D(64, kernel_size=(3,3), padding='same', input_shape=input_shape)
        self.l1_batchn = tf.keras.layers.BatchNormalization(axis=batch_norm_axis)
        self.l1_activation = tf.keras.layers.Activation('relu')
        self.l1_maxpooling = tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=2)
        
        # 2nd convolution block
        self.l2_conv = tf.keras.layers.Conv2D(128, kernel_size=(3,3), padding='valid')
        self.l2_batchn = tf.keras.layers.BatchNormalization(axis=batch_norm_axis)
        self.l2_activation = tf.keras.layers.Activation('relu')
        self.l2_maxpooling = tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=2)


        # Fully connected block
        self.l3_flatten = tf.keras.layers.Flatten()
        self.l3_dense = tf.keras.layers.Dense(512)
        self.l3_activation = tf.keras.layers.Activation('relu')
        self.l3_dropout = tf.keras.layers.Dropout(0.3)

        # Output layer
        self.l_output = tf.keras.layers.Dense(num_classes, activation='softmax')
    
    def build(self, input_shape):
        # No weight to train.
        super(MnistModel, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        x = self.l1_conv(x)
        x = self.l1_batchn(x)
        x = self.l1_activation(x)
        x = self.l1_maxpooling(x)
        
        x = self.l2_conv(x)
        x = self.l2_batchn(x)
        x = self.l2_activation(x)
        x = self.l2_maxpooling(x)
        
        x = self.l3_flatten(x)
        x = self.l3_dense(x)
        x = self.l3_activation(x)
        x = self.l3_dropout(x)
        
        return self.l_output(x)

if __name__ == '__main__':
    print ("tensorflow version:", tf.__version__)
    parser = argparse.ArgumentParser()

    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--learning-rate', type=float, default=0.01)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--gpu-count', type=int, default=os.environ['SM_NUM_GPUS'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--training', type=str, default=os.environ['SM_CHANNEL_TRAINING'])
    parser.add_argument('--validation', type=str, default=os.environ['SM_CHANNEL_VALIDATION'])
    
    args, _ = parser.parse_known_args()
    
    epochs     = args.epochs
    lr         = args.learning_rate
    batch_size = args.batch_size
    gpu_count  = args.gpu_count
    model_dir  = args.model_dir
    training_dir   = args.training
    validation_dir = args.validation
    
    os.makedirs(model_dir, exist_ok=True)
    x_train = np.load(os.path.join(training_dir, 'training.npz'))['image']
    y_train = np.load(os.path.join(training_dir, 'training.npz'))['label']
    x_val  = np.load(os.path.join(validation_dir, 'validation.npz'))['image']
    y_val  = np.load(os.path.join(validation_dir, 'validation.npz'))['label']
    
    # input image dimensions
    img_rows, img_cols = 28, 28

    # Tensorflow needs image channels last, e.g. (batch size, width, height, channels)
    K.set_image_data_format('channels_last')  
    print(K.image_data_format())

    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
        x_val = x_val.reshape(x_val.shape[0], 1, img_rows, img_cols)
        input_shape = (None, 1, img_rows, img_cols)
        batch_norm_axis=1
    else:
        # channels last
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_val = x_val.reshape(x_val.shape[0], img_rows, img_cols, 1)
        input_shape = (None, img_rows, img_cols, 1)
        batch_norm_axis=-1

    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_val.shape[0], 'test samples')
    
    # Normalize pixel values
    x_train  = x_train.astype('float32')
    x_val    = x_val.astype('float32')
    x_train /= 255
    x_val   /= 255
    
    # Convert class vectors to binary class matrices
    num_classes = 10
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_val   = keras.utils.to_categorical(y_val, num_classes)
    
    model = MnistModel(input_shape=input_shape, batch_norm_axis=batch_norm_axis, num_classes=num_classes)
    model.build(input_shape)
    print(model.summary())

    if gpu_count > 1:
        model = multi_gpu_model(model, gpus=gpu_count)
        
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=tf.keras.optimizers.SGD(lr=lr, decay=1e-6, momentum=0.9, nesterov=True),
                  metrics=['accuracy'])
    
    datagen = ImageDataGenerator(
     rotation_range=20,
     width_shift_range=0.2,
     height_shift_range=0.2,
     horizontal_flip=True)

    datagen.fit(x_train)
    model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
                    validation_data=(x_val, y_val), 
                    epochs=epochs,
                    steps_per_epoch=len(x_train) / batch_size,
                    verbose=1)
    
    score = model.evaluate(x_val, y_val, verbose=0)
    print('Validation loss    :', score[0])
    print('Validation accuracy:', score[1])
    
    # save Keras model for Tensorflow Serving
#     sess = K.get_session()
#     tf.saved_model.save(
#         model,
#         os.path.join(model_dir, 'model/1')
#         )
    #os.makedirs(model_dir, exist_ok=True)
    #model.save(os.path.join(model_dir, '1'))
    tf.saved_model.save(model, os.path.join(model_dir, "1"))
    