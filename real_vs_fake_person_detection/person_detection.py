from keras.models import Sequential
from keras.layers import Conv2D, Flatten, Dense, MaxPool2D
import numpy as np
from keras.preprocessing import image
from keras.callbacks import EarlyStopping, ModelCheckpoint
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from keras.utils import load_img, img_to_array

try:
    from tensorflow.python.util import module_wrapper as deprecation
except ImportError:
    from tensorflow.python.util import deprecation_wrapper as deprecation
deprecation._PER_MODULE_WARNING_LIMIT = 0

nbatch = 128
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10.,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory('dataset/training',
                                                 target_size=(128, 128),
                                                 batch_size=nbatch,
                                                 class_mode='binary')

test_set = test_datagen.flow_from_directory('dataset/test',
                                            target_size=(128, 128),
                                            batch_size=nbatch,
                                            class_mode='binary')


h1 = plt.hist(training_set.classes, bins=range(0, 3),
              alpha=0.8, color='blue', edgecolor='black')
h2 = plt.hist(test_set.classes,  bins=range(0, 3),
              alpha=0.8, color='red', edgecolor='black')
plt.ylabel('# of instances')
plt.xlabel('Class')


for X, y in training_set:
    print(X.shape, y.shape)
    plt.figure(figsize=(16, 16))
    for i in range(16):
        plt.subplot(4, 4, i+1)
        plt.axis('off')
        plt.title('Label: ')
        img = np.uint8(255*X[i, :, :, 0])
        plt.imshow(img, cmap='gray')
    break


model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=(128, 128, 3)))

model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Conv2D(64, kernel_size=(3, 3),
                 activation='relu'))

model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Conv2D(128, kernel_size=(3, 3),
                 activation='relu'))

model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Flatten())

model.add(Dense(activation="relu",
                units=256))

model.add(Dense(activation="sigmoid",
                units=1))

model.summary()

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

callbacks_list = [
    EarlyStopping(monitor='val_loss', patience=10),
    ModelCheckpoint(filepath='model_checkpoint.hdf5',
                    monitor='val_loss', save_best_only=True, mode='max'),
]

history = model.fit_generator(
    training_set,
    steps_per_epoch=80,
    epochs=10,
    validation_data=test_set,
    validation_steps=28,
    callbacks=callbacks_list
)

training_set.class_indices

plt.figure(figsize=(16, 6))
plt.subplot(1, 2, 1)
nepochs = len(history.history['loss'])
# plt.plot(range(nepochs), history.history['loss'],     'r-', label='train')
# plt.plot(range(nepochs), history.history['val_loss'], 'b-', label='test')
# plt.legend(prop={'size': 20})
# plt.ylabel('loss')
# plt.xlabel('# of epochs')
# plt.subplot(1, 2, 2)
# # plt.plot(range(nepochs), history.history['acc'],     'r-', label='train')
# # plt.plot(range(nepochs), history.history['val_acc'], 'b-', label='test')
# plt.legend(prop={'size': 20})
# plt.ylabel('accuracy')
# plt.xlabel('# of epochs')


def ImagePrediction(loc):
    test_image = load_img(loc, target_size=(128, 128))
    plt.axis('off')
    plt.imshow(test_image)
    test_image = img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    result = model.predict(test_image)
    if result[0][0] == 1:
        predictions = 'Fake'
    else:
        predictions = 'Real'
    print('Prediction: ', predictions)
