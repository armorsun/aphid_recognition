from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD, RMSprop
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Activation, Dropout, Flatten, Dense
import h5py
from keras.utils.visualize_util import plot
import matplotlib.pyplot as plt
import matplotlib.pyplot as mpl
from sklearn.cross_validation import StratifiedKFold
import numpy

# mpl.use('GTK')
seed = 7
numpy.random.seed(seed)

# kfold=StratifiedKFold()

img_width, img_height = 128, 128

training_data_dir = 'trainingData/train'
test_data_dir = 'trainingData/test'

num_training = 2009
num_test = 322
num_epoch = 250
name = 'aphids_classifier_models/model_mynet_adagrad9'
#weight_path='./vgg16_weights.h5'

model = Sequential()
model.add(ZeroPadding2D((1, 1), input_shape=(3, img_width, img_height)))
model.add(Convolution2D(8, 3, 3, activation='relu'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(8, 3, 3, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
# model.add(Dropout(0.25))

model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(16, 3, 3, activation='relu'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(16, 3, 3, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

model.add(ZeroPadding2D((2, 2)))
model.add(Convolution2D(16, 5, 5, activation='relu'))
model.add(ZeroPadding2D((2, 2)))
model.add(Convolution2D(16, 5, 5, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

#model.add(ZeroPadding2D((2, 2)))
#model.add(Convolution2D(32, 5, 5, activation='relu'))
#model.add(ZeroPadding2D((2, 2)))
#model.add(Convolution2D(32, 5, 5, activation='relu'))
#model.add(ZeroPadding2D((2, 2)))
#model.add(Convolution2D(32, 5, 5, activation='relu'))
#model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
# model.add(Dropout(0.25))

# model.add(ZeroPadding2D((1, 1)))
# model.add(Convolution2D(256, 3, 3, activation='relu'))
# model.add(ZeroPadding2D((1, 1)))
# model.add(Convolution2D(256, 3, 3, activation='relu'))
# model.add(ZeroPadding2D((1, 1)))
# model.add(Convolution2D(256, 3, 3, activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

# model.add(ZeroPadding2D((1, 1)))
# model.add(Convolution2D(256, 3, 3, activation='relu'))
# model.add(ZeroPadding2D((1, 1)))
# model.add(Convolution2D(256, 3, 3, activation='relu'))
# model.add(ZeroPadding2D((1, 1)))
# model.add(Convolution2D(256, 3, 3, activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

# model.add(ZeroPadding2D((1, 1)))
# model.add(Convolution2D(128, 3, 3, activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

# set optimizer as rmsprop
#rmsprop = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08)
model.compile(loss='binary_crossentropy', optimizer='adagrad', metrics=['accuracy'])
print(model.summary())

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    training_data_dir,
    target_size=(img_width, img_height),
    batch_size=32,
    class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(img_width, img_height),
    batch_size=32,
    class_mode='binary')

his = model.fit_generator(
    train_generator,
    samples_per_epoch=num_training,
    nb_epoch=num_epoch,
    validation_data=validation_generator,
    nb_val_samples=num_test
)

# scores = model.evaluate(x, y, verbose=0)
# print("Accuracy: %.2f%%" % (scores[1]*100))


# print(his.history.keys())

file=open('mynet_adadelta9.txt','w')
file.write('loss: {}, val_loss: {}, acc: {}, val_acc: {}\n'.format(his.history['loss'], his.history['val_loss'], his.history['acc'], his.history['val_acc']))

plt.plot(his.history['acc'])
plt.plot(his.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='lower right')
plt.savefig(name + '_accuracy.png')
plt.show()

plt.plot(his.history['loss'])
plt.plot(his.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.savefig(name + '_loss.png')
plt.show()

plot(model, to_file=name + '.png')
model_json = model.to_json()
with open(name + ".json", "w") as json_file: json_file.write(model_json)
model.save_weights(name + '.h5')
file.close()
