import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import to_categorical
import h5py

## Data Processing

# File Names
TRAIN_FILE = '../data/train'
TEST_FILE = '../data/test'

# Unpickling data
def unpickle(file):
    import cPickle
    f = open(file, 'r')
    d = cPickle.load(f)
    f.close()
    return d

# Get Training features and labels
def get_train_data(prep_img=True):
    return get_cifar100_data(TRAIN_FILE, prep_img)

# Get Testing features and labels
def get_test_data(prep_img=True):
    return get_cifar100_data(TEST_FILE, prep_img)

# Get Data
def get_cifar100_data(file_ip, prep_img=True, normalize=True, one_hot=True, num_labels=100):
    d = unpickle(file_ip)
    data = np.array(d['data']).astype('float32')
    labels = np.array(d['fine_labels'])
    data, labels = data_shuffle(data, labels)
    if prep_img:
        data = prep_images(data)
    if one_hot:
        labels = to_categorical(labels, num_labels)
    if normalize:
        data = data.astype('float32')
        data = data / 255.
    return data, labels

def data_shuffle(data, labels):
    labels = labels[:,np.newaxis]
    c = np.column_stack((data, labels))
    for _ in xrange(10):
        np.random.shuffle(c)
    d, l = c[:,:-1], c[:,-1]
    return d, l

# Convert vector to image
def prep_images(images, shape=(32,32,3)):
    return images.reshape(-1,shape[0], shape[1], shape[2])

# Get Data
x_train, y_train = get_train_data()
x_test, y_test = get_test_data()

## Building Model

model = Sequential()

model.add(Conv2D(32, (5, 5), activation='elu', padding='same', input_shape=(32, 32, 3)))
model.add(Conv2D(32, (3, 3), activation='elu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), activation='elu', padding='same'))
model.add(Conv2D(64, (3, 3), activation='elu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, (3, 3), activation='elu'))
model.add(Conv2D(128, (3, 3), activation='elu'))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(1024, activation='elu'))
model.add(Dropout(0.5))
model.add(Dense(100, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=100, epochs=30, validation_split=0.2)

model.save_weights('mod_C_weights_new_30')

print 'Testing'
print model.evaluate(x_test, y_test, batch_size=50, verbose=1)
