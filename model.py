import csv
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Load the data.
csv_path = '../data_joystick/driving_log.csv'
img_path = '../data_joystick/IMG/'
lines = []
with open(csv_path) as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

images = []
measurements = []
for line in lines:
    print(line)
    source_path = line[0]
    filename = source_path.split('/')[-1]
    current_path = img_path + filename
    image = cv2.imread(current_path)
    print(image.size)
    images.append(np.asarray(image))
    measurement = float(line[3])
    measurements.append(measurement)

plt.hist(measurements)
plt.savefig('measuremts.png')



X_train = np.array(images)
y_train = np.array(measurements)
print("input shape:", X_train.shape)
print("image shpae:", X_train[0].shape)

# Train the model
from keras.models import Sequential, model_from_json
from keras.layers.core import Dense, Activation, Flatten, Dropout, Lambda
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.advanced_activations import ELU
from keras.regularizers import l2, activity_l2
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, Callback

model = Sequential()

    # Normalize
model.add(Lambda(lambda x: x/127.5 - 1.0,input_shape=(160,320,3)))

    # Add three 5x5 convolution layers (output depth 24, 36, and 48), each with 2x2 stride
model.add(Convolution2D(24, 5, 5, subsample=(2, 2), border_mode='valid', W_regularizer=l2(0.001)))
model.add(ELU())
model.add(Convolution2D(36, 5, 5, subsample=(2, 2), border_mode='valid', W_regularizer=l2(0.001)))
model.add(ELU())
model.add(Convolution2D(48, 5, 5, subsample=(2, 2), border_mode='valid', W_regularizer=l2(0.001)))
model.add(ELU())

    #model.add(Dropout(0.50))
    
    # Add two 3x3 convolution layers (output depth 64, and 64)
model.add(Convolution2D(64, 3, 3, border_mode='valid', W_regularizer=l2(0.001)))
model.add(ELU())
model.add(Convolution2D(64, 3, 3, border_mode='valid', W_regularizer=l2(0.001)))
model.add(ELU())

    # Add a flatten layer
model.add(Flatten())

    # Add three fully connected layers (depth 100, 50, 10), tanh activation (and dropouts)
model.add(Dense(100, W_regularizer=l2(0.001)))
model.add(ELU())
    #model.add(Dropout(0.50))
model.add(Dense(50, W_regularizer=l2(0.001)))
model.add(ELU())
    #model.add(Dropout(0.50))
model.add(Dense(10, W_regularizer=l2(0.001)))
model.add(ELU())
    #model.add(Dropout(0.50))

    # Add a fully connected output layer
model.add(Dense(1))

    # Compile and train the model, 
    #model.compile('adam', 'mean_squared_error')
model.compile(optimizer=Adam(lr=1e-4), loss='mse')

model.fit(X_train, y_train, validation_split=0.2, shuffle=True)

model.save('model.h5')
