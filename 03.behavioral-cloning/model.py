import csv
import cv2
import os
from numpy import source
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import sklearn

from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Lambda, Convolution2D, Cropping2D

def ShiftImageRandomly(image, steering_angle, x_shift_range, y_shift_range):
    #get random values for horizontal and vertical shifts (in pixels)
    x_shift = x_shift_range * (np.random.uniform() - 0.5)
    y_shift = y_shift_range * (np.random.uniform() - 0.5)
    # calculate new steering angle.  0.4 is the steering angle adjustment for maximum horizontal shift to one side (x_shift_range)
    new_steering_angle = steering_angle + x_shift/x_shift_range * 0.4
    # use Affine transformation for image shifting
    Trans_M = np.float32([[1, 0, x_shift],[0, 1, y_shift]])
    shifted_image = cv2.warpAffine(image,Trans_M,(320,160))
    return shifted_image, new_steering_angle

def AddImageAndSteeringAngle(images, steering_angles, current_training_data_dir, recorded_image_file_path, steering_angle):
    current_image_file_path = os.path.join(current_training_data_dir, 'IMG', recorded_image_file_path.split('/')[-1])
    if(not os.path.isfile(current_image_file_path)):
        print ("File {} does not exist. Skipping... ", current_image_file_path)
    else:
        image = cv2.cvtColor(cv2.imread(current_image_file_path), cv2.COLOR_BGR2RGB)

        ### add original image and steering angle
        images.append(image)
        steering_angles.append(steering_angle)

        ### add randomly shift original image
        shifted_image, shifted_angle = ShiftImageRandomly(image, steering_angle, x_shift_range=50, y_shift_range=40)
        images.append(shifted_image)
        steering_angles.append(shifted_angle)

        ### add horizontally flipped image and inverted angle to emulate reverse driving direction case
        flipped_image = cv2.flip(image, 1)
        images.append(flipped_image)
        steering_angles.append(steering_angle * -1.0)


def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images = []
            steering_angles = []
            for batch_sample in batch_samples:
                steering_angle = float(batch_sample[1][3])
                include_zero_angle_image = np.random.randint(0,2)
                if (steering_angle != 0.0 or include_zero_angle_image):
                    # first column contains left camera image file path
                    current_training_data_dir = os.path.join(training_data_root_path, batch_sample[0])
                    AddImageAndSteeringAngle(images, steering_angles, current_training_data_dir, batch_sample[1][0], steering_angle)
                    # second column contains file path of image from left camera image
                    AddImageAndSteeringAngle(images, steering_angles, current_training_data_dir, batch_sample[1][1], steering_angle + stering_angle_correction)
                    # third column contains file path of image from right camera
                    AddImageAndSteeringAngle(images, steering_angles, current_training_data_dir, batch_sample[1][2], steering_angle - stering_angle_correction)

            X_train = np.array(images)
            y_train = np.array(steering_angles)
            yield sklearn.utils.shuffle(X_train, y_train) 

def DisplaySteeringAnglesHistohram(title, angles):
    plt.hist(angles, bins=80, rwidth=0.7, histtype='bar')
    plt.xlim(xmin=-0.7, xmax=0.7)
    plt.title(title)
    plt.xlabel("Steering angle")
    plt.ylabel("Number of samples")
    plt.show()


training_data_root_path = './data/'
training_data_dirs = [ directory for directory in os.listdir(training_data_root_path) if os.path.isdir(os.path.join(training_data_root_path, directory)) ]
print ('Using data samples from following folders:', training_data_dirs)

stering_angle_correction = 0.25         # steering angle adjustment value for side camera image
data_sample_augumentation_factor = 9    # 1 data sample = 3 cameras x (1 + augumentation_types_number)

data = []
steer_angles=[]
for data_dir in training_data_dirs:
    with open(os.path.join(training_data_root_path, data_dir, 'driving_log.csv')) as csvfile:
        reader = csv.reader(csvfile)
        for table_row in reader:
            data.append((data_dir, table_row))
            steer_angles.append(float(table_row[3]))

print("Number of track points for training: ", len(data))
training_data, validation_data = train_test_split(data, test_size=0.2)

training_data_generator = generator(training_data, batch_size=32)
validation_data_generator = generator(validation_data, batch_size=32)

print("Number of imgaes in training dataset: ", len(training_data) * data_sample_augumentation_factor)
print("Number of imgaes in validation dataset: ", len(validation_data) * data_sample_augumentation_factor)

DisplaySteeringAnglesHistohram("Distribution of steering angles in initial training dataset", steer_angles)

model = Sequential()
# crop 40 pixel from the top (useless image payload) and 20 pixel from the bottom (hood)
model.add(Cropping2D(cropping=((40,20), (0,0)), input_shape=(160,320,3)))
# normalize pixel values in the image
model.add(Lambda(lambda x: x / 255.0 - 0.5))
# NVIDIA network model
model.add(Convolution2D(24,5,5, border_mode='valid', subsample=(2,2), activation='relu'))
model.add(Convolution2D(36,5,5, border_mode='valid', subsample=(2,2), activation='relu'))
model.add(Convolution2D(48,5,5, border_mode='valid', subsample=(2,2), activation='relu'))
model.add(Convolution2D(64,3,3, border_mode='valid', subsample=(1,1), activation='relu'))
model.add(Convolution2D(64,3,3, border_mode='valid', subsample=(1,1), activation='relu'))
model.add(Flatten())
model.add(Dense(1164))
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
training_history = model.fit_generator(training_data_generator, validation_data = validation_data_generator, nb_epoch = 5, samples_per_epoch = len(training_data) * data_sample_augumentation_factor, nb_val_samples = len(validation_data)  * data_sample_augumentation_factor)

### plot the training and validation loss for each epoch
plt.plot(training_history.history['loss'])
plt.plot(training_history.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()

# store the model
model.save('model.h5') 