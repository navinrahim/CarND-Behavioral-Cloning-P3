#Importing Packages
import csv
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten,Dense,Activation,Convolution2D,Cropping2D,MaxPooling2D,Lambda,Dropout
from keras.callbacks import EarlyStopping,ModelCheckpoint
import sklearn
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

#Importing the rows from the driving_log.csv file which inlcudes the links to the images and the steering angles for them
samples=[]
with open('../Data/driving_log.csv') as csvfile:
    reader=csv.reader(csvfile)
    for line in reader:
        samples.append(line)

#Splitting the data into training and validation sets
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

#Generator Code
def generator(samples, batch_size):
    '''
    Takes in the input sample, shuffles them, divide them into batches, apply augmentation and return the images.
    Input images are converted from BGR to RGB, since data is read through opencv
    Left camera and right camera images are taken and a correction of 0.25 is applied to their corresponding angles
    Also, left and right camera images are flipped and added to the dataset
    '''
    num_samples = len(samples)
    while 1: 
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                
                #Take center camera images and angles
                name = '../Data/IMG/'+batch_sample[0].split('/')[-1]
                center_image = cv2.imread(name)
                center_image=cv2.cvtColor(center_image, cv2.COLOR_BGR2RGB)
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)
                
                #Take left camera images and add an angle of 0.25
                name = '../Data/IMG/'+batch_sample[1].split('/')[-1]
                left_image = cv2.imread(name)
                left_image=cv2.cvtColor(left_image, cv2.COLOR_BGR2RGB)
                left_angle = float(batch_sample[3])+0.25
                images.append(left_image)
                angles.append(left_angle)
                
                #Take right camera images and subtract an angle of 0.25
                name = '../Data/IMG/'+batch_sample[2].split('/')[-1]
                right_image = cv2.imread(name)
                right_image=cv2.cvtColor(right_image, cv2.COLOR_BGR2RGB)
                right_angle = float(batch_sample[3])-0.25
                images.append(right_image)
                angles.append(right_angle)
                
                #Horizontal flipped image from the left camera
                horizontal_left_image = cv2.flip(left_image,1)
                images.append(horizontal_left_image)
                angles.append(left_angle*-1)
                
                #Horizontal flipped image from the right camera
                horizontal_right_image = cv2.flip(right_image,1)
                images.append(horizontal_right_image)
                angles.append(right_angle*-1)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)
			
#Generators to fetch training and validation samples
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

#Define the network
model=Sequential()

#Normalising the data
model.add(Lambda(lambda x: (x /255)-0.5,input_shape=(160,320,3)))

#Cropping unwanted pixels off the images
model.add(Cropping2D(cropping=((50,20),(0,0))))

#Adding layers to the network
model.add(Convolution2D(6,5,5))
model.add(Activation('relu'))
model.add(MaxPooling2D((2,2)))

model.add(Convolution2D(16,5,5))
model.add(Activation('relu'))
model.add(MaxPooling2D((2,2)))

model.add(Flatten())

model.add(Dense(120))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(84))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(1))

#Compile the model with a loss function of mean squared error and an Adam optimizer is used
model.compile(loss='mse', optimizer='adam',metrics=['accuracy'])

#Print the summary of the model
model.summary()

#Adding Callback functions for stopping the training early if a best model is found and to save the best model
early_stopping=EarlyStopping(patience=5)
check_point=ModelCheckpoint("model.h5",save_best_only=True)

#Training the model
model.fit_generator(train_generator, samples_per_epoch= len(train_samples), validation_data=validation_generator,nb_val_samples=len(validation_samples), nb_epoch=30, callbacks=[early_stopping,check_point])

