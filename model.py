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

samples=[]
with open('../Data/driving_log.csv') as csvfile:
    reader=csv.reader(csvfile)
    for line in reader:
        samples.append(line)

train_samples, validation_samples = train_test_split(samples, test_size=0.2)

#Generator Code
def generator(samples, batch_size):
    num_samples = len(samples)
    while 1: 
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                
                #Take center camera images
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
                angles.append(right_angle)#Horizontal flipped image from the left camera
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
			
#Generators to fetch training and validation datasets
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

#Define the network
model=Sequential()

#Normalising the data
model.add(Lambda(lambda x: (x /255)-0.5,input_shape=(160,320,3)))

#Cropping unwanted pixels off the images
model.add(Cropping2D(cropping=((50,20),(0,0))))
          
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

#load previous model weights and initialise
#model.load_weights("model.h5")

model.compile(loss='mse', optimizer='adam',metrics=['accuracy'])
model.summary()

early_stopping=EarlyStopping(patience=5)
check_point=ModelCheckpoint("model.h5",save_best_only=True)
          
model.fit_generator(train_generator, samples_per_epoch= len(train_samples), validation_data=validation_generator,nb_val_samples=len(validation_samples), nb_epoch=30, callbacks=[early_stopping,check_point])


