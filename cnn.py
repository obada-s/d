from keras.models import Sequential
from keras.layers import Convolution2D,\
    MaxPooling2D,Flatten,Dense

#CNN
#32 features (3 X 3)
#each image (64 X 64) , 3 colored image
#Convolution
cls = Sequential()
cls.add(Convolution2D(32,3,3,
                      input_shape=(64,64,3),
                      activation='relu'))
#Max Pooling
cls.add(MaxPooling2D(pool_size=(2,2)))
#Flattening
cls.add(Flatten())
#Layers
cls.add(Dense(units=128,activation='relu'))
cls.add(Dense(units=1,activation='sigmoid'))
cls.compile(optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy'])
#Fitting Image to CNN
from keras.preprocessing.image \
    import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        'dataset/training_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

test_generator = test_datagen.flow_from_directory(
        'dataset/test_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

cls.fit_generator(
        train_generator,
        steps_per_epoch=4000,
        epochs=10,
        validation_data=test_generator,
        validation_steps=1000)

# Single Prediction
import numpy as np
from keras.preprocessing import image

test_img=image.load_img('dataset\single_prediction\cat_or_dog_1.jpg',target_size=(64,64))
test_img=image.img_to_array(test_img)

#reshape
# بعمل نفس عمل Flatten
test_img=np.expand_dims(test_img,axis=0)

#Prediction
result=cls.predict(test_img)

# leable
print(test_generator.class_indices)

if result [0][0] ==1:
    print('Dog')
else:
    print('cat')

