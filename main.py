import tensorflow as tf
import logging
tf.get_logger().setLevel(logging.ERROR)

import numpy as np
import matplotlib.pyplot as plt
from keras.utils import image_dataset_from_directory
from keras import layers 
from keras.models import Sequential
from keras.optimizers import Adam
from keras.losses import SparseCategoricalCrossentropy
from sklearn.utils import compute_class_weight
from keras.callbacks import EarlyStopping
from keras.regularizers import l2

# In[0]

path = './Car'
imageSize = (64, 64)
batchSize = 32

Xtrain = image_dataset_from_directory(path,
                                      subset = 'training',
                                      validation_split = 0.3,
                                      image_size = imageSize,
                                      batch_size = batchSize,
                                      seed=145
                                      )

Xval = image_dataset_from_directory(path,
                                    subset = 'validation',
                                    validation_split = 0.3,
                                    image_size = imageSize,
                                    batch_size = batchSize,
                                    seed=145
                                    )

classes = Xtrain.class_names
numOfClasses = 6
print(classes)

Xtest = Xval.skip(int(len(Xval)*0.5))
Xval = Xval.take(int(len(Xval)*0.5))


print(len(Xtrain), len(Xval), len(Xtest))



# In[1]

labels = np.array([], dtype=int)
for img, lab in Xval:
    labels = np.concatenate((labels, lab.numpy()))

classWeights = compute_class_weight(class_weight='balanced',
                                    classes=np.array(range(6)),
                                    y=labels)


print('Tezine klasa: ',classWeights)

classWeightsDict = {0:classWeights[0], 1:classWeights[1], 2:classWeights[2], 3:classWeights[3], 4:classWeights[4], 5:classWeights[5]}


N = 10

plt.figure()
for img, lab in Xtrain.take(1):
    for i in range(N):
        plt.subplot(2, int(N/2), i+1)
        plt.imshow(img[i].numpy().astype('uint8'))
        plt.title(classes[lab[i]])
        plt.axis('off')

plt.show()

dataAugmentationLayers = Sequential([
    layers.RandomRotation(factor=(-0.3,0.3)),
    layers.RandomZoom(height_factor=(0.1,0.3)),
    layers.RandomBrightness(factor=[-0.2,0.2])
])

for img, val in Xtrain.take(1):
    for i in range(10):
        newImg = dataAugmentationLayers(img)
        plt.subplot(2, 5, i+1)
        plt.imshow(newImg[0].numpy().astype('uint8'))
        plt.axis('off')

plt.show()

model = Sequential([
    dataAugmentationLayers,
    layers.Rescaling(1. / 255, input_shape=(64, 64, 3)),
    layers.Conv2D(filters=32, kernel_size=(3,3), padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Dropout(0.25),
    layers.Conv2D(filters=64, kernel_size=(3,3), padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Dropout(0.25),
    layers.Conv2D(filters=128, kernel_size=(3,3), padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Dropout(0.25),
    layers.Conv2D(filters=256, kernel_size=(3,3), padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Dropout(0.25),
    layers.Flatten(),
    layers.Dense(units=512, activation='relu'),
    layers.Dense(units=256, activation='relu'),
    layers.Dense(units=128, activation='relu'),
    layers.Dense(units=numOfClasses, activation='softmax')
])

model.compile(Adam(learning_rate=0.001),
              loss=SparseCategoricalCrossentropy(),
              metrics='accuracy')

es = EarlyStopping(monitor='val_loss', mode='min', patience=20, verbose=1, restore_best_weights=True)

history = model.fit(Xtrain,
                    epochs=200,
                    validation_data=Xval,
                    verbose=1,
                    class_weight=classWeightsDict,
                    callbacks=[es]
                    )

trainingAcc = history.history['accuracy']
validationAcc = history.history['val_accuracy']

trainingLoss = history.history['loss']
validationLoss = history.history['val_loss']

plt.figure()
plt.subplot(1, 2, 1)
plt.plot(trainingAcc)
plt.plot(validationAcc)
plt.title('Tačnost')
plt.subplot(1, 2, 2)
plt.plot(trainingLoss)
plt.plot(validationLoss)
plt.title('Gubitak')
plt.show()


# In[14]:

labels = np.array([])
pred = np.array([])
for img, lab in Xtest:
    labels = np.append(labels, lab)
    pred = np.append(pred, np.argmax(model.predict(img, verbose=0), axis=1))


# In[15]:

from sklearn.metrics import accuracy_score
print('Tačnost modela je: ' + str(100*accuracy_score(labels, pred)) + '%')

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
cm = confusion_matrix(labels, pred)
cmDisplay = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
cmDisplay.plot()
plt.show()

#%%

model.save('model.h5')
