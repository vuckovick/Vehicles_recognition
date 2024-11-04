#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import logging
tf.get_logger().setLevel(logging.ERROR)


# # Vežbe 6 - Kovolucione neuralne mreže
# 
# Konvolucione neuralne mreže su pogodne za klasifikaciju podataka koji su oblika mreže, tj. slika. 
# 
# U sebi sadrže konvolucione slojeve koji služe za ekstrakciju obeležja i pogodni su zato što ne gledaju celu sliku, već samo mali deo. Zahvaljujući tome, broj parametara jednog sloja je relativno mali. Jedan konvolucioni sloj se sastoji od više filtara, a svaki od njih ima ulogu da izvuče neka druga obeležja. 
# 
# U kombinaciji sa konvolucionim slojevima se koriste i pooling slojevi koji služe za selekciju obeležja. Pooling slojevi smanjuju prostorna dimenzije slike, i na taj način smanjuju broj podataka koji će se koristiti u narednom sloju.
# 
# Na kraju CNN se nalaze potpuno povezani slojevi koji služe za klasifikaciju.
# 
# ![cnn.png](attachment:18ddadf4-91e9-4f59-bcaa-f7da2ffaec35.png)

# In[5]:


import numpy as np
import matplotlib.pyplot as plt

main_path = 'Car/'

img_size = (64, 64)
batch_size = 32


# In[6]:


from keras.utils import image_dataset_from_directory
from sklearn.utils import class_weight

Xtrain = image_dataset_from_directory(main_path, 
                                      subset='training', 
                                      validation_split=0.2,
                                      image_size=img_size,
                                      batch_size=batch_size,
                                      seed=123)

Xval = image_dataset_from_directory(main_path,
                                    subset='validation',
                                    validation_split=0.2,
                                    image_size=img_size,
                                    batch_size=batch_size,
                                    seed=123)

classes = Xtrain.class_names
#%%

#class_weight ubaci
labels = []
for images, label in Xtrain:
    labels.extend(label.numpy().tolist())

Ytraining = []
for i in labels:
    Ytraining.append(classes[i])

#%%
weights = class_weight.compute_class_weight(class_weight='balanced', 
                                            classes=classes, 
                                            y=Ytraining)

print(classes)


# In[5]:


N = 10

plt.figure()
for img, lab in Xtrain.take(1):
    for i in range(N):
        plt.subplot(2, int(N/2), i+1)
        plt.imshow(img[i].numpy().astype('uint8'))
        plt.title(classes[lab[i]])
        plt.axis('off')


# In[6]:


from keras import layers
from keras import Sequential

data_augmentation = Sequential(
  [
    layers.RandomFlip("horizontal", input_shape=(img_size[0], img_size[1], 3)),
    layers.RandomRotation(0.25),
    layers.RandomZoom(0.1),
  ]
)


# In[7]:


N = 10

plt.figure()
for img, lab in Xtrain.take(1):
    plt.title(classes[lab[0]])
    for i in range(N):
        aug_img = data_augmentation(img)
        plt.subplot(2, int(N/2), i+1)
        plt.imshow(aug_img[0].numpy().astype('uint8'))
        plt.axis('off')


# In[8]:


from keras import Sequential
from keras import layers
from keras.optimizers import Adam
from keras.losses import SparseCategoricalCrossentropy  
from keras.regularizers import l2

num_classes = len(classes)

#L2 regularizacija i dodaj neke slojeve

model = Sequential([
    data_augmentation,
    layers.Rescaling(1./255, input_shape=(64, 64, 3)),
    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(128, 3, padding='same', activation='relu'),
    layers.Dropout(0.2),
    layers.Flatten(),
    layers.Dense(128, activation='relu', kernel_regularizer=l2()),
    layers.Dense(64, activation='relu', kernel_regularizer=l2()),
    layers.Dense(32, activation='relu', kernel_regularizer=l2()),
    layers.Dense(num_classes, activation='softmax')
])

model.summary()

model.compile(Adam(learning_rate=0.001), 
              loss=SparseCategoricalCrossentropy(),
              metrics='accuracy')


# In[9]:


history = model.fit(Xtrain,
                    epochs=50,
                    validation_data=Xval,
                    verbose=1,
                    class_weight={0:weights[0], 1:weights[1], 2:weights[2], 3:weights[3], 4:weights[4], 5:weights[5]},
                    batch_size=1)


# In[13]:


acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure()
plt.subplot(121)
plt.plot(acc)
plt.plot(val_acc)
plt.title('Accuracy')
plt.subplot(122)
plt.plot(loss)
plt.plot(val_loss)
plt.title('Loss')
plt.show()


# In[14]:


labels = np.array([])
pred = np.array([])
for img, lab in Xval:
    labels = np.append(labels, lab)
    pred = np.append(pred, np.argmax(model.predict(img, verbose=0), axis=1))    


# In[15]:


from sklearn.metrics import accuracy_score
print('Tačnost modela je: ' + str(100*accuracy_score(labels, pred)) + '%')

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
cm = confusion_matrix(labels, pred, normalize='true')
cmDisplay = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
cmDisplay.plot()
plt.show()

