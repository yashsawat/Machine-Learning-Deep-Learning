#!/usr/bin/env python
# coding: utf-8

# In[20]:


import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# In[28]:


# Set up data generators
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    'Training data set',
    target_size=(224, 224),
    batch_size=10,
    class_mode='binary')

test_generator = test_datagen.flow_from_directory(
    'Testing data set',
    target_size=(224, 224),
    batch_size=10,
    class_mode='binary')


# In[29]:


# Build model
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])


# In[30]:


# Compile model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


# In[45]:


# Train model
model.fit(train_generator, epochs=5)


# In[43]:


# Evaluate model on test set
test_loss, test_acc = model.evaluate(test_generator)
print('Test accuracy:', test_acc)



# In[48]:


# Make predictions on test set
predictions = model.predict(test_generator)

# Print first 10 predictions
for i in range(20):
    print('Prediction:', predictions[i][0])


# In[ ]:




