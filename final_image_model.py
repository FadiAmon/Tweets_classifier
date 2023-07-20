#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import random
from sklearn.model_selection import train_test_split
import cv2
import numpy as np
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import requests
import io
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from tensorflow.keras.models import load_model




# In[2]:


def load_and_visualize_images(dir_non_violence, dir_violence, num_samples):
    images = []
    labels = []
    sample_images = []
    sample_labels = []
    num_violence_samples = num_samples // 2
    num_non_violence_samples = num_samples - num_violence_samples
    
    # Load and collect non-violence images
    for filename in os.listdir(dir_non_violence):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            img = cv2.imread(os.path.join(dir_non_violence, filename))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            images.append(img)
            labels.append('non_violence')
            
            if len(sample_images) < num_non_violence_samples:
                sample_images.append(img)
                sample_labels.append('non_violence')
    
    # Load and collect violence images
    for filename in os.listdir(dir_violence):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            img = cv2.imread(os.path.join(dir_violence, filename))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            images.append(img)
            labels.append('violence')
            
            if len(sample_images) < num_samples:
                sample_images.append(img)
                sample_labels.append('violence')
    
    # Check if the collected samples are enough to visualize
    if len(sample_images) < num_samples:
        print("Insufficient number of images to visualize.")
        return np.array(images), np.array(labels)
    
    # Visualize sample images
    fig, axes = plt.subplots(1, num_samples, figsize=(12, 4))
    for i in range(num_samples):
        axes[i].imshow(sample_images[i])
        axes[i].set_title(sample_labels[i])
        axes[i].axis('off')
    plt.show()
    
    return np.array(images), np.array(labels)




def analyze_image_sizes(images):
    # Calculate image sizes
    image_sizes = np.array([img.shape[:2] for img in images])
    widths = image_sizes[:, 1]
    heights = image_sizes[:, 0]
    
    # Plot image size distribution
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.hist(widths, bins=20)
    plt.xlabel('Width')
    plt.ylabel('Count')
    plt.title('Image Width Distribution')
    
    plt.subplot(1, 2, 2)
    plt.hist(heights, bins=20)
    plt.xlabel('Height')
    plt.ylabel('Count')
    plt.title('Image Height Distribution')
    
    plt.tight_layout()
    plt.show()

def analyze_class_distribution(labels):
    # Calculate class distribution
    unique_labels, label_counts = np.unique(labels, return_counts=True)
    
    # Plot class distribution
    plt.figure(figsize=(6, 4))
    plt.bar(unique_labels, label_counts)
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.title('Class Distribution')
    plt.show()

def pie_chart_class_distribution(labels):
    # Data
    unique_labels, label_counts = np.unique(labels, return_counts=True)
    categories = unique_labels
    counts = label_counts

    # Plotting
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_facecolor('white')
    patches, texts, autotexts = ax.pie(counts, labels=categories, autopct='%1.1f%%', textprops={'color': 'white', 'fontsize': 12})

    # Set title properties
    ax.set_title('Distribution of Offensive and Non-Offensive', color='white', fontsize=14)

    # Set label colors to white
    for text in texts:
        text.set_color('white')

    # Set autopct colors to white
    for autotext in autotexts:
        autotext.set_color('white')

    # Display the pie chart
    plt.show()





# Set the directories for non-violence and violence images
dir_non_violence = r"./non_violence"
dir_violence = r"./violence"


# In[3]:


images, labels = load_and_visualize_images(dir_non_violence,dir_violence, num_samples=6)


# In[5]:


analyze_image_sizes(images)


# In[6]:


analyze_class_distribution(labels)


# In[7]:


pie_chart_class_distribution(labels)


# # Pre-processing and building the model

# In[ ]:


def get_data(data_dir):
    

    data_dir = data_dir
    categories = ['violence', 'non_violence']

    image_paths = []
    labels = []

    # Iterate over the categories
    for category in categories:
        category_path = os.path.join(data_dir, category)
        
        # Check if the directory exists
        if not os.path.exists(category_path):
            print(f"Directory '{category_path}' does not exist.")
            continue
        
        # Iterate over the images in the category folder
        for image_name in os.listdir(category_path):
            # Construct the image path
            image_path = os.path.join(category_path, image_name)
            
            # Append the image path and corresponding label to the lists
            image_paths.append(image_path)
            labels.append(category)
        
    return image_paths, labels



    
image_paths, labels=get_data(r'C:\Users\user\Desktop\imdata')


# In[ ]:


# Preprocess the images
def pre_proccessing(image_paths):
    target_size = (128, 128)
    preprocessed_images = []

    # Iterate over the image paths
    for image_path in image_paths:

        image = cv2.imread(image_path)
        image = cv2.resize(image, target_size)
        image = image / 255.0

        # Append the preprocessed image to the list
        preprocessed_images.append(image)

    return preprocessed_images


preprocessed_images=pre_proccessing(image_paths)
preprocessed_images = np.array(preprocessed_images)
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)
onehot_encoder = OneHotEncoder(sparse=False)
labels = onehot_encoder.fit_transform(labels.reshape(-1, 1))
# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(preprocessed_images, labels, test_size=0.2, random_state=42)


# In[ ]:


def get_model():
    # Define the model architecture
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='softmax'))

    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

model=get_model()
history = model.fit(X_train, y_train, batch_size=32, epochs=10, validation_data=(X_test, y_test))
#model.save('final_image_model.tf')


# # Classification with saved model

# In[8]:


def predict_image(model_path,image_path=None,url=None):
    # Load the image
    target_size = (128, 128)
    model_path=model_path
    if  image_path:
        image = cv2.imread(image_path)
        
        image = cv2.resize(image, target_size)
    elif  url:
        response = requests.get(url)
        image= Image.open(io.BytesIO(response.content))
        image = image.resize(target_size)
    else:
        raise "Provide image path or url"
    # Preprocess the image
    image = np.array(image)
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    # Load the model
    model = load_model(model_path)
    # Make the prediction
    predictions = model.predict(image)
    return predictions[0][0]


# In[9]:


predict_image("final_image_model.tf",url="https://pbs.twimg.com/media/Fvh8mqXXgAI9Xfq?format=jpg&name=small")

