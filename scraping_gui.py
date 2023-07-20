#!/usr/bin/env python
# coding: utf-8

# In[1]:


from selenium import webdriver
import re
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager 
from selenium.webdriver.chrome.service import Service
import time
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import NoSuchElementException
import tkinter as tk
from tkinter import messagebox
from tkinter import ttk
from PIL import ImageTk, Image
import requests
from io import BytesIO
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords # to remove the stopwords
import requests
from PIL import Image
import io
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from selenium.webdriver.chrome.options import Options
import cv2
import csv
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.preprocessing.text import Tokenizer


# In[2]:


def get_driver():
    cookies1 = [
        #secret
    ]




    cookies2 = [
        #secret
    ]



    cookies3 = [
        #secret
    ]


    all_cookies = cookies1 + cookies2 + cookies3

    # Set path to chromedriver executable
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))

    # navigate to the website's homepage
    driver.get('https://twitter.com/home')
    # clear existing cookies
    driver.delete_all_cookies()
    # add all the cookies
    cookies = all_cookies
    for cookie in cookies:
        driver.add_cookie(cookie)


    driver.get('https://twitter.com/explore')
    # refresh the page and the user should be logged in
    driver.refresh()
    return driver


# In[3]:


def return_tweets_content_url(posts_divs):
    tweets_content=[]
    image_urls=[]
    for each_post_div in posts_divs:
        try:
            result = each_post_div.find_element(By.CSS_SELECTOR, 'div[data-testid="tweetText"]')
            spans = result.find_elements(By.CSS_SELECTOR, 'span')
            text = ""
            for span in spans:
                text += span.text
            tweets_content.append(re.sub(r'\n+', ' ', text).strip())

        except NoSuchElementException:
            # Handle the case when the tweet content is not found
            tweets_content.append(None)

        try:
            image_div = each_post_div.find_element(By.CSS_SELECTOR, 'div[aria-label="Image"]')
            image = image_div.find_element(By.TAG_NAME, 'img')
            image_urls.append(image.get_attribute('src'))
        except NoSuchElementException:
            # Handle the case when there is no image for the post
            image_urls.append(None)
        

    tweets_url = tuple(zip(tweets_content, image_urls))
    return tweets_url


# In[4]:


def get_query_data(search_query):
    driver=get_driver()
    # Wait for the search input box to be visible
    search_input = WebDriverWait(driver, 10).until(EC.visibility_of_element_located((By.CSS_SELECTOR, 'input[aria-label="Search query"]')))

    # Enter the search query
    search_input.send_keys(search_query)

    # Submit the search query
    search_input.submit()
    
    latest_tab = WebDriverWait(driver, 10).until(EC.visibility_of_element_located((By.XPATH, '//span[text()="Latest"]/parent::div/parent::div/parent::a')))
    latest_tab.click()

    time.sleep(5)

    # Wait for the search results to load
    posts_divs = WebDriverWait(driver, 10).until(EC.visibility_of_all_elements_located((By.CSS_SELECTOR, 'div[data-testid="cellInnerDiv"]')))

    tweets_url = return_tweets_content_url(posts_divs)


    scroll_increment=3000
    scrolls_amount=3
    for i in range(scrolls_amount):
        driver.execute_script(f"window.scrollBy(0, {scroll_increment});")
        time.sleep(2)  # Adjust the delay as needed
        posts_divs = WebDriverWait(driver, 10).until(EC.visibility_of_all_elements_located((By.CSS_SELECTOR, 'div[data-testid="cellInnerDiv"]')))
        scrolled_tweets_url=return_tweets_content_url(posts_divs)
        tweets_url=set(list(tweets_url) + list(scrolled_tweets_url))
    return tweets_url


# In[5]:


text_model = load_model("./final_text_model.tf")
image_model = load_model("./final_image_model.tf")


# In[6]:


def preprocess(tweet):
    tweet = re.sub(r'@\w+', '', tweet)
    
    # Remove hashtags
    tweet = re.sub(r'#\w+', '', tweet)
    
    # Remove retweet indicators
    tweet = re.sub(r'RT', '', tweet)
    
    # Remove non-letter characters
    cleaned_tweet = re.sub('[^a-zA-Z]', ' ', tweet)
    
    return cleaned_tweet


# In[7]:


def read_tweets_from_csv(csv_file):
    tweets = []
    with open(csv_file, 'r', encoding='utf-8') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            if row:
                tweets.append(row[0])
    return tweets

def classify_text(model,tweets):
    tokenizer = Tokenizer()
    clean_tweet = read_tweets_from_csv(r"new_data.csv")  # Read tweets from CSV file
    tokenizer.fit_on_texts(clean_tweet)
    tweets=preprocess(tweets)
    tweet_sequence = tokenizer.texts_to_sequences([tweets])
    padded_tweet_sequence = pad_sequences(tweet_sequence, maxlen=1403)
    prediction = model.predict(padded_tweet_sequence)
    return prediction[0][0]


# In[8]:


tweets="I don’t condone the usage of the words nigger and or nigga - in life - and also in art, unless they are being used to tell a story.#Art  #Life  #NWord  #Nigga  #Nigger"
classify_text(text_model,tweets)


# In[9]:


def classify_image(model, image_path=None, url=None):
    # Load the image
    target_size = (128, 128)
    if image_path:
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        image = cv2.resize(image, target_size)
    elif url:
        response = requests.get(url)
        image = Image.open(io.BytesIO(response.content))
        image = image.resize(target_size)
    else:
        raise ValueError("Provide image path or URL")

    # Convert PIL Image to numpy array
    image = np.array(image)

    # Check if the image has an alpha channel (4th dimension)
    if image.shape[-1] == 4:
        image = image[..., :3]  # Remove the alpha channel

    # Preprocess the image
    image = image / 255.0
    image = np.expand_dims(image, axis=0)

    # Make the prediction
    predictions = model.predict(image)
    return predictions[0][0]


# # Example usage
# image_url = "https://media.istockphoto.com/id/972902038/photo/the-love-of-best-friends.jpg?s=612x612&w=0&k=20&c=wno6K_BsRklIWqiUWEsVJBI7gqW4pJr1iGBnIwZR5mM="
# predicted_class = classify_image(image_model, image_url)
# print("Predicted class:", predicted_class)


# #hate #violence #racism

# In[10]:


def search_tweets():
    query = query_entry.get()
    if not query:
        messagebox.showerror("Error", "Please enter a search query.")
        return

    # Get the tweets and image URLs
    tweets_url = get_query_data(query)

    # Clear the existing tweets in the frame
    for widget in tweets_frame.winfo_children():
        widget.destroy()

    # Display the tweets in the scrollable frame
    for i, (tweet, url) in enumerate(tweets_url, start=1):
        tweet_label = ttk.Label(tweets_frame, text=f"{i}) {tweet}", wraplength=500)
        tweet_label.pack(anchor='w', pady=20)

        if url is not None:
            try:
                response = requests.get(url)
                image_data = response.content
                image = Image.open(BytesIO(image_data))
                image = image.resize((400, 400), Image.ANTIALIAS)
                photo = ImageTk.PhotoImage(image)
                image_label = ttk.Label(tweets_frame)
                image_label.image = photo  # Store reference to the photo object
                image_label.configure(image=photo)
                image_label.pack()

                # Create "View full image" button
                view_button = ttk.Button(tweets_frame, text="View full image", command=lambda url=url: view_full_image(url))
                view_button.pack()
            except:
                # Handle image loading errors
                image_error_label = ttk.Label(tweets_frame, text="Error loading image")
                image_error_label.pack()
        else:
            no_image_label = ttk.Label(tweets_frame, text="No image for tweet")
            no_image_label.pack()

        classify_button = ttk.Button(tweets_frame, text="Classify", command=lambda tweet=tweet, url=url: classify_tweet(tweet, url))
        classify_button.pack()

def classify_tweet(tweet, url):
    # Function to open a new window and display the prediction result
    def display_result(prediction_text=None, prediction_image=None):
        result_window = tk.Toplevel()
        result_window.title("Tweet Classification Result")
        result_window.geometry("800x600")

        fig = plt.figure(figsize=(12, 6))
        labels = ["Violence/Hate Speech", "Non Violence/Hate Speech"]
        colors = ["red", "green"]

        if prediction_text is not None and prediction_image is not None:
            # Both text and image available, display two pie charts
            ax1 = fig.add_subplot(1, 2, 1)
            label_size_text = [prediction_text*100, (1-prediction_text)*100]
            ax1.pie(label_size_text, colors=colors, autopct='%1.1f%%')
            ax1.set_title("Text Sentiment", fontsize=16)

            ax2 = fig.add_subplot(1, 2, 2)
            label_size_image = [prediction_image*100, (1-prediction_image)*100]
            ax2.pie(label_size_image, colors=colors, autopct='%1.1f%%')
            ax2.set_title("Image Sentiment", fontsize=16)

            # Equalize the aspect ratio of both pie charts
            ax1.set_aspect('equal')
            ax2.set_aspect('equal')

            # Adjust spacing between the subplots
            fig.subplots_adjust(wspace=0.4)

            # Create a legend
            fig.legend(labels, loc='upper left')

            # Calculate average violence probability
            avg_probability = (prediction_text + prediction_image) / 2

        elif prediction_text is not None:
            # Only text available, display single pie chart for text
            label_size = [prediction_text*100, (1-prediction_text)*100]
            plt.pie(label_size, colors=colors, autopct='%1.1f%%')
            plt.title("Text Sentiment", fontsize=16)

            # Create a legend
            plt.legend(labels, loc='upper left')

            # Set average violence probability based on text prediction
            avg_probability = prediction_text

        elif prediction_image is not None:
            # Only image available, display single pie chart for image
            label_size = [prediction_image*100, (1-prediction_image)*100]
            plt.pie(label_size, colors=colors, autopct='%1.1f%%')
            plt.title("Image Sentiment", fontsize=16)

            # Create a legend
            plt.legend(labels, loc='upper left')

            # Set average violence probability based on image prediction
            avg_probability = prediction_image

        plt.tight_layout()

        # Add text based on average violence probability
        if avg_probability > 0.5:
            text = f"This tweet does contain violence/hate speech with {avg_probability*100:.2f}% certainty"
        else:
            text = f"This tweet doesn't contain violence/hate speech with {(1-avg_probability)*100:.2f}% certainty"

        plt.figtext(0.5, 0.02, text, ha="center", fontsize=14)

        canvas = FigureCanvasTkAgg(fig, master=result_window)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        plt.show()

    # Check if tweet and url are None
    if tweet is None and url is None:
        messagebox.showerror("Error", "Tweet text and image URL are missing.")
        return

    if tweet is not None:
        # Perform text classification and get text prediction
        prediction_text = classify_text(text_model, tweet)
    else:
        prediction_text = None

    if url is not None:
        # Perform image classification and get image prediction
        prediction_image = classify_image(image_model, url=url)
    else:
        prediction_image = None

    # Call the function to display the result in a new window
    display_result(prediction_text, prediction_image)



# Function to view the full image in its original size
def view_full_image(url):
    response = requests.get(url)
    image_data = response.content
    image = Image.open(BytesIO(image_data))
    image.show()

# Create the main window
root = tk.Tk()
root.title("Twitter Search")
root.geometry("800x600")

# Create the query input field
query_label = ttk.Label(root, text="Search Query:")
query_label.pack(anchor='w')

query_entry = ttk.Entry(root, width=50)
query_entry.pack()

# Create the search button
search_button = ttk.Button(root, text="Search", command=search_tweets)
search_button.pack(pady=10)

# Create the scrollable frame for displaying tweets
canvas = tk.Canvas(root)
canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

scrollbar = ttk.Scrollbar(root, orient=tk.VERTICAL, command=canvas.yview)
scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

tweets_frame = ttk.Frame(canvas)

canvas.configure(yscrollcommand=scrollbar.set)
canvas.create_window((0, 0), window=tweets_frame, anchor='nw')

def configure_canvas(event):
    canvas.configure(scrollregion=canvas.bbox("all"), width=780, height=500)

tweets_frame.bind("<Configure>", configure_canvas)

scrollbar.configure(command=canvas.yview)

# Call configure_canvas initially to configure the canvas size
configure_canvas(None)

root.mainloop()

