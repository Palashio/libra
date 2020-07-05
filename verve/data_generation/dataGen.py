from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import json
import os
import urllib
import argparse
import requests


# def generate_data(word):
#     # identifies the search term and the base google address
#     searchterm = word
#     url = "https://www.google.co.in/search?q=" + \
#         searchterm + "&source=lnms&tbm=isch"

#     # identifies the chrome driver and the location to store the images
#     PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
#     DRIVER_BIN = os.path.join(PROJECT_ROOT, "chromedriver")

#     # establishes web browser and launches
#     browser = webdriver.Chrome(executable_path=DRIVER_BIN)
#     browser.get(url)
#     img_count = 0

#     # extensions for images that're accepted
#     extensions = {"jpg", "jpeg", "png", "gif"}
#     if not os.path.exists(searchterm):
#         os.mkdir(searchterm)

#     # continues to scroll down the page when it reaches the bottom
#     for _ in range(500):
#         browser.execute_script("window.scrollBy(0,10000)")

#     html = browser.page_source.split('["')
#     imges = []

#     # identifies the url of the image, if it exists, and processes it
#     # appropriately
#     for i in html:
#         if i.startswith('http') and i.split(
#                 '"')[0].split('.')[-1] in extensions:
#             imges.append(i.split('"')[0])

#     i = 0

#     # splits into only neccesary part to identify image and download
#     for link in imges:
#         filename = link.split('/')[-1]
#         urllib.urlretrieve(link, searchterm + "/" + filename)

#     browser.quit()

#     return image_preprocess(searchterm)


# def generate_set(class1, class2, class3):
#     return generate_set(class1), generate_set(class2), generate_set(class3)
