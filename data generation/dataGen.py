from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import os
import urllib

def generate_data(word):
    # identifies the search term and the base google address
    searchterm = word
    url = "https://www.google.co.in/search?q=" + \
        searchterm + "&source=lnms&tbm=isch"

    # identifies the chrome driver and the location to store the images
    PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
    DRIVER_BIN = os.path.join(PROJECT_ROOT, "chromedriver")

    # establishes web browser and launches
    browser = webdriver.Chrome(executable_path=DRIVER_BIN)
    browser.get(url)
    img_count = 0
    #adding the training and test folders with subdirectories
    #for the searchterm
    sets=['training_set','test_test']
    for i in sets:
        if not os.path.exists(i):
            if not os.path.exists(i+'/'+searchterm):
                os.makedirs(i+'/'+searchterm)
        else:
            if not os.path.exists(i+'/'+searchterm):
                os.makedirs(i+'/'+searchterm)
    
    # extensions for images that're accepted
    extensions = {"jpg", "jpeg", "png", "gif"}

    # continues to scroll down the page when it reaches the bottom
    for _ in range(500):
        browser.execute_script("window.scrollBy(0,10000)")

    html = browser.page_source.split('["')
    imges = []

    # identifies the url of the image, if it exists, and processes it
    # appropriately
    for i in html:
        if i.startswith('http') and i.split(
                '"')[0].split('.')[-1] in extensions:
            imges.append(i.split('"')[0])

    # splits into only neccesary part to identify image and download
    for link in imges:
        filename = link.split('/')[-1]
        ++img_count
        if img_count>(10000*0.2):
            urllib.urlretrieve(link, 'test_set/'+searchterm + "/" + filename)
        else:
            urllib.urlretrieve(link, 'training_set/'+searchterm + "/" + filename)
    browser.quit()


def generate_set(class1, class2, class3):
    return generate_set(class1), generate_set(class2), generate_set(class3)

