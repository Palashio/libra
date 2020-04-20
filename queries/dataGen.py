from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import json
import os
import urllib
import argparse

searchterm = 'Okocha'#input your search item here
url = "https://www.google.co.in/search?q="+searchterm+"&source=lnms&tbm=isch"
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
DRIVER_BIN = os.path.join(PROJECT_ROOT, "chromedriver")

browser = webdriver.Chrome(executable_path = DRIVER_BIN)
browser.get(url)
img_count = 0
extensions = { "jpg", "jpeg", "png", "gif" }
if not os.path.exists(searchterm):
    os.mkdir(searchterm)

for _ in range(500):
    browser.execute_script("window.scrollBy(0,10000)")
    
html = browser.page_source.split('["')
imges = []
for i in html:
    if i.startswith('http') and i.split('"')[0].split('.')[-1] in extensions:
        imges.append(i.split('"')[0])
print(imges)