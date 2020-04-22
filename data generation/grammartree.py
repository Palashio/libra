import os 
from textblob import TextBlob
import pandas as pd 

def getValueFromInstruction(sent):
    blob = TextBlob(sent)
    blob.parse()

    tags = blob.tags

    decoded = ""

    truth = False
    for x in range(len(tags)):
        if "JJ" in tags[x]:
            truth = True

    if truth:
        try:
            for x in range (len(tags)):
                if "JJ" in tags[x]:
                    q = x + 1
                    decoded += sent.split()[x] + "_"
                    while("VBN" in tags[q] or "VBG" in tags[q] or "NN" in tags[q] or "NNS" in tags[q] or "RB" in tags[q] or ("NNS" in tags[q] and "IN" in tags[q + 1])):
                        decoded += sent.split()[q] + "_"
                        if("IN" in tags[q + 1]):
                            decoded += sent.split()[q + 1] + "_"
                            q += 2
                            continue
                        q += 1
                        if q >= len(tags):
                            break

        except:
            print("")
            

    if not truth:
        try:
            for x in range (len(tags)):
                if x < len(tags) - 1:
                    if "IN" in tags[x + 1]:
                        decoded += sent.split()[x] + "_" 
                        decoded += sent.split()[x + 1] + "_" 
                        x = x + 2
                        continue
                if "NN" in tags[x] or "NNS" in tags[x] or "RB" in tags[x] or "VBG" in tags[x] or "VBN" in tags[x]:
                    decoded += sent.split()[x] + "_" 
                else:   
                    continue
        except:
            print(x)
            print(tags[x])
            print("Please try re-typing your sentence")

    decoded = decoded[:-1]  

    if len(sent.split()) == 2:
        decoded = sent.split()[1]

    
    return decoded

