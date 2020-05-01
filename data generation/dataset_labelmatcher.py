#Multiclass text classification algorithm here
from keras.models import Model
from keras.layers import Input, LSTM, Dense
import numpy as np
import pandas as pd
import string 
from sklearn.metrics.pairwise import cosine_similarity
from grammartree import getValueFromInstruction
#generates mask for sequence to sequence processing
def produceMask(instruction_label):
    #creates a mask for a string with representation [1, 0, 2] based on character frequencies
    tokenized_string = [char for char in instruction_label]
    ascii_list = list(string.ascii_lowercase)
    res = {i : instruction_label.count(i) for i in set(instruction_label)} 

    mask = [0] * len(ascii_list)

    #stores the masks in an object with the base mask and returns 
    for x in range(len(ascii_list)):
         if ascii_list[x] in res.keys():
            mask[x] = res[str(ascii_list[x])]
    
    return mask 

#levenshtein-distance is a metric used to compute distances between masks. This was adapted. 
def levenshtein_distance(a,b):
    n, m = len(a), len(b)
    if n > m:
        a,b = b,a
        n,m = m,n
        
    current = range(n+1)
    for i in range(1,m+1):
        previous, current = current, [i]+[0]*n
        for j in range(1,n+1):
            add, delete = previous[j]+1, current[j-1]+1
            change = previous[j-1]
            if a[j-1] != b[i-1]:
                change = change + 1
            current[j] = min(add, delete, change)
            
    return current[n]

#Takes in a dataset and compares the returned instruction identifer based on the part-of-speech identifier and selects the column with the most similar column. 
def getmostSimilarColumn(instruction_label, dataset):
    instruction = produceMask(instruction_label)

    data_column_masks = []
    distances = []

    #creating masks for every column value and comparing to the one for the instruction
    for element in dataset.columns:
        data_column_masks.append(produceMask(str(element)))

    for element in data_column_masks:
        distances.append(levenshtein_distance(instruction, element))

    val, idx = min((val, idx) for (idx, val) in enumerate(distances))

    #returning the column with the index of most similarity
    return dataset.columns[idx]


#exact exame to getmostSimilarColumn(). Adapted to allow for small changes for model similarity identification 
def getmostSimilarModel(model_requested, model_keys):
    masked = produceMask(model_requested)

    data_column_masks = []
    distances = []

    for element in model_keys:
        data_column_masks.append(produceMask(str(element)))
        
    for element in model_keys:
        distances.append(levenshtein_distance(masked, element))
    
    val, idx = min((val, idx) for (idx, val) in enumerate(distances))

    return model_keys[idx]

