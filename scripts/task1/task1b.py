import re
import csv
import pandas as pd
import numpy as np
import textdistance
from fuzzywuzzy import fuzz

##load in the data
abt=pd.read_csv('data/abt.csv',encoding = 'ISO-8859-1').astype(str)
buy=pd.read_csv('data/buy.csv',encoding = 'ISO-8859-1').astype(str)

abt.replace(np.nan, "", inplace = True)
abt.sort_values(by="name", inplace = True)
abt.drop('price', 1, inplace = True)
abt.rename(columns={'idABT':'idAbt', 'name': 'name_abt', 'description': 'description_abt'}, inplace = True)

buy.replace(np.nan, "", inplace = True)
buy.sort_values(by="name", inplace = True)
buy.drop('price', 1, inplace = True)
buy.rename(columns={'name': 'name_buy', 'description': 'description_buy', 'manufacturer':'manufacturer_buy'}, inplace = True)


for column in abt.columns:
    abt[column] = abt[column].str.lower().replace("[\'\"\\\{\}\[\]\(\)\|_#/.,:\$&@-]", '', regex=True)
    abt[column] = abt[column].str.strip().replace('( + )', ' ', regex=True)
    
for column in buy.columns:
    buy[column] = buy[column].str.lower().replace('[\'\"\\\{\}\[\]\(\)\|_#/.,:\$&@-]', '', regex=True)
    buy[column] = buy[column].str.strip().replace('( + )', ' ', regex=True)
    

# this function assumes that the first word in a product name is the brand for the product
def get_brand(product_name):
    strings = product_name.split()
    brand = strings[0]
    return brand

# this function returns true if two brands are a match
def match_brand(brand1, brand2):
    if textdistance.levenshtein(brand1, brand2) <= 2:
        return True
    if (brand1 in brand2) or (brand2 in brand1):
        return True
    return False


# add the brands as a feature to the abt and buy DataFrames
brands_abt = []
for row in abt['name_abt'].values:
    row_search = get_brand(row)
    brands_abt.append(row_search)
abt['brand_abt'] = brands_abt

brands_buy = []
for row in buy['name_buy'].values:
    row_search = get_brand(row)
    brands_buy.append(row_search)
buy['brand_buy'] = brands_buy


# dictionary to keep track of the block numbers
block_dict = {}
block_count = 0


with open('output/abt_blocks.csv', 'w') as file:
    writer = csv.writer(file)
    writer.writerow(['block_key', 'product_id'])
    for row in abt[['idAbt', 'brand_abt']].values:
        idAbt = row[0]
        brand = row[1]

        # check to see if the brand is already in the dictionary
        if brand in block_dict.keys():
            writer.writerow([block_dict[brand], idAbt])
        # otherwise make a new item in the dictionary
        else:
            block_count += 1
            block_dict[brand] = block_count
            writer.writerow([block_dict[brand], idAbt])
        
with open('output/buy_blocks.csv', 'w') as file:
    writer = csv.writer(file)
    writer.writerow(['block_key', 'product_id'])
    for row in buy[['idBuy', 'brand_buy', 'manufacturer_buy']].values:
        idBuy = row[0]
        brand = row[1]
        manufacturer = row[2]
        
        # check to see if the brand is already in the dictionary
        if brand in block_dict.keys():
            writer.writerow([block_dict[brand], idBuy])
            continue
        found_block = False
        # otherwise check to see if the manufacturer is already in the dictionary
        for key in block_dict.keys():
            if match_brand(key, manufacturer):
                writer.writerow([block_dict[key], idBuy])
                found_block = True
                break                    
        # otherwise make a new item in the dictionary      
        if not found_block:
            block_count += 1
            block_dict[brand] = block_count
            writer.writerow([block_dict[brand], idBuy])

