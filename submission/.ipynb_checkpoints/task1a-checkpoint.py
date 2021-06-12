import re
import csv
import pandas as pd
import numpy as np
import textdistance
from fuzzywuzzy import fuzz

##load in the data
abt=pd.read_csv('abt_small.csv',encoding = 'ISO-8859-1').astype(str)
buy=pd.read_csv('buy_small.csv',encoding = 'ISO-8859-1').astype(str)

abt.replace(np.nan, "", inplace = True)
abt.drop('price', 1, inplace = True)
abt.rename(columns={'idABT':'idAbt', 'name': 'name_abt', 'description': 'description_abt'}, inplace = True)

buy.replace(np.nan, "", inplace = True)
buy.drop('price', 1, inplace = True)
buy.rename(columns={'name': 'name_buy', 'description': 'description_buy', 'manufacturer':'manufacturer_buy'}, inplace = True)


for column in abt.columns:
    abt[column] = abt[column].str.lower().replace("[\'\"\\\{\}\[\]\(\)\|_#/.,:+\$&@-]", '', regex=True)
    abt[column] = abt[column].str.strip().replace('( + )', ' ', regex=True)
    
for column in buy.columns:
    buy[column] = buy[column].str.lower().replace('[\'\"\\\{\}\[\]\(\)\|_#/.,:+\$&@-]', '', regex=True)
    buy[column] = buy[column].str.strip().replace('( + )', ' ', regex=True)


# this function returns the first and last words of a product name as the brand and serial No. This is an assumption.
def get_brand_and_serialNo(product_name):
    strings = product_name.split()
    brand = strings[0]
    serialNo = strings[-1]
    return [brand, serialNo]

# find the assumed brands and serialNos in for each product 
brands_abt = []
serialNos_abt = []
for row in abt['name_abt'].values:
    row_search = get_brand_and_serialNo(row)
    brands_abt.append(row_search[0])
    serialNos_abt.append(row_search[1])
abt['brand_abt'] = brands_abt
abt['serialNo_abt'] = serialNos_abt

brands_buy = []
serialNos_buy = []
for row in buy['name_buy'].values:
    row_search = get_brand_and_serialNo(row)
    brands_buy.append(row_search[0])
    serialNos_buy.append(row_search[1])
buy['brand_buy'] = brands_buy
buy['serialNo_buy'] = serialNos_buy



# This function compares 3 strings, returns true if 2 brands (or a manufacturer) are a match
def match_brand(brand_abt, brand_buy, manufacturer_buy):
    if textdistance.levenshtein(brand_abt, brand_buy) <= 1:
        return True
    if (textdistance.levenshtein(brand_abt, manufacturer_buy) <= 1) or (brand_abt in manufacturer_buy) or (manufacturer_buy in brand_abt):
        return True

    
# This function returns true if 2 serialNo's are a match
def match_serialNo(serialNo_abt, serialNo_buy):
    if textdistance.levenshtein(serialNo_abt, serialNo_buy) == 0:
        return True
    if (serialNo_abt in serialNo_buy) or (serialNo_buy in serialNo_abt):
        return True
    return False


# This function searches the given search_list for a serialNo, returns true if the list contains that serialNo
def search_serialNo(serialNo_abt, search_list: list):
    for string in search_list:
        if textdistance.levenshtein(serialNo_abt, string) == 0:
            return True
        # The shortest serialNo found in data sets is 4 chars, any less and there's too much noise
        if len(string) >= 4:
            if (string in serialNo_abt) or (serialNo_abt in string):
                return True
    return False

# This function assumes that strings with at least 8 characters and contains a digit are serialNo's
def find_serialNo(string):
    strings = string.split()
    output = []
    for string in strings:
        # Usually serialNos are at least 8 chars, if a word contains a digit and >= 8 chars
        # There's a very high probability that it is a serialNo
        if len(string) >= 8:
            for char in string:
                if char.isdigit():
                    output.append(string)
                    break
    return output

# This function checks the 2 given lists for a match in any serialNo's between the lists
def cross_check_serialNos(list1, list2):
    for serial1 in list1:
        for serial2 in list2:
            if match_serialNo(serial1, serial2):
                return True
    return False


found_abt = []
with open('task1a.csv', 'w') as task1a:
    writer = csv.writer(task1a)
    writer.writerow(['idAbt', 'idBuy'])
    
    # This first loop assumes the first word in a product name is the brand for the product
    # It also assumes the last word in that product name is the serialNo for the product
    # It will find all the cases where there is both a match in brand and serialNo
    for row1 in abt[['idAbt', 'brand_abt', 'serialNo_abt', 'name_abt', 'description_abt']].values:
        idAbt = row1[0]
        brand_abt = row1[1]
        serialNo_abt = row1[2]
        name_abt = row1[3]
        description_abt = row1[4]
        
        matchID_buy = ''
        for row2 in buy[['idBuy', 'brand_buy', 'manufacturer_buy', 'serialNo_buy', 'name_buy', 'description_buy']].values:
            idBuy = row2[0]
            brand_buy = row2[1]
            manufacturer_buy = row2[2]
            serialNo_buy = row2[3]
            name_buy = row2[4]
            description_buy = row2[5]
            
            try:
                # if the brands match
                if match_brand(brand_abt, brand_buy, manufacturer_buy):
                    # check if the assumed serialNo's are a perfect match
                    if match_serialNo(serialNo_abt, serialNo_buy):
                        matchID_buy = idBuy
                        writer.writerow([idAbt, idBuy])
                        found_abt.append(idAbt)
                        break
                    # if the assumed ones don't match, check whether abt's serialNo is anywhere inside the name and 
                    # description of 'buy' product
                    search_list = name_buy.split() + description_buy.split()
                    if search_serialNo(serialNo_abt, search_list):
                        matchID_buy = idBuy
                        writer.writerow([idAbt, idBuy])
                        found_abt.append(idAbt)
                        break
            except Exception as e:
                pass
            
        # if there was a match, remove this item from buy DataFrame
        if matchID_buy:
            buy.drop(buy[buy.idBuy == matchID_buy].index, inplace=True)
                    
    # remove all items matched from abt DataFrame
    for id_found in found_abt:
        abt.drop(abt[abt.idAbt == id_found].index, inplace=True)

    
    ## Second loop to search for edge cases
    for row1 in abt[['idAbt', 'brand_abt', 'serialNo_abt', 'name_abt', 'description_abt']].values:
        idAbt = row1[0]
        brand_abt = row1[1]
        serialNo_abt = row1[2]
        name_abt = row1[3].replace(brand_abt, '').replace(serialNo_abt, '')
        description_abt = row1[4].replace(serialNo_abt, '').replace(brand_abt, '')
        
        # find all strings that look like serialNo's in abt product
        found_serialNos_abt = find_serialNo(name_abt + " " + description_abt)
        
        matchID_buy = ''
        for row2 in buy[['idBuy', 'brand_buy', 'manufacturer_buy', 'serialNo_buy', 'name_buy', 'description_buy']].values:
            idBuy = row2[0]
            brand_buy = row2[1]
            manufacturer_buy = row2[2]
            serialNo_buy = row2[3]
            name_buy = row2[4].replace(serialNo_buy, '').replace(brand_buy, '').replace('( + )', ' ')
            description_buy = row2[5].replace(serialNo_buy, '').replace(brand_buy, '').replace('( + )', ' ')
            
            # find all strings that look like serialNo's in buy product
            found_serialNos_buy = find_serialNo(name_buy + " " + description_buy)

            try:
                if match_brand(brand_abt, brand_buy, manufacturer_buy):
                    
                    search_list_abt = name_abt.split() + description_abt.split()
                    search_list_buy = name_buy.split() + description_buy.split()
                    # Check whether we have any matching serialNo's
                    if cross_check_serialNos(found_serialNos_abt, found_serialNos_buy):
                        matchID_buy = idBuy
                        writer.writerow([idAbt, idBuy])
                        break

                    # Check to see if the shortened version (by 1 char) of the serialNo matches
                    if (match_serialNo(serialNo_abt[:-1], serialNo_buy) or match_serialNo(serialNo_abt, serialNo_buy[:-1]) or
                       search_serialNo(serialNo_abt[:-1], search_list_buy)):
                        matchID_buy = idBuy
                        writer.writerow([idAbt, idBuy])
                        break    
                        
                    str1 = name_buy
                    str2 = name_abt + " " + description_abt
                    # Check to see if the names and descriptions match at least 90%
                    # Description of buy is left out as it is usually uninformative
                    if fuzz.token_set_ratio(str1, str2) >= 90 or fuzz.token_set_ratio(name_abt, name_buy) >= 90:
                        matchID_buy = idBuy
                        writer.writerow([idAbt, idBuy])
                        break
                        
                    str1 = str1.replace(' ', '')
                    str2 = str2.replace(' ', '')
                    if fuzz.partial_ratio(str1, str2) >= 95:
                        matchID_buy = idBuy
                        writer.writerow([idAbt, idBuy])
                        break
                    
            except Exception as e:
                pass
        
        # if there was a match, remove this item from buy DataFrame
        if matchID_buy:
            buy.drop(buy[buy.idBuy == matchID_buy].index, inplace=True)





