import os
import json
import pandas as pd
from pandas.io.json import json_normalize as jsNorm
whatToOrient = 'columns'

#Function to combine all the JSON files to one single CSV file suitable for small size folders
def getDatFrame(folerName):
    global whatToOrient
    directory_ = 'C:\Users\user\Desktop\DUTH\ΠΤΥΧΙΑΚΗ\\Json-output\\'
    path_ = directory_ + folerName
    file_list = os.listdir(path_)    
    df = pd.DataFrame()
    csvName = '\\' + folerName + '.csv'
    for file_ in file_list:
        filePath = path+"\\"+file_
        fileName = os.listdir(filePath)
       
        for singFile in fileName:
            with open(filePath + "\\" + singFile) as f:
                line_ = f.readline()
                data_ = json.loads(line_)
                vals_ = data_['hits']['hits']
           
                normData = pd.DataFrame.from_dict(jsNorm(vals_), orient = whatToOrient)
            df2 = pd.concat([df, normData], sort=True)
            
            #print(csvName)
            
    expCsv = df2.to_csv(r'C:\Users\user\Desktop\DUTH\ΠΤΥΧΙΑΚΗ\Demand' + csvName, index = None, header = True)
    return folerName

foldName_ = getDatFrame("stdCaseFl")
foldName_[:1]


def getGenDataFrame(mainFold):
    global whatToOrient
    directory_ = "C:\Users\user\Desktop\DUTH\ΠΤΥΧΙΑΚΗ\\Json-output\\"
    path_ = directory_ + mainFold
    fileLst = os.listdir(path_)

    for file_ in fileLst:
        filePath_ = path_ + "\\" + file_
        fileName = os.listdir(filePath_)
        df = pd.DataFrame()
        #print(file_)
        for singFile_ in fileName:
            # print(singFile_)
            with open(filePath_ + "\\" + singFile_) as f:
                line_ = f.readline()
                data_ = json.loads(line_)
                vals_ = data_['hits']['hits']
                normData = pd.DataFrame.from_dict(jsNorm(vals_), orient = whatToOrient)
            df2 = pd.concat([df, normData], sort=True)
            
        if not os.path.exists('C:\Users\user\Desktop\DUTH\ΠΤΥΧΙΑΚΗ\Demand\\testCase' + mainFold):
            os.makedirs('C:\Users\user\Desktop\DUTH\ΠΤΥΧΙΑΚΗ\Demand\\testCase' + mainFold)
        csvName_ = file_ + ".csv"
        foldNameCsv = df2.to_csv(r'C:\Users\user\Desktop\DUTH\ΠΤΥΧΙΑΚΗ\Demand\\testCase' + main_folder + '\\' + csvName_,
                                   index=None, header=True)

    return df2

foldName_ = getGenDataFrame("stdCaseFlGeneral")
foldName_[:1]
