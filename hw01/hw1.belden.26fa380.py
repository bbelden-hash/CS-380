import numpy as np

dataMatrix = np.zeros((50, 8), dtype = np.float64)

# extract the word vector of the word 'w' from the file with name 'fn'
def extractVector(fn, w):
    
    n = 0
    rows = 51
    col = np.zeros(rows, dtype = object)
    
    with open(fn, "r") as file:
        
        # 'for' loop iterates through all lines in file 'fn'
        for line in file:
            
            # macro 'items' is an array with each index an item from 'line'
            items = line.split()
            
            if not items:
                continue
            
            skipLine = False
            
            # 'for' loop iterates through all items in 'line'
            for i in range(len(items)):
                
                # bcs when parsing files all items (words, letters, floats, integers, etc.) are of type 'string' ...
                # if an item in the file is a float -> assign to type float, if an item in the file is a string -> assign to type string
                try:
                    item = float(items[i])
                except ValueError:
                    item = items[i]
                    
                if isinstance(item, str) and item != w:
                    skipLine = True
                    break
                elif col[0] == 0 and isinstance(item, float):
                    skipLine = True
                    break
                elif col[0] == w and isinstance(item, float):
                    if n < rows:
                        col[n] = item
                        n += 1
                elif item == w:
                    if n < rows:
                        col[n] = item
                        n += 1
                elif col[rows - 1] != 0:
                    return col
                
            if skipLine:
                continue
           
    return col

def intoMatrix(row, col):
    
    for i in range(len(dataMatrix)):
        dataMatrix[i][col] = row[i]
        
def buildMatrix(fn):
    
    col = 0
    
    with open(fn, "r") as file:
        for line in file:
            wordsinLine = line.split()
            
            if not wordsinLine:
                continue
            word = wordsinLine[0]
            
            vector = extractVector("glove.2024.wikigiga.50d.txt", word)
            
            if vector is not None:
                vector = vector[1:]
                
                intoMatrix(vector, col)
                col += 1
            else:
                print(f"warning: word vector not found for line: {line.strip()}")
                
def loadData(fn):
    
    data = np.load(fn)
    return data

def prettyPrint(data):
    
    for row in data:
        for value in row:
            print(f"{value:10.4f}", end=" ")
        print()
    
           
# buildMatrix("words.txt")
data = loadData("dataMatrix01_0000.npy")
prettyPrint(data)