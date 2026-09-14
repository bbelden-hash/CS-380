import numpy as np

dataMatrix = np.zeros((50, 8), dtype = np.float64)
myCos = np.zeros((8, 8), dtype = np.float64)

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
            print(f"{value:10.5f}", end=" ")
        print()
        
        
# cosine similarity
def myCosSim(v1, v2):
    
    # using a for loop
    
    howAlike = 0
    
    if len(v1) != len(v2):
        return ("error: dimension of vectors (50) must be same to take inner product between the two")
    
    for i in range(len(v1)):
        product = v1[i] * v2[i]
        howAlike += product
        
    return howAlike
       
def pNorm(p, v):
    
    sum = 0
    
    for num in v:
        pth = abs(num)**p
        sum += pth
        
    norm = sum**(1/p)
    return norm
    
def buildVectors(fn):
    
    # initialize an empty dictionary to store output
    # index using word (key), output will be vector associated with word (value)
    vectors = {}
    
    targets = {"good", "evil", "devil", "angel", "goods", "cup", "happy", "ethical"}
    
    with open(fn, "r") as file:
        for line in file:
            wordsinLine = line.split()
            
            if not wordsinLine:
                continue
            word = wordsinLine[0]
            
            if word in targets:
                vectorData = extractVector("glove.2024.wikigiga.50d.txt", word)
                
                vectors[word] = vectorData[1:]
    
    return vectors

def twoNorms(v):
    
    norms = {}
    targets = {"good", "evil", "devil", "angel", "goods", "cup", "happy", "ethical"}
    
    for word, vector in v.items():
        
        if word in targets:
            norm = pNorm(2, vector)
            norms[word] = norm
            
    return norms

def unitVectors(vectors, norms):
    
    for wordV, vector in vectors.items():
        for wordN, norm in norms.items():
            
            if wordV == wordN:
                
                for i, value in enumerate(vector):
                    vector[i] = value / norm
                    
def cosTheta(embedding):
    
    i = 0
    
    for w1, v1 in embedding.items():
        j = 0
        for w2, v2 in embedding.items():
            
            cos = myCosSim(v1, v2)
            myCos[i][j] = cos
            j += 1
            
        i += 1
            
        
        
                        
                    
    
    
            
            
        

# =============================================================================================================
         
# buildMatrix("words.txt")
# bigData = loadData("dataMatrix01_0000.npy")
# prettyPrint(data)

myVectors = buildVectors("words.txt")
myNorms = twoNorms(myVectors)
unitVectors(myVectors, myNorms)
cosTheta(myVectors)
cosData = loadData("myCos.npy")
prettyPrint(cosData)


        
            
            
            


            
            
            





                
                    
            
                    
                    
                   
                        
                    
                    
                    
                
            
    
    
    
    
    
    
    
    
    
    


        
        
        
        

