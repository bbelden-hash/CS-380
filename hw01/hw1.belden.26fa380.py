import numpy as np

dataMatrix = np.zeros((50, 8), dtype = np.float64)
myCos = np.zeros((8, 8), dtype = np.float64)
myMikCos = np.zeros((4, 4), dtype = np.float64)

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

def npCosSim(v1, v2):
    # using NumPy to do the heavy lifting
    
    v = np.array(v1)
    w = np.array(v2)
    
    if len(v1) != len(v2):
        return ("error: dimension of vectors (50) must be same to take the inner product between the two")
    
    dotProduct = np.dot(v, w)
    normV1 = np.linalg.norm(v)
    normV2 = np.linalg.norm(w)
    
    if normV1 == 0 or normV2 == 0:
        return 0.00000
    
    result = dotProduct / (normV1 * normV2)
    return result
    
# Euclidean length    
def pNorm(p, v):
    # using a for loop
    
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
    
    targets = {"good", "evil", "devil", "angel", "goods", "cup", "happy", "ethical"} # update targets accordingly based on what word vectors you are looking for in GloVe
    
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
    
    for word, vector in v.items():
        norms[word] = pNorm(2, vector)
            
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
            myCos[i][j] = cos # change cosine simlarity matrix dimensions (at top with 'global' macros) pending on number of words looking for in GloVe file 
            j += 1
            
        i += 1
        
def cosThetaMik(embedding):
    
    i = 0
    
    for w1, v1 in embedding.items():
        j = 0
        for w2, v2 in embedding.items():
            
            cos = myCosSim(v1, v2)
            myMikCos[i][j] = cos
            j += 1
            
        i += 1
        
    return myMikCos
        
def nplen(v):
    # using NumPy
    
    myLen = np.linalg.norm(v)
    return myLen

# mikolov example 
def mikolov(fn):
    
    vectors = {}
    targets = {"king", "queen", "man", "woman"}
    
    with open(fn, "r") as file:
        for line in file:
            wordsinLine = line.split()
            
            if not wordsinLine:
                continue
            word = wordsinLine[0]
            
            if word in targets:
                mikolovData = extractVector("glove.2024.wikigiga.50d.txt", word)
                
                vectors[word] = mikolovData[1:]
                
    return vectors

def computeMik(v):
    
    vector1 = v["king"] - v["man"] + v["woman"]
    vector2 = v["queen"]
     
    # Print a header column
    print(f"{'Index':<8} | {'Math Vector':<12} | {'Queen Vector':<12}")
    print("-" * 40)
    
    # Loop through and pair them up row by row
    for i, (val1, val2) in enumerate(zip(vector1, vector2)):
        print(f"Row {i:<4} | {val1:12.5f} | {val2:12.5f}")
        
    similarity = npCosSim(vector1, vector2)
    return similarity
    

# ==========================================================================================================================
# Place to call all above functions, read and call accordingly ...
         
# buildMatrix("words.txt")
# bigData = loadData("dataMatrix01_0000.npy")
# prettyPrint(bigData)

# myVectors = buildVectors("words.txt")
# myNorms = twoNorms(myVectors)
# unitVectors(myVectors, myNorms)
# cosTheta(myVectors)
# cosData = loadData("myCos.npy")
# prettyPrint(cosData)

myMikolov = mikolov("mikolovWords.txt")
# mikNorms = twoNorms(myMikolov)
# unitVectors(myMikolov, mikNorms)
# cosMikData = cosThetaMik(myMikolov)
similarity = computeMik(myMikolov)