import numpy as np
import  nltk.translate.bleu_score as bleu

def WRR(text1,text2):
    a = set(text1.lower().split())
    b = set(text2.lower().split())
    
    if (len(a) == 0) and (len(b) == 0):
        return .5
    
    c = a.intersection(b)
    return float(len(c))/(len(a) + len(b) - len(c))

def levenshtein(seq1, seq2):
    seq1 = seq1.lower()
    seq2 = seq2.lower()
    
    size_x = len(seq1) + 1
    size_y = len(seq2) + 1
    matrix = np.zeros ((size_x, size_y))
    for x in range(size_x):
        matrix [x, 0] = x
    for y in range(size_y):
        matrix [0, y] = y

    for x in range(1, size_x):
        for y in range(1, size_y):
            if seq1[x-1] == seq2[y-1]:
                matrix [x,y] = min(
                    matrix[x-1, y] + 1,
                    matrix[x-1, y-1],
                    matrix[x, y-1] + 1
                )
            else:
                matrix [x,y] = min(
                    matrix[x-1,y] + 1,
                    matrix[x-1,y-1] + 1,
                    matrix[x,y-1] + 1
                )
    #print (matrix)
    return (matrix[size_x - 1, size_y - 1])

def CRR(text1, text2):
    try:
        return 1 - float(levenshtein(text1,text2))/max(len(text1),len(text2))
    except:
        return 0

def bleu_score(text1,text2):
    return bleu.sentence_bleu([text1.lower().split()],text2.lower().split())