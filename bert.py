from common import calSimilarity
import tensorflow_hub as hub
import tensorflow_text as text
from pymongo import MongoClient
import details
import numpy as np

bert_preprocess = hub.KerasLayer('encoders/bert_en_cased_preprocess_3')
bert_encoder = hub.KerasLayer('encoders/bert_en_wwm_cased_L-24_H-1024_A-16_4')

def embedd(ques):
    preprocessed_ques = bert_preprocess(ques)
    encodings = bert_encoder(preprocessed_ques)
    return encodings['pooled_output']

def nembedd(ques):
    embeddings = []
    bsize = 100
    print('Encodings process start\n')
    val = (len(ques)//bsize)
    print(val)

    for i in range(val):
        print(f'i = {i}')
        sr = i*bsize
        er = bsize*(i+1)
        print(f'{sr}:{er}')
        bques = ques[sr:er]
        batchData = embedd(bques)
        embeddings+=list(batchData)
        print(len(embeddings))
        del batchData

    sr = (i+1)*bsize
    print(f'{sr}')
    bques = ques[sr:]
    print(len(bques))
    batchData = embedd(bques)
    embeddings+=list(batchData)
    print(len(embeddings))
        
    print('Encodings process done\n\n')
    embeddings = np.array(embeddings)
    return embeddings


def runBert(ques, encodings, query, cname):
    
    print('***********Runnung BERT MODEL****************')
    print('Encodings Shape: \n', encodings.shape)
    
    query_vector = embedd([query])
    print('Query Vector Shape: \n', query_vector.shape)
    
    simScore = calSimilarity(encodings, query_vector)
    
    conn = MongoClient(host='localhost')
    db = conn[details.dbname]
    col = db[cname]
    
    simQuesAns = []
    nques = 5
    result = {'result':{}}
    temp_simScore = sorted(simScore)
    tempMvalue = temp_simScore[:nques]
    print(len(tempMvalue))
    for i in tempMvalue:
        index = simScore.index(i)
        res = col.find({'question':ques[index]}, {'_id':0})
        for r in res:
            simQuesAns.append(r)
    
    print(len(simQuesAns))
    result['result'] = simQuesAns
    return result
