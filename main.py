from bert import *
from classPredict import predict

def run(query):
    ques, cname = predict(query)
    if cname == 1:
        return ques
    encodings = nembedd(ques)
    result = runBert(ques, encodings, query, cname)
    return result