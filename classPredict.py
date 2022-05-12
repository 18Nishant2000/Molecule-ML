import bert
from tensorflow import keras
import numpy as np
import common
from pymongo import MongoClient
import sys
import details

model1 = keras.models.load_model('./model/max_accuracy_model.epoch24-accuracy0.85')
model2 = keras.models.load_model('./model/max_val_accuracy_model.epoch23-val_accuracy0.87')
model3 = keras.models.load_model('./model/min_loss_model.epoch24-val_accuracy0.39')
model4 = keras.models.load_model('./model/min_val_loss_model.epoch23-val_accuracy0.40')

def predict(query):
    queryVector = bert.embedd([query])
    pred = []
    pred.append(model1.predict(queryVector))
    pred.append(model2.predict(queryVector))
    pred.append(model3.predict(queryVector))
    pred.append(model4.predict(queryVector))
    print(f'pred1 : {pred[0]} pred2 : {pred[1]} pred3 : {pred[2]} pred4 : {pred[3]}')
    label = []
    for i in pred:
        tempIndex = np.where(i[0] == np.amax(i))
        index = tempIndex[0][0]
        label.append(common.labelEncoder.inverse_transform([index]))    
    print(label)
    for i in label:
        print(i[0])
    label = [i[0] for i in label]
    print(label)
    
    result = {}
    for i in set(label):
        result[i] = label.count(i)
    
    mvalue = max([i for i in result.values()])
    flable = ''
    for key, value in result.items():
        if value == mvalue:
            flable = key
            break
    print(f'Final Label {flable}')
    
    
    conn = MongoClient(host='localhost')
    db = conn[details.dbname]
    cname = str(flable)
    col = db[cname]
    
    try:
        data = col.find({},{'_id':0, 'answers':0})
    except Exception as e:
        print(e)
        
    ques = []
    for i in data:
        ques.append(i['question'])
    print(len(ques))
    
    flag = 0
    if query in ques:
        result = {'result':{}}
        res = list(col.find({'question': query},{'_id':0}))
        result['result'] = res
        flag = 1
        return result, flag

    return ques, cname
