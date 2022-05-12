from sklearn import preprocessing
from sklearn.metrics.pairwise import cosine_similarity

labelEncoder = preprocessing.LabelEncoder()
classes = ['brainDisorder', 'general', 'inner_body_parts', 'pregnacy_menstrual', 'skin', 'cancer', 'covid19']
print(classes)

encoded_classes = labelEncoder.fit_transform(classes)
print(encoded_classes)

def calSimilarity(encodings, query_vector):
    simScore = []
    for i in encodings:
        value = cosine_similarity([i], query_vector)
        simScore.append(value[0][0])
    return simScore
