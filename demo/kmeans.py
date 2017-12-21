import numpy as np
import matplotlib.pyplot as plt
import json
from sklearn.cluster import KMeans
f=open('tagDATA.json','r')
data=json.load(f)
f.close()
f=open('tags.json','r')
label=json.load(f)
f.close()
labels=[]
frame=[]
for i in label:
    labels.append(label[str(i)]['name'])
for j in data:
    temp = []
    k = 0
    for i in labels:
        if i in data[str(j)]['tags']:
            temp.append(1)
            k+=1
        else:
            temp.append(0)
            k += 1
    frame.append(temp)

X = np.array(frame)
kmeans = KMeans(n_clusters=10, random_state=0).fit(X)
cat=kmeans.predict(frame)
cat=cat.tolist()
print(type(cat))
f=open('tagDATA2.json','w')
n=0
for i in data:
    data[str(i)]['tags']=[cat[n]]+data[str(i)]['tags']
    n=n+1
f.write(json.dumps(data))
