# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 20:48:57 2019
@author: water
"""

import games.arkanoid.communication as comm
from games.arkanoid.communication import ( \
    SceneInfo, GameInstruction, GameStatus, PlatformAction
)

import pickle
import os

Frame=[]
Status=[]
Ballposition=[]
PlatformPosition=[]
Bricks=[]
filenamelist = []

log = 'C:\\Users\\justin\\Desktop\\robot\\MLGame-beta4.0\\games\\arkanoid\\log'
filenames = os.listdir(log)

for filename in filenames:
    fileroutine = log + '\\' + filename
    with open(fileroutine,"rb") as f:
        data_list = pickle.load(f)
    for i in range(0,len(data_list)):
        Frame.append(data_list[i].frame)
        Status.append(data_list[i].status)
        Ballposition.append(data_list[i].ball)
        PlatformPosition.append(data_list[i].platform)
        Bricks.append(data_list[i].bricks)

import numpy as np
PlatX=np.array(PlatformPosition)[:,0][:,np.newaxis]
PlatX_next=PlatX[1:,:]
instruct=(PlatX_next-PlatX[0:len(PlatX_next),0][:,np.newaxis])/5


Ballarray=np.array(Ballposition[:-1])
pp=np.array(Ballposition[1:])
x=np.hstack((Ballarray,pp,PlatX[0:-1,0][:,np.newaxis]))

y=instruct
print(x)



from sklearn.model_selection import train_test_split


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.0001, random_state = 0)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(x_train, y_train)

print(x_test)
y_knn=neigh.predict(x_test)

acc=accuracy_score(y_knn, y_test)

"""from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
scaler.fit(x_train)
x_train_stdnorm=scaler.transform(x_train)
neigh.fit(x_train_stdnorm, y_train)
x_test_stdnorm=scaler.transform(x_test)
yt=neigh.predict(x_test_stdnorm)
acc=accuracy_score(yt, y_test)"""



filename="Knn_example.sav"
pickle.dump(neigh, open(filename, 'wb'))


l_model=pickle.load(open(filename, 'rb'))
yp_l=l_model.predict(x_test)
print("acc load: %f " % accuracy_score(yp_l, y_test))