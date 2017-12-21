# 0.8441
import numpy as np
import tensorflow as tf
import json
import h5py
from keras.datasets import cifar10
from keras import Sequential
from keras.layers import Input, Flatten, Conv2D, MaxPooling1D, Dropout, Conv1D, MaxPooling2D, Dense
from keras.models import Model
from keras import optimizers
from keras.models import load_model
import urllib.request
from matplotlib import pyplot as plt
import matplotlib.animation as animation
from appJar import gui
from som import SOM
import json

f = open('player.json', 'r')
data0 = json.load(f)
f.close()
f = open('appdata.json', 'r')
applist = json.load(f)
applist = applist['applist']['apps']
f.close()
f = open('tagDATA2.json', 'r')
tag = json.load(f)
f.close()

data2 = []
for i in data0:
    data2.append(data0[i]['gameplay'])
data = np.array(data2)
# Train a 20x30 SOM with 400 iterations
som = SOM(5, 2, 10, 100)  # My parameters
som.train(data)
cnn0 = load_model('model0.h5')
cnn1 = load_model('model1.h5')
cnn2 = load_model('model2.h5')
cnn3 = load_model('model3.h5')
cnn4 = load_model('model4.h5')
cnn5 = load_model('model5.h5')
cnn6 = load_model('model6.h5')
cnn7 = load_model('model7.h5')
cnn8 = load_model('model8.h5')
cnn9 = load_model('model9.h5')


def gameRecommendation(id):
    id = str(id)
    url = 'http://api.steampowered.com/IPlayerService/GetOwnedGames/v0001/?key=DF0E97F40E0BEE667B0BF01B6626A9DA&steamid=' \
          + id + '&format=json'
    u = urllib.request.urlopen(url)
    data = json.load(u)
    gameplay = data['response']['games']
    tagstat = np.zeros((1, 10))
    stat = np.zeros((1, 10, 10))
    for i in gameplay:
        gametemp = i['appid']
        gametime = i['playtime_forever']
        try:
            gameclass = tag[str(gametemp)]['tags'][0]
            tagstat[0][gameclass] += gametime
        except:
            pass
    mapped = som.map_vects(tagstat)
    playerClass = mapped[0].tolist()[0] + mapped[0].tolist()[1] * 5
    for i in range(playerClass):
        stat[0][i] = tagstat[0]
    pre0 = cnn0.predict(stat)
    pre1 = cnn1.predict(stat)
    pre2 = cnn2.predict(stat)
    pre3 = cnn3.predict(stat)
    pre4 = cnn4.predict(stat)
    pre5 = cnn5.predict(stat)
    pre6 = cnn6.predict(stat)
    pre7 = cnn7.predict(stat)
    pre8 = cnn8.predict(stat)
    pre9 = cnn9.predict(stat)
    pre = [pre0[0][0], pre1[0][0], pre2[0][0], pre3[0][0], pre4[0][0], pre5[0][0], pre6[0][0], pre7[0][0], pre8[0][0],
           pre9[0][0]]
    gameRec = pre
    game_rec = []
    url = 'http://api.steampowered.com/ISteamUser/GetFriendList/v0001/?key=DF0E97F40E0BEE667B0BF01B6626A9DA&steamid=' + id + '&relationship=friend'
    u = urllib.request.urlopen(url)
    friend = json.load(u)
    while len(game_rec) < 5:
        for i in range(len(gameRec)):
            flag = 0
            if gameRec[i] > 0.91:
                if flag == 1:
                    break
                for j in friend['friendslist']['friends']:
                    if flag == 1:
                        break
                    url = 'http://api.steampowered.com/IPlayerService/GetOwnedGames/v0001/?key=DF0E97F40E0BEE667B0BF01B6626A9DA&steamid=' + \
                          j['steamid'] + '&format=json'
                    u = urllib.request.urlopen(url)
                    friendgame = json.load(u)
                    try:
                        friendgame = friendgame['response']['games']
                        for k in friendgame:
                            try:
                                if tag[str(k['appid'])]['tags'][0] == i and k['appid'] not in game_rec and k[
                                    'appid'] not in gameplay:
                                    game_rec.append(k['appid'])
                                    flag = 1
                                    break
                            except:
                                continue
                    except:
                        continue
    for i in range(len(game_rec)):
        for j in applist:
            if j['appid'] == game_rec[i]:
                game_rec[i] = j['name']
                break
    return game_rec


demo = gui('Game Recommendation', '550x309')
demo.addLabelEntry('                 Steam ID                 ')
demo.setEntryDefault('                 Steam ID                 ', 'Please enter your Steam ID')
demo.setBgImage("giphy-2.gif")
id = demo.getEntry('                 Steam ID                 ')


def press(button):
    try:
        demo.stopSubWindow()
    except:
        pass
    id = demo.getEntry('                 Steam ID                 ')
    demo.setFont(12)
    mess = gameRecommendation(id)
    mess2 = ''
    for i in mess:
        mess2 += (i + '\n')
    try:
        demo.setMessage("Game recommended", mess2)
    except:
        demo.addMessage("Game recommended", mess2)


demo.addIconButton('search', press, 'search')
demo.go()
