import urllib.request
import json
import os
import sys
import math
import matplotlib.pyplot as plt
import statistics as stats
import numpy as np
import pandas as pd
from optparse import OptionParser
from datetime import datetime, timedelta
from collections import Counter
from sklearn.cluster import KMeans
from sklearn import preprocessing

#import statsmodels.stats.api as sms
#import ../database/dbMiddleware.py

valM = [5, 10, 15, 20, 40]
valP = [5, 10, 15, 20, 50]

def pause():
        programPause = input("Press the <ENTER> key to continue...")

def plot_histogram(start, end):

    if os.path.exists(opt.datadir + "/" + start + "-" + end + ".json"):
        data = json.load(open(opt.datadir + "/" + start + "-" + end + ".json", "r"))
    else:
        #httpresponse = urllib.request.urlopen("https://www.dcc.ufrrj.br/ocupationdb/api.php?period_from=2019-12-06%2008:00:00&period_to=2019-12-06%2018:00:00&type=data&mac_id=813")
        #httpresponse = urllib.request.urlopen("https://www.dcc.ufrrj.br/ocupationdb/api.php?period_from=" + urllib.parse.quote(start) + "&period_to=" + urllib.parse.quote(end) + "&type=data")
        httpresponse = urllib.request.urlopen("https://www.dcc.ufrrj.br/ocupationdb/api.php?period_from=" + urllib.parse.quote(start) + "&period_to=" + urllib.parse.quote(end))
        # Was hoping text would contain the actual json crap from the URL, but seems not...
        data = json.loads(httpresponse.read().decode())
        json.dump(data, open(opt.datadir + "/" + start + "-" + end + ".json", "w"))

    # get mac id's
    macs = []
    for pkt in data:
        macs.append(pkt['mac'])

    # remove dupplicates
    macs = list(dict.fromkeys(macs))

    # Read samples
    samples = {}
    for pkt in data:
        mac = pkt['mac']
        sensor = pkt['device_id']

        temp = pkt['t']
        #tipo gestao = "0"
        #tipo dado = "8"
        #Beacon = "80"
        #ProbeRequest = "40"
        #dado = "08"
        
        if temp[3] == "0" and (temp[2] == "8" or temp[2] == "4"):
            tipo[mac] = 1

        else:
            if temp[3] == "8":
                tipo[mac] = 0


        if mac not in samples:
            samples[mac] = {}
        if sensor not in samples[mac]:
            samples[mac][sensor] = []
        samples[mac][sensor].append(int(pkt['s']))


    for mac in samples:
        rssiMin = 0.0
        rssiMax = -100.00
        for sensor in samples[mac]:

            samples[mac][sensor].sort(reverse=True)

            lmeans = []

            # calc mean
            mean = sum(samples[mac][sensor]) / len(samples[mac][sensor])
            lmeans.append(mean)

            # calc xue17
            #print(samples[mac][sensor][:maxM])
            for M in valM:
                xue17 = sum(samples[mac][sensor][:M]) / len(samples[mac][sensor][:M])
                lmeans.append(xue17)

            # calc Msensivel percentual
            for P in valP:
                tamanho = len(samples[mac][sensor])
                namostras = tamanho * (P/100.0)
                namostras = math.ceil(namostras)
                #print(dezporcento)
                mSensivel = sum(samples[mac][sensor][:namostras]) / len(samples[mac][sensor][:namostras])
                lmeans.append(mSensivel)
                #print(mSensivel)
            
            # calc ajustavel pelo std
            stdev = stats.stdev(samples[mac][sensor]) if tamanho > 1 else 0.0
            percentual = 1.0 / stdev if stdev > 1.0 else 1.0
            namostras = tamanho * percentual
            namostras = math.ceil(namostras)
            mStdev = sum(samples[mac][sensor][:namostras]) / len(samples[mac][sensor][:namostras])

            statisticas = samples[mac][sensor]

            data = Counter(statisticas)
            #moda = stats.mode(statisticas)
            moda = data.most_common(1)
            mediana = np.median(statisticas, axis = None)


            # write to file
            f = open(opt.outdir + "/" + str(mac) + str(sensor), "a")
            #f.write("{}\t{:4d}\t{}\t{}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\n".format(start, tamanho, M, P, mean, xue17, mSensivel, stdev, mStdev, moda, mediana))
            f.write("Tempo: {}\t Tamanho: {:4d}\t M: {}\tP: {}\t Media: {:.3f}\t MediaXue: {:.3f}\t M variando: {:.3f}\t ST: {:.3f}\t STM : {:.3f}\t Moda: {}\t Mediana: {:.3f}\t\n".format(start, tamanho, M, P, mean, xue17, mSensivel, stdev, mStdev, moda, mediana))
            f.close()

            if mac not in rssi:
                rssi[mac] = {}
                allsamples[mac] = {}
            if sensor not in rssi[mac]:
                rssi[mac][sensor] = []
                allsamples[mac][sensor] = []
                #rssi[mac][sensor][start] = []
            #rssi[mac][sensor][start] = [mean, xue17, mSensivel, mStdev]
            rssi[mac][sensor].append(mean)
            allsamples[mac][sensor].append(samples[mac][sensor])

        
        if mac in qtdJanelasOuvida:
            qtdJanelasOuvida[mac] += 1 
        else:
            qtdJanelasOuvida[mac] = 1

        difJanela[mac] = rssiMax - rssiMin

        if len(rssi[mac][sensor]) > 1:
            if tipo[mac] == 0 or 1: continue
            else:
                tipo[mac] = 2        

                #rssi[mac][sensor]["mean"] = []
                #rssi[mac][sensor]["xue17"] = []

            #rssi[mac][sensor]["time"].append(start)
            #rssi[mac][sensor]["mean"].append(mean)
            #rssi[mac][sensor]["xue17"].append(xue17)


# Read args from command line
parser = OptionParser()
parser.add_option("-o", "--out-dir", type="string", dest="outdir", default="output", help="output directory DIR", metavar="DIR")
parser.add_option("-d", "--data-dir", type="string", dest="datadir", default="data", help="json data directory DIR", metavar="DIR")
parser.add_option("-w", "--max", type="int", dest="delta", default=1, help="RSSI window size", metavar="NUM")
parser.add_option("-v", "--verbose", action="store_true", dest="verbose", default=False, help="don't print debug messages to stdout")
parser.add_option("-p", "--qtd", type="int", dest="qtd", default=1, help="Quantidade de percentis a se ver")

(opt, args) = parser.parse_args()

# create directories
try:
    os.mkdir(opt.datadir)
    print("Directory", opt.datadir, "created.") 
except FileExistsError:
    print("Directory", opt.datadir, "already exists.")

try:
    os.mkdir(opt.outdir)
    print("Directory", opt.outdir, "created.") 
except FileExistsError:
    print("Directory", opt.outdir, "already exists.")

delta = opt.delta
start_date = datetime(2019, 12, 6, 8, 0, 0)
end_date = datetime(2019, 12, 6, 20, 0, 0)
delta = timedelta(minutes=opt.delta)

rssi = {}
qtdJanelasOuvida = {}
difJanela = {}
features = {}
tipo = {}
allsamples = {}
timestamps = []
while start_date <= end_date:
    start = start_date.strftime("%Y-%m-%d %H:%M:%S")
    timestamps.append(start)
    end = (start_date+delta).strftime("%Y-%m-%d %H:%M:%S")
    if opt.verbose: print("Getting data from " + start + " to " + end)
    plot_histogram(start, end)
    start_date += delta

for mac in rssi:
    # quantidade de sensores para cada MAC
    # quantidade de janelas de tempo em que o MAC foi detectado
    difJanela[mac] = 0.0
    qtdSensor = 0
    contadorDP = 0.0
    mediaDP = 0.0
    for sensor in rssi[mac]:

        difRssi = max(rssi[mac][sensor]) - min(rssi[mac][sensor])

        if difRssi > difJanela[mac]:
             difJanela[mac] = difRssi

        if len(rssi[mac][sensor]) > 1:
            mediaDP += stats.stdev(rssi[mac][sensor])
            contadorDP += 1.0            

            f = open(opt.outdir + "/" + str(mac) + str(sensor), "a")
            #print("Mac:{} Sensor:{} Media:{} DesvioPadrao:{}\n".format(mac, sensor, sum(rssi[mac][sensor])/len(rssi[mac][sensor]), stats.stdev(rssi[mac][sensor])))
            f.write("Mac:{} Sensor:{} Media:{} DesvioPadrao:{}\n".format(mac, sensor, sum(rssi[mac][sensor])/len(rssi[mac][sensor]), stats.stdev(rssi[mac][sensor])))
            arq = open(opt.outdir + "/" + str(mac), "a")
            arq.write("Mac:{} Numero de Aparicoes:{} Media:{} DesvioPadrao:{}\n".format(mac, qtdSensor, sum(rssi[mac][sensor])/len(rssi[mac][sensor]), stats.stdev(rssi[mac][sensor])))
        else:
            continue
    
        if max(rssi[mac][sensor]) > -99965:
            qtdSensor += 1

    if contadorDP > 0:
        features[mac] = [difJanela[mac], qtdSensor, tipo[mac]]
        #features[mac] = [qtdJanelasOuvida[mac],qtdSensor, mediaDP/contadorDP]
        #features[mac] = [qtdJanelasOuvida[mac],qtdSensor]
    else:
        continue
        #features[mac] = [qtdJanelasOuvida[mac],qtdSensor, 0]

    arq = open(opt.outdir + "/" + str(mac), "a") 
    arq.write("Mac: {} Features: {}".format(mac, features[mac]))

    f.close()
    arq.close()


featuresDF = pd.DataFrame(features)
#print(featuresDF.T)


x = featuresDF.T.values #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
featuresDF = pd.DataFrame(x_scaled)


print(featuresDF)

kmeans = KMeans(n_clusters=2).fit(featuresDF)
print(kmeans.labels_)
print(kmeans.cluster_centers_)
plt.scatter(featuresDF.T.iloc[0], featuresDF.T.iloc[1], c=kmeans.labels_)
#a, b = zip(*kmeans.cluster_centers_)
a, b, c = zip(*kmeans.cluster_centers_)
plt.scatter(a, b, marker="x")
pdf = plt.gcf()
#plt.show()
pdf.savefig("data1.pdf")