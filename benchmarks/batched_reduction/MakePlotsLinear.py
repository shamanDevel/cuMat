import sys
import os
import json
import matplotlib.pyplot as plt
import math
import seaborn as sns

setPath = sys.argv[1]
setName = setPath[setPath.rfind('/')+1:]

resultFile = setPath + ".json"
with open(resultFile, 'r') as f:
    results = json.load(f)

size = results["Size"]
sets = ["Row", "Column", "Batch"]
methods = ["CUB", 
           "Thread", 
           "Warp", 
           "Block64", "Block128", "Block256", "Block512", 
           "Device1", "Device2", "Device4", "Device8", "Device16", "Device32"]
xlabel = "2^N entries along reduced axis"
ylabel = "Time (ms)"
xdata = [math.log2(vx[0]) for vx in results[sets[0]]]
xscale = 'linear'
yscale = 'log'
colors = \
     sns.color_palette("muted", 3)+ \
     sns.color_palette("ch:4.5,-.2,dark=.3", 4)+ \
     sns.color_palette("ch:3.5,-.2,dark=.3", 6)

for set in sets:
    # now create the plot
    plt.figure(dpi=500)
    for (i,m),col in zip(enumerate(methods), colors):
        plt.plot(xdata, [vx[i+1] for vx in results[set]], 
                 '-o', label=m, color=col)
    plt.xscale(xscale)
    plt.yscale(yscale)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title("Reduction axis: " + set + "\nTotal vector size: " + str(size))
    plt.legend()
    plt.xticks(xdata)
    plt.savefig(setPath+"_"+set+'.png', bbox_inches='tight', dpi=500)
