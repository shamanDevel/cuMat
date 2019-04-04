import sys
import os
import json
import matplotlib.pyplot as plt
import math

setPath = sys.argv[1]
setName = setPath[setPath.rfind('/')+1:]

resultFile = setPath + ".json"
with open(resultFile, 'r') as f:
    results = json.load(f)

size = results["Size"]
sets = ["Row", "Column", "Batch"]
methods = ["Baseline", 
           "Thread", 
           "Warp", 
           "Block64", "Block128", "Block256", "Block512", 
           "Device8", "Device16", "Device32"]
xlabel = "2^N entries along reduced axis"
ylabel = "Time (ms)"
xdata = [math.log2(vx[0]) for vx in results[sets[0]]]
xscale = 'linear'
yscale = 'log'

for set in sets:
    # now create the plot
    plt.figure(dpi=500)
    for i,m in enumerate(methods):
        plt.plot(xdata, [vx[i+1] for vx in results[set]], '-o', label=m)
    plt.xscale(xscale)
    plt.yscale(yscale)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title("Reduction axis: " + set + "\nTotal vector size: " + str(size))
    plt.legend()
    plt.xticks(xdata)
    plt.savefig(setPath+"_"+set+'.png', bbox_inches='tight', dpi=500)
