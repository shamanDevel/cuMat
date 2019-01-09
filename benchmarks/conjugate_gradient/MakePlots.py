import sys
import os
import json
import matplotlib.pyplot as plt

setPath = sys.argv[1]
setName = setPath[setPath.rfind('/')+1:]

resultFile = setPath + ".json"
with open(resultFile, 'r') as f:
    results = json.load(f)

config = None
with open(sys.argv[2], 'r') as f:
    config = json.load(f)
params = config['Sets'][setName]

# The part that has to be adopted for every test set
title = "2D-Poisson problem, CG solver"
xlabel = "Grid size (square)"
ylabel = "Time (ms)"
xdata = [vx[0] for vx in params]
xscale = 'log'
yscale = 'log'

# now create the plot
runAheadList = [0,1,5,20]
for runAhead in runAheadList:
	plt.plot(xdata, [d[0] for d in results["CuMat_%d"%runAhead]], '-o', label='cuMat, runAhead=%d'%runAhead)
	for i,j in zip([xdata[0], xdata[-1]],[results["CuMat_%d"%runAhead][0][0], results["CuMat_%d"%runAhead][-1][0]]):
		plt.annotate(str(j),xy=(i,j), xytext=(-10,-10), textcoords='offset points')
	
plt.plot(xdata, [d[0] for d in results["Eigen"]], '-o', label='Eigen')
for i,j in zip([xdata[0], xdata[-1]],[results["Eigen"][0][0], results["Eigen"][-1][0]]):
    plt.annotate(str(j),xy=(i,j), xytext=(-10,5), textcoords='offset points')
plt.xscale(xscale)
plt.yscale(yscale)
plt.xlabel(xlabel)
plt.ylabel(ylabel)
plt.title(title)
plt.legend()
plt.xticks(xdata)

#plt.show()
plt.savefig(setPath+'.png', bbox_inches='tight', dpi=300)
