import sys
import os
import json
import matplotlib.pyplot as plt

setName = sys.argv[1]

resultFile = setName + ".json"
with open(resultFile, 'r') as f:
    results = json.load(f)

config = None
with open(sys.argv[2], 'r') as f:
    config = json.load(f)
params = config['Sets'][setName]

# The part that has to be adopted for every test set
if setName == "Linear Combination - Constant Size":
    title = "constant size (1000000 entries), varying combination count"
    xlabel = "# Combinations"
    ylabel = "Time (ms)"
    xdata = [vx[1] for vx in params]
    xscale = 'linear'
    yscale = 'log'
else:
    title = "constant number of combinations (2), varying vector size"
    xlabel = "Vector size"
    ylabel = "Time (ms)"
    xdata = [vx[0] for vx in params]
    xscale = 'log'
    yscale = 'log'

# now create the plot
plt.plot(xdata, [d[0] for d in results["CuMat"]], '-o', label='cuMat')
plt.plot(xdata, [d[0] for d in results["Eigen"]], '-o', label='Eigen')
plt.plot(xdata, [d[0] for d in results["Numpy"]], '-o', label='numpy')
plt.plot(xdata[:len(results["Tensorflow"])], [d[0] for d in results["Tensorflow"]], '-o', label='Tensorflow')
for i,j in zip([xdata[0], xdata[-1]],[results["CuMat"][0][0], results["CuMat"][-1][0]]):
    plt.annotate(str(j),xy=(i,j), xytext=(-10,-10), textcoords='offset points')
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
plt.savefig(setName+'.png', bbox_inches='tight', dpi=300)