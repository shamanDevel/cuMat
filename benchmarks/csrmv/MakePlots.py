import sys
import os
import json
import matplotlib.pyplot as plt

setPath = sys.argv[1]
setName = setPath[setPath.rfind('/')+1:]

resultFile = setPath + ".json"
print("file:", resultFile)
with open(resultFile, 'r') as f:
    results = json.load(f)

config = None
with open(sys.argv[2], 'r') as f:
    config = json.load(f)
params = config['Sets'][setName]

title = "CSRMV: Sparse Matrix - Dense Vector multiplication, 2D Poisson matrix"
xlabel = "Matrix size (square)"
ylabel = "Time (ms)"
xdata = [vx[0] for vx in params]
xscale = 'log'
yscale = 'log'

# now create the plot
plt.plot(xdata, [d[0] for d in results["CuMat_CSR"]], '-o', label='cuMat - CSR')
plt.plot(xdata, [d[0] for d in results["CuMat_ELLPACK"]], '-o', label='cuMat - ELLPACK')
plt.plot(xdata, [d[0] for d in results["CuBlas"]], '-o', label='cuSPARSE')
plt.plot(xdata, [d[0] for d in results["Eigen"]], '-o', label='Eigen')
for i,j in zip([xdata[0], xdata[-1]],[results["CuMat_CSR"][0][0], results["CuMat_CSR"][-1][0]]):
    plt.annotate(str(j),xy=(i,j), xytext=(-10,-10), textcoords='offset points')
for i,j in zip([xdata[0], xdata[-1]],[results["CuBlas"][0][0], results["CuBlas"][-1][0]]):
    plt.annotate(str(j),xy=(i,j), xytext=(-10,5), textcoords='offset points')
for i,j in zip([xdata[0], xdata[-1]],[results["Eigen"][0][0], results["Eigen"][-1][0]]):
    plt.annotate(str(j),xy=(i,j), xytext=(-10,-10), textcoords='offset points')
plt.xscale(xscale)
plt.yscale(yscale)
plt.xlabel(xlabel)
plt.ylabel(ylabel)
plt.title(title)
plt.legend()
plt.xticks(xdata)

#plt.show()
plt.savefig(setPath+'.png', bbox_inches='tight', dpi=300)
