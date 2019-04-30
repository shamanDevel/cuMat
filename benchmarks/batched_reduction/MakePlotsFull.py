import sys
import os
import json
import matplotlib as mpl
import matplotlib.pyplot as plt
import math
import numpy as np
import seaborn as sns

if len(sys.argv)>2:
    setPath = sys.argv[1]
    title = sys.argv[2]
else:
    setPath = "../batched_reductions_full_row"
    title = "Row" 
setName = setPath[setPath.rfind('/')+1:]

# get decision function
from AlgorithmDecision import Algorithm, get_alg_row, get_alg_col, get_alg_batch
if title=="Row":
    decisionFun = get_alg_row
elif title=="Column":
    decisionFun = get_alg_col
else:
    decisionFun = get_alg_batch

# load result file
resultFile = setPath + ".txt"
results = np.genfromtxt(resultFile, delimiter='\t', names=True)
fieldNames = results[0].dtype.names
print(fieldNames)

# get min and max bounds
numBatches = sorted(set([r[0] for r in results]))
batchSize = sorted(set([r[1] for r in results]))
print("numBatches:", numBatches,"->",len(numBatches))
print("batchSize:", batchSize,"->",len(batchSize))
minNumBatches = numBatches[0]
maxNumBatches = numBatches[-1]
minBatchSize = batchSize[0]
maxBatchSize = batchSize[-1]

# create colorbar
Z = np.full((len(numBatches),len(batchSize)), -1, dtype=np.float)
for r in results:
    nb = r[0]
    bs = r[1]
    x = numBatches.index(nb)
    y = batchSize.index(bs)
    Z[x,y] = np.argmin([r[i] for i in range(2, len(fieldNames))])
Z = Z.T
colors = [(1.0,1.0,1.0)]+ \
     sns.color_palette("muted", 3)+ \
     sns.color_palette("ch:4.5,-.2,dark=.3", 5)+ \
     sns.color_palette("ch:3.5,-.2,dark=.3", 6)
cmap = mpl.colors.ListedColormap(colors)
bounds=[v-1.5 for v in range(len(fieldNames))]
norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

# create figure
#fig = plt.figure(figsize=plt.figaspect(0.5))
#fig.suptitle("Reduction axis: "+title)
plt.title("Reduction axis: "+title)

# create plot of measurements
#ax = fig.add_subplot(1, 2, 1)
ax = plt.gca()
img = ax.imshow(Z,interpolation='nearest',
                cmap = cmap, norm=norm,
                origin='lower')
ax.set_xlabel("log2(Num Batches)")
ax.set_ylabel("log2(Batch Size)")
def is_power_of_two(num):
    num = int(num)
    return ((num & (num - 1)) == 0) and num > 0
#ax.set_xticks([i for i in range(len(numBatches))], False)
print([i for i in range(len(numBatches)) if is_power_of_two(numBatches[i])])
ax.set_xticks([i for i in range(len(numBatches)) if is_power_of_two(numBatches[i])])
ax.set_xticklabels([str(int(math.log2(numBatches[i]))) for i in range(len(numBatches)) if is_power_of_two(numBatches[i])])
ax.set_yticks([i for i in range(len(batchSize)) if is_power_of_two(batchSize[i])])
ax.set_yticklabels([str(int(math.log2(batchSize[i]))) for i in range(len(batchSize)) if is_power_of_two(batchSize[i])])
cbar = plt.colorbar(img, cmap=cmap, norm=norm, boundaries=bounds, ticks=range(-1, len(fieldNames)-1))
cbar.ax.set_yticklabels(['N/A'] + list(fieldNames[2:]))

# plot decisions / clusters
Xcluster = numBatches
Ycluster = batchSize
UPSCALING = 4
Xcluster = np.interp(
    [x/UPSCALING for x in range(UPSCALING*(len(Xcluster)-1)+1)],
    range(len(Xcluster)),
    Xcluster)
Ycluster = np.interp(
    [x/UPSCALING for x in range(UPSCALING*(len(Ycluster)-1)+1)],
    range(len(Ycluster)),
    Ycluster)
print("Xcluster",Xcluster)
print("Ycluster",Ycluster)
Zcluster = np.full((len(Xcluster),len(Ycluster)), -1, dtype=np.float)
for x,nb in enumerate(Xcluster):
    for y,bs in enumerate(Ycluster):
        Zcluster[x,y] = decisionFun(nb, bs)-1
    #break
Zcluster = Zcluster.T
ax.imshow(Zcluster,
                interpolation='nearest',
                cmap = cmap, norm=norm,
                origin='lower',
                alpha=0.2,
                extent = img.get_extent())

# create cross section
#ax = fig.add_subplot(1, 2, 2)

#plt.show()

# output
outputFile = setPath + ".png"
plt.savefig(outputFile, bbox_inches='tight', dpi=500)
