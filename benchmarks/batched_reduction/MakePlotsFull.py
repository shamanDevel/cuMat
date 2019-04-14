import sys
import os
import json
import matplotlib as mpl
import matplotlib.pyplot as plt
import math
import numpy as np
import seaborn as sns

setPath = "../batched_reductions_full_batch" #sys.argv[1]
setName = setPath[setPath.rfind('/')+1:]
title = "Row" #sys.argv[2]

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
colors = sns.color_palette("muted", len(fieldNames)-1)
cmap = mpl.colors.ListedColormap( \
    [(1.0,1.0,1.0)]+ \
     sns.color_palette("muted", 3)+ \
     sns.color_palette("ch:4.5,-.2,dark=.3", 5)+ \
     sns.color_palette("ch:3.5,-.2,dark=.3", 6))
bounds=[v-1.5 for v in range(len(fieldNames))]
norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

# create figure
fig = plt.figure(figsize=plt.figaspect(0.5))
fig.suptitle("Reduction axis: "+title)

# create 3d plot
ax = fig.add_subplot(1, 2, 1)
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

# create cross section
ax = fig.add_subplot(1, 2, 2)

plt.show()

# output
outputFile = setPath + ".png"
