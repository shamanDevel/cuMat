import sys
import json
import os
import numpy as np
import time

config = None
with open(sys.argv[1], 'r') as f:
    config = json.load(f)
params = config['Sets'][sys.argv[2]]

results = []

runs = 10
numConfigs = len(params)
for config in range(numConfigs):
    vectorSize = int(params[config][0])
    numCombinations = int(params[config][1])
    sys.stderr.write('  VectorSize:' + str(vectorSize) + ', Num-Combinations:' + str(numCombinations))

    # Create matrices
    vectors = [None] * numCombinations
    factors = [0] * numCombinations
    for i in range(numCombinations):
        vectors[i] = np.random.rand(vectorSize)
        factors[i] = np.random.rand()

    totalTime = 0.0
    for run in range(runs):

        start_time = time.perf_counter()

        # Main computation
        result = vectors[0] * factors[0]
        for i in range(1, numCombinations):
            result += vectors[i] * factors[i]

        elapsed_time = time.perf_counter() - start_time
        totalTime += elapsed_time * 1000

    finalTime = totalTime / runs
    sys.stderr.write(' -> ' + str(finalTime) + 'ms\n')
    results.append([finalTime])

sys.stdout.write(json.dumps(results) + '\n')