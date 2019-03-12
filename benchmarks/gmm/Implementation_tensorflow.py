import sys
import json
import os
import tensorflow as tf
import numpy as np
import random
import time

config = None
with open(sys.argv[1], 'r') as f:
    config = json.load(f)
params = config['Sets'][sys.argv[2]]

results = []

runs = 1
numConfigs = len(params)
for config in range(numConfigs):
    vectorSize = int(params[config][0])
    numCombinations = int(params[config][1])
    sys.stderr.write('  VectorSize: ' + str(vectorSize) + ', Num-Combinations: ' + str(numCombinations))

    sess = tf.Session()

    try:
        # Create matrices
        vectors = [None] * numCombinations
        factors = [0] * numCombinations
        for i in range(numCombinations):
            vectors[i] = tf.constant(np.random.rand(vectorSize))
            factors[i] = 1 #random.random()
        sess.run(tf.global_variables_initializer())

        totalTime = 0.0
        for run in range(runs+1):

            with tf.device('/gpu:0'):
                result = vectors[0] * factors[0]
                for i in range(1, numCombinations):
                    result += vectors[i] * factors[i]

            start_time = time.perf_counter()
            sess.run(result)
            elapsed_time = time.perf_counter() - start_time

            if run>0: #Trow away the first result
                totalTime += elapsed_time * 1000

        finalTime = totalTime / runs
    except:
        sys.stderr.write("Error: {0}\n".format(sys.exc_info()[0]))
        finalTime = -1

    sys.stderr.write(' -> ' + str(finalTime) + 'ms\n')
    if (finalTime >= 0):
        results.append([finalTime])

sys.stdout.write(json.dumps(results) + '\n')