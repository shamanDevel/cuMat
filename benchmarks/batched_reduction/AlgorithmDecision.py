from enum import IntEnum
import math

class Algorithm(IntEnum): # values equal the color indices
    CUB = 1
    Thread = 2
    Warp = 3
    Block512 = 7
    Device1 = 9
    Device2 = 10
    Device4 = 11
    
MAX_X = 20
MAX_Y = 22
    
CONDITIONS_ROW = [
    #(Algorithm, [Condition (a,b,c) s.t. a*nb+b*bs>=c], List of enclosing points)
    (Algorithm.Device1, [(1.2,1,19.5), (-1,0,-2.5)]),
    (Algorithm.Device2, [(0.42857142857142855,1,17.821428571428573), (-1,0,-4.25), (1,0,2.5)]),
    (Algorithm.Device4, [(0,1,16.25),(-1,0,-5.5),(1,0,4.25)]),
    (Algorithm.Block512,[(-1.6, 1, 8), (-1,0,-5)]),
    (Algorithm.Thread,  [(0.475, -1, 2.01875), (0, -1, -4.75)]),
    (Algorithm.Warp,    [])
]

CONDITIONS_COLUMN = [
    (Algorithm.Device1, [(1.5,1,19.5), (-1,0,-2.5)]),
    (Algorithm.Device2, [(0,1,15.5), (1,0,2.5), (-1,0,-4)]),
    (Algorithm.Device4, [(0,1,15.75), (1,0,4), (-1,0,-5.75)]),
    (Algorithm.Block512,[(0,1,9), (-1,0,-2.5)]),
    (Algorithm.Warp,    [(0,1,4), (-1,0,-11.75)]),
    (Algorithm.Thread,  [])
]

CONDITIONS_BATCH = [
    (Algorithm.Device1, [(-1,0,-2), (1.875,1,19)]),
    (Algorithm.Device4, [(1,0,2), (-1,0,-4.25), (10, 9, 184.25)]),
    (Algorithm.Device2, [(1,0,2), (-1,0,-4.25), (-0.22222, 1, 14.085555)]),
    (Algorithm.CUB,     [(1,0,4), (0,1,11.5), (-1,0,-8.5)]),
    (Algorithm.Block512,[(0,1,8), (-1,0,-2)]),
    (Algorithm.Warp,    [(0,1,2.75), (-1,0,-11.75)]),
    (Algorithm.Thread,  [])
]

def _get_alg(numBatches, batchSize, specs):
    nb = math.log2(numBatches)
    bs = math.log2(batchSize)
    #print(nb,bs,sep=', ')
    for alg, condx in specs:
        success = True
        for a,b,c in condx:
            #print('  alg=',alg,'; a=',a,', b=',b, ', c=',c,' -> ',a*nb+b*bs, sep='')
            if not a*nb+b*bs>=c:
                success = False
                break
        if success:
            #print('  found in alg=',alg,sep='')
            return alg
    
def get_alg_row(numBatches, batchSize):
    return _get_alg(numBatches, batchSize, CONDITIONS_ROW)
def get_alg_col(numBatches, batchSize):
    return _get_alg(numBatches, batchSize, CONDITIONS_COLUMN)
def get_alg_batch(numBatches, batchSize):
    return _get_alg(numBatches, batchSize, CONDITIONS_BATCH)
