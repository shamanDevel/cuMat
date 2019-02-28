import sys
import numpy as np

if __name__== "__main__":
    """ 
    Arguments: dimensions num-components num-points output-file-name.
    Output file format: let d=dimensions, c=num-components, n=num-points
      d c n
      c lines for each component
        weight m_1 ... m_d (mean) v_00 ... v_d0 v_01 ... v_dd (covariance)
      n lines describing the points
        x_1 ... x_d
    """

    #parse input
    d = int(sys.argv[1])
    c = int(sys.argv[2])
    n = int(sys.argv[3])
    seed = int(sys.argv[4])
    print("Dimensions:",d,", Num Components:",c,", Num Points",n,", Seed", seed)
    outName = sys.argv[5]
    print("Output file name:",outName);

    #ground truth
    np.random.seed(66)
    c_means = np.random.normal(size=[c, d]) * 10
    c_covariances = np.abs(np.random.normal(size=[c, d, d])) * 1 + (1 if n>0 else 5) * np.array([(3+np.abs(np.random.normal()))*np.diag([1]*d) for i in range(c)])
    c_weights = np.abs(np.random.normal(size=[c])) + 1

    #pertubate data
    np.random.seed(seed)
    c_means += np.random.normal(size=[c, d]) * 5
    c_covariances += 1 * np.random.normal(size=[c, 1, 1]) * np.eye(d).reshape((1, d, d))
    c_covariances = np.matmul(c_covariances, np.transpose(c_covariances, (0,2,1)))
    c_weights += (np.random.normal(size=[c]) * 0.01 + 0.01)
    c_weights /= np.sum(c_weights)

    result = np.zeros((n, d), dtype=np.float32)
    for i in range(n):
        comp = np.random.choice(np.array(range(c)), p=c_weights)
        result[i] = np.random.multivariate_normal(
            c_means[comp], c_covariances[comp]
        )

    # write result
    with open(outName, 'w') as f:
        f.write('%d %d %d\n'%(d,c,n))
        for ic in range(c):
            f.write('%f'%c_weights[ic]);
            for x in range(d):
                f.write(' %f'%c_means[ic,x])
            for x in range(d):
                for y in range(d):
                    f.write(' %f'%c_covariances[ic,x,y])
            f.write('\n')
        for i in range(n):
            for x in range(d):
                f.write('%f '%result[i,x])
            f.write('\n')

    print("Done")
