import numpy as np

for i in range(100):
    a = (np.random.rand(3,1) * 100 ).astype(np.int32)
    b = (np.random.rand(3,1) * 100 ).astype(np.int32)
    dot_product = a.T.dot(b)
    trace = np.trace(a.dot(b.T))
    if trace != dot_product[0,0]:
        print "not equal"
    else:
        print  "equal"
