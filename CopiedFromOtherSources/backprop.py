# From http://iamtrask.github.io/2015/07/12/basic-python-network/

import numpy as np

# sigmoid function
''' gives results:
[[ 0.03178421]
 [ 0.02576499]
 [ 0.97906682]
 [ 0.97414645]]
'''
def nonLin(x,deriv=False):
    if(deriv==True):
        return x*(1-x)
    return 1/(1+np.exp(-x))

''' Tried max(0.01x, min(1 + 0.01 *(x - 1), x)) (variation of Leaky ReLU)
where the predictions results are closer to 0 and 1:
[[-0.0095237 ]
 [-0.00938117]
 [ 1.00820392]
 [ 1.00834645]]
def nonLin(x,deriv=False):
    result = np.copy(x)
    for r in range(len(x)):
        for c in range(len(x[r])):
            if result[r, c] > 1:
                if (deriv==True):
                    result[r, c] = 0.01
                else:
                    result[r, c] = 1 + 0.01 * (result[r, c] - 1)
            elif result[r, c] >= 0:
                if (deriv==True):
                    result[r, c] = 1
            else:
                if (deriv==True):
                    result[r, c] = 0.01
                else:
                    result[r, c] = 0.01 * result[r, c]
    return result
'''

# input dataset
X = np.array([  [0,0,1],
                [0,1,1],
                [1,0,1],
                [1,1,1] ])
    
# output dataset            
y = np.array([[0,0,1,1]]).T

# seed random numbers to make calculation
# deterministic (just a good practice)
np.random.seed(1)

# initialize weights randomly with mean 0
syn0 = 2*np.random.random((3,1)) - 1

for iter in xrange(1000):

    # forward propagation
    l0 = X                       # l0 is 4x3
    l1 = nonLin(np.dot(l0,syn0)) # l1 is 4x1

    # how much did we miss?
    l1_error = y - l1

    # multiply how much we missed by the 
    # slope of the sigmoid at the values in l1
    l1_delta = l1_error * nonLin(l1,True)

    # update weights
    syn0 += np.dot(l0.T,l1_delta)

print "Output After Training:"
print l1