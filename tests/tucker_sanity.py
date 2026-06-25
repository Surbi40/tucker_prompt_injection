import tensorly
from tensorly.decomposition import tucker
import numpy as np

T = np.random.rand(10,10,5)
core, factors = tucker(T, rank=[3,3,2])

print("Core shape: ", core.shape)
print("Factor 0:", factors[0].shape)
print("factor 1:",factors[1].shape)
print("factor 2: ",factors[2].shape)

