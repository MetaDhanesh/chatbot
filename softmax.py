"""Softmax."""

scores = [3.0, 1.0, 0.2]

import numpy as np

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    # return np.exp(x) / np.sum(np.exp(x), axis = 0)
    x = [np.exp(t) for t in x]
    denominator = sum(x)
    out = np.array(x) / denominator
    return out

print(softmax(scores))
# with Increasing magnitude, softmax gives answers closer to 1 and 0
print(softmax(np.array(scores) * 10))
# with decreasing magnitude, it gives answers near to uniform distribution
print(softmax(np.array(scores) / 10))
# Plot softmax curves
import matplotlib.pyplot as plt
x = np.arange(-2.0, 6.0, 0.1)
scores = np.vstack([x, np.ones_like(x), 0.2 * np.ones_like(x)])
print (scores)
plt.plot(x, softmax(scores).T, linewidth=2)
plt.show()
