# Based on code from https://github.com/miloharper/simple-neural-network/blob/master/short_version.py

from numpy import exp, array, random, dot

training_inputs = array([[0,0,1],[1,1,1],[1,0,1],[0,1,1]])
training_outputs = array([[0,1,1,0]]).T
random.seed(1)
weights = 2 * random.random((3,1)) - 1

count = 0
maxError = 1

while maxError > 0.01:
    outputs = 1 / (1 + exp(-(dot(training_inputs,weights))))
    error = training_outputs - outputs
    adjust = dot(training_inputs.T, error * outputs * (1 - outputs))
    weights += adjust
    maxError = max(error)
    count += 1

print(outputs)
print(count)

