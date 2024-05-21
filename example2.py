import numpy as np

from rsdl import Tensor
from rsdl.layers import Linear
from rsdl.optim import SGD
from rsdl.losses import loss_functions

X = Tensor(np.random.randn(100, 3))
coef = Tensor(np.array([-7, +3, -9]))
y = X @ coef + 5

l = Linear(3, 1)

optimizer = SGD([l])

learning_rate = 0.1
batch_size = 5

for epoch in range(100):
    
    epoch_loss = 0.0
    
    for start in range(0, 100, batch_size):
        end = start + batch_size


        inputs = X[start:end]

        predicted = l.forward(inputs)

        actual = y[start:end]
        actual.data = actual.data.reshape(batch_size, 1)
        loss = loss_functions.MeanSquaredError(predicted, actual)
        
        loss.backward()
        
        epoch_loss += loss

        optimizer.step()
        l.zero_grad()
        
print(l.weight)
print(l.bias)
 
