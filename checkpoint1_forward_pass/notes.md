# Checkpoint 1 — Forward pass

## Task
Implement `y_pred = X @ W + b` and MSE loss using raw NumPy.

## Shapes (work this out before coding)
- X: (n_samples, n_features) = (5, 3)
- W: (n_features, n_outputs) = (3, 1)
- b: (1, n_outputs) = (1, 1)
- X @ W: ? 
    - (5,1)
- (X @ W) + b: ? (how does broadcasting make this work?)
    - So when the shapes don't match, broadcasting takesplaxe which is the process of "strectching" the bias by duplicating the the single row 5 times in this case. So the shape of the b will be (5,1)
- y_pred: (5,1)

## My derivation
So foward propagation is a process where input data passes through each layer of a NN to produce an output. It's sused to make predictions.
pred = Weights * input_vector + bias 


## Result
