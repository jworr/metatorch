# Metatorch

Metatorch is a library and framework for prototyping PyTorch models with Haskell.
The idea is to enable developers to layout and verify the design of a model.
Specially, the trouble with PyTorch is often ensuring that the dimensions of different neural network and tensor operations match correctly.
The goal to enable developers to rapidly prototype and successfully implement a PyTorch model with a few runtime errors as possible.

Metatorch allows developers to quickly find errors in their model by writing just a few lines of code.
The library is indented to be used like a domain specific language, no deep knowledge of Haskell is needed.
Compiling and running the prototype of the model will either succeed and show a outline of the mode or will provide an error message on exactly which operation failed.
The idea is that models can be quickly prototype and debugged piece by piece by interactively writing and running the prototype through REPL.
Error messages provided by Metatorch are based on the PyTorch documentation and intended to streamline the development and debugging process.

Metatorch is a work in progress and is a proof of concept at this point.
More Pytorch layers and operations will be implemented in the near future.

Look in `Example.hs` for example models. Compile the example with the following command: `ghc --make Example -outputdir=lib`. Run the example with the command `./Example`

Currently, Metatorch has been tested versus Torch version 1.11

## Example

A bi-directional LSTM model with linear layer to predict one of 4 classes:

```haskell
import Tensor (Tensor(..))
import Layer (check, input, linear, crossEnt)
import Generate (generate)
import Layer.RNN (lstmBi)
import Dim (lit, var, multiply)

n = var "n"
k = var "k"
h = var "h"
h2 = h `multiply` (lit 2)
_4 = lit 4

--makes a prediction per "token" on a single sequence, 4 classes
perTokenPred = input [n, k]
             >>= lstmBi k h
             >>= linear h2 _4
             >>= crossEnt _4 (Vector n)

main :: IO ()
main = do
   
   putStrLn $ check perTokenPred

   putStrLn "---Code---"

   putStrLn $ generate True perTokenPred
```

The output of `check` is
```
Final Output: Vector n

Layers:                                                      Output Dimensions:
Input nxk                                                    nxk
bi-LSTM k -> 2*h                                             nx2*h
Linear 2*h -> 4                                              nx4
Cross Entropy 4                                              n
```

`generate` produces the following code:

```Python
import torch as t
import torch.nn as nn

class Model(nn.Module):

    def __init__(self, k, h):
        super().__init__()

        self.k = k
        self.h = h
        self.lstm = nn.LSTM(k, 2*h, bidirectional=True)
        self.linear = nn.Linear(2*h, 4)

    def forward(self, tensor):

        n, k = tensor.size()
        tensor, _ = self.lstm(tensor)
        tensor = self.linear(tensor)

        return tensor
```

### Catching Errors

Suppose with the same example, a bi-directional LSTM model with linear
layer to predict one of 4 classes.
However suppose that a mistake was made setting up the model, and the developer forgot that a bi-LSTM model will double the hidden layer size since it is concatenating the output of two LSTM passes.
The model would look like this (notice that the linear layer takes an input size of "h"):

```haskell
import Tensor (Tensor(..))
import Layer (check, input, linear, crossEnt)
import Generate (generate)
import Layer.RNN (lstmBi)
import Dim (lit, var, multiply)

n = var "n"
k = var "k"
h = var "h"
h2 = h `multiply` (lit 2)
_4 = lit 4

--makes a prediction per "token" on a single sequence, 4 classes
perTokenPred = input [n, k]
             >>= lstmBi k h
             >>= linear h _4
             >>= crossEnt _4 (Vector n)

main :: IO ()
main = do
   
   putStrLn $ check perTokenPred

   putStrLn "---Code---"

   putStrLn $ generate True perTokenPred
```

The output of the model will be:

```
Final Output:

Layers:                                                      Output Dimensions:
Input nxk                                                    nxk
bi-LSTM k -> 2*h                                             nx2*h
Error: Linear Layer: last dimension of nx2*h does not match expected input size h

---Code---
Could not generate code:
```

The error points out that the linear layer actually received a tensor with since "2h" rather than the expected size "h".
The developer can followup by changing `linear h _4` to `linear h2 _4`.


## TO-DOs

* Documentation!
* More loss functions such as MSE
* Multi-layer option for RNNs
* Add Transformer Layers
* `join` function to connect two separate network flows
* Arithmetic for dimensions is fragile, needs improvement.
