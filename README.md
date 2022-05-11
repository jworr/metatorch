# Metatorch

Metatorch is a library and framework for prototyping PyTorch models with Haskell.
The idea is to enable developers to quickly layout and verify the design of a model.
Specially, the trouble with PyTorch is often ensuring that the dimensions of different neural network and tensor operations match correctly.
The goal to enable developers to rapidly prototype and successfully implement a PyTorch model with a few runtime errors as possible.

Metatorch allows developers to quickly find errors in their model by writing just a few lines of code.
Compiling and running the prototype of the model, will either succeed and show a outline of the mode or will provide an error message on exactly which operation failed.
The error message is based on the PyTorch documentation and should guide the developer towards a correction.

Metatorch is a work in progress and is a proof of concept at this point.
More Pytorch layers and operations will be implemented in the near future.
Ultimately the goal of the project is to create a code generator that produces Python classes based on a prototype.

Look in `Example.hs` for example models. Compile the example with the following command: `ghc --make Example -outputdir=lib`. Run the example with the command `./Example`


## TO-DOs

* Convolutional Layers
* Pooling Layers
* More loss functions such as MSE
* Multi-layer option for RNNs
* `join` function to connect two separate network flows
* PyTorch code generator
