{-
   Copyright 2022 J. Walker Orr

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
-}

module Metatorch.Layer (
   Layer(..),
   Flow,
   start,
   input,
   check,
   record,
   crossEnt,
   mseLoss,
   linear,
   relu,
   sigmoid,
   tanhAct,
   permute,
   squeeze,
   unsqueeze,
   reshape,
   mean,
   maximize
)
where

import Control.Monad.Writer
import Data.List (map, intercalate)
import Text.Printf (printf)

import Metatorch.Dim (Dim, lit, multiplyAll, multiply, addAll)
import qualified Metatorch.Dim as D
import Metatorch.Tensor (Tensor, ETensor, dim, fromDim, isScalar)


data Layer = Linear Dim Dim

           --name, input features, hidden layer, bidirectional
           | RNN String Dim Dim Bool

           --name, input channels, hidden layer, bi, batch
           | RNNLast String Dim Dim Bool Bool
           
           --input channels, channels, window (kernel) size, stride, pad
           | Conv1d Dim Dim Int Int Int

           --name, window, stride, pad
           | Pool1d String Int Int Int

           --channels, window size, stride, pad
           | Conv2d Dim Dim Int Int Int

           --name, window, stride, pad
           | Pool2d String Int Int Int

           --vocab size, embeding dim
           | Embedding Dim Dim

           | Average Int
           | Max Int
           | Permute [Int] 
           | Squeeze Int
           | UnSqueeze Int
           | Reshape [Dim]
           | Activation String

           --number of classes
           | CELoss Dim  
           | MSELoss 
           | Input [Dim]
           | Broken String
           | Sequential [Layer]
           deriving (Ord, Eq)

instance Show Layer where

   show (Linear i d)           = printf "Linear %s" (fmtTrans i d)
   show (RNN name i d True)    = printf "bi-%s %s" name (fmtTrans i d)
   show (RNN name i d False)   = printf "%s %s" name (fmtTrans i d)
   
   show (RNNLast name i d False False) = printf "Last-%s %s" name (fmtTrans i d)
   show (RNNLast name i d True False)  = printf "Last-bi-%s %s" name (fmtTrans i d)
   show (RNNLast name i d False True)  = printf "Last %s %s (batch first)" name (fmtTrans i d)
   show (RNNLast name i d True True)   = printf "Last-bi-%s %s (batch first)" name (fmtTrans i d)

   show (Conv1d i d w s p)  = printf "Conv1D %s, window %d, stride %d, pad %d" (fmtTrans i d) w s p
   show (Conv2d i d w s p)  = printf "Conv2D %s, window %d, stride %d, pad %d" (fmtTrans i d) w s p

   show (Embedding v d)     = printf "Embeding vocab %s, embedding size %s" (show v) (show d)

   show (Activation name)   = name
   show (CELoss d)          = printf "Cross Entropy %s" (show d)
   show MSELoss             = printf "MSE Loss"
   show (Broken msg)        = printf "Error: %s" msg
   show (Permute ord)       = printf "Permute: %s" (intercalate "," $ map show ord)
   show (Squeeze d)         = printf "Squeeze: %d" d
   show (UnSqueeze d)       = printf "UnSqueeze: %d" d
   show (Reshape ds)        = printf "Reshape: %s" (fmtDim ds)
   show (Pool1d name d s p) = printf "%s-pooling-1d %d %d %d" name d s p
   show (Pool2d name d s p) = printf "%s-pooling-2d %d %d %d" name d s p
   show (Average d)         = printf "Mean %d" d
   show (Max d)             = printf "Max %d" d
   show (Input d)           = printf "Input %s" (fmtDim d)
   show (Sequential l)      = printf "Sequential (%d layers)" (length l)


fmtTrans :: Dim -> Dim -> String
fmtTrans inFeats outFeats = printf "%s -> %s" (show inFeats) (show outFeats)

--principle type, an operation in the flow of network
type Flow = Writer [(Layer, [Dim])] ETensor


{- Applies the Loss layer in the flow of the network -}
crossEnt :: Dim -> Tensor -> ETensor -> Flow
crossEnt numClasses target tensor = 
   let
      wrt = record (CELoss numClasses)
      loss = crossEntChk numClasses target
   in
      wrt $ tensor >>= loss

{- Applies cross entropy loss -}
crossEntChk :: Dim -> Tensor -> Tensor -> ETensor
crossEntChk numClasses target tensor 
  
   --input (C) target ()
   | inDim == 1 && tarDim == 0 && isNc inputDims = out

   --input (N, C) target (N)
   | inDim == 2 && tarDim == 1 && sameFirst && (isNc $ tail inputDims) = out

   --input (N, C, d1, d2, ..) target (N, d1, d2, ...)
   | inDim >= 3 && tarDim >= 2 && sameFirst && sameEnd && (isNc $ tail inputDims) = out

   --error
   | otherwise = Left $ msg (fmtDim inputDims) (fmtDim targetDims) (show numClasses)

   where inputDims  = dim tensor 
         targetDims = dim target
         inDim      = length inputDims
         tarDim     = length targetDims
         out        = Right target
         sameFirst  = head inputDims == head targetDims
         sameEnd    = (tail inputDims) == tail targetDims
         isNc d     = numClasses == head d
         msg        = printf crossEntError

crossEntError :: String
crossEntError = unlines ["Cross Ent: input %s does not match target %s with %s classes",
   "Should be one of the following:",
   "input (C) target ()",
   "input (N, C) target (N)",
   "input (N, C, d1, d2,...) target (N, d1, d2,...)"]


{- Applies the MSE loss layer to the flow of the network -}
mseLoss :: Tensor -> ETensor -> Flow
mseLoss target inputTensor = record MSELoss $ inputTensor >>= mseCheck target


{- Check that input to MSE matches the expected dimension -}
mseCheck :: Tensor -> Tensor -> ETensor
mseCheck target input 
   
   -- check that the input and target dimensions are the same
   | inDim == targetDim = Right target

   -- create an error message
   | otherwise = Left $ printf msg (fmtDim inDim) (fmtDim targetDim)

   where inDim     = dim input
         targetDim = dim target
         msg       = "MSE: input %s does not match the target %s"


{- Applies the linear layer in the flow of the network -}
linear :: Dim -> Dim -> ETensor -> Flow
linear inSize outSize eTen = 
   let
      wrt = record $ Linear inSize outSize
   in
      wrt $ eTen >>= (linearChk inSize outSize)
       
{- Applies a linear NN layer -}
linearChk :: Dim -> Dim -> Tensor -> ETensor
linearChk inSize outSize tensor
   | notScalar && (last size == inSize)  = Right $ fromDim (init size ++ [outSize])
   | otherwise                           = Left $ msg (fmtDim size) (show inSize)
   
   where notScalar = not $ isScalar tensor
         size = dim tensor
         msg = printf "Linear Layer: last dimension of %s does not match expected input size %s"


{- Applies ReLU as a activation layer-}
relu :: ETensor -> Flow
relu = act "ReLU"

{- Applies Sigmoid as a activation layer-}
sigmoid :: ETensor -> Flow
sigmoid = act "Sigmoid"

{- Applies Tanh as a activation layer-}
tanhAct :: ETensor -> Flow
tanhAct = act "Tanh"

{- Applies an activation function -}
act :: String -> ETensor -> Flow
act name = record $ Activation name

{- Squeezes out a dimension of the tensor -}
squeeze :: Int -> ETensor -> Flow
squeeze sDim tensor = record (Squeeze sDim) $ tensor >>= (squeezeChk sDim)

{- Applies and checks that the squeeze operation can be applied to the tensor-}
squeezeChk :: Int -> Tensor -> ETensor
squeezeChk sDim tensor 
   | inRange && isOne  = Right $ removeDim sDim tensor
   | otherwise         = Left $ msg
   
   where dims = dim tensor
         inRange = sDim < (length dims) && sDim >= 0
         isOne = (dims !! sDim) == lit 1
         msg = printf "Squeeze: cannot remove dimension %d from %s" sDim (fmtDim dims)

{- Unsqueezes (adds/creates) a dimension to the tensor -}
unsqueeze :: Int -> ETensor -> Flow
unsqueeze sDim tensor = record (UnSqueeze sDim) $ tensor >>= (unsqueezeChk sDim)

{- Applies and checks that the unsqueeze operation can be applied to the tensor -}
unsqueezeChk :: Int -> Tensor -> ETensor
unsqueezeChk sDim tensor
   | 0 <= sDim && sDim <= length dims  = Right $ addDim sDim tensor
   | otherwise                         = Left msg

   where dims = dim tensor
         msg  = printf "Unsqueeze: cannot add a dimension %d to %s" sDim (fmtDim dims)

{- Find the maximum of a dimension -}
maximize :: Int -> ETensor -> Flow
maximize maxDim tensor = record (Max maxDim) $ tensor >>= (maxChk maxDim)

{- Checks that the max operation can be applied -}
maxChk :: Int -> Tensor -> ETensor
maxChk maxDim tensor
   | maxDim < (length dims) && maxDim >= 0 = Right $ removeDim maxDim tensor
   | otherwise                             = Left $ msg
  
  where dims = dim tensor
        msg = printf "Max: cannot remove dimension %d from %s" maxDim (fmtDim dims)

{- Average out a dimension -}
mean :: Int -> ETensor -> Flow
mean avgDim tensor = record (Average avgDim) $ tensor >>= (meanChk avgDim)

{- Checks that average matches the tensor -}
meanChk :: Int -> Tensor -> ETensor
meanChk avgDim tensor
   | avgDim < (length dims) && avgDim >= 0 = Right $ removeDim avgDim tensor
   | otherwise                             = Left $ msg
  
  where dims = dim tensor
        msg = printf "Mean: cannot remove dimension %d from %s" avgDim (fmtDim dims)

{- Remove the dimension from the tensor -}
removeDim :: Int -> Tensor -> Tensor
removeDim sDim tensor = 
   let 
      --split at the index given
      (pre, post) = splitAt sDim (dim tensor)
         
      --the dimension to drop will be the first of "post"
      newDims     = pre ++ (tail post)
   in
      fromDim newDims

{- Add a dimension to the tensor -}
addDim :: Int -> Tensor -> Tensor
addDim aDim tensor =
   let
      --split at the index given
      (pre, post) = splitAt aDim (dim tensor)

      --add the dimension
      newDims = pre ++ [lit 1] ++ post
   in
      fromDim newDims

{- Permutes the dimensions of the tensor -}
permute :: [Int] -> ETensor -> Flow
permute order tensor = 
   let
      wrt = record $ Permute order
   in
      wrt $ tensor >>= (permuteChk order)

{- Applies a permutation of the tensors dimension -}
permuteChk :: [Int] -> Tensor -> ETensor
permuteChk order tensor 
   | match && valid = Right $ fromDim newDims
   | otherwise      = Left $ msg
   
   where dims = dim tensor
         inDims = length $ dims
         match = length order == inDims
         valid = (maximum order) < inDims && (minimum order) == 0
         msg = printf "Permute: %s does not match input shape %s" ord (fmtDim dims)
         ord = intercalate "," $ map show order
         newDims = map (dims !!) order

{- Reshapes the given tensor -}
reshape :: [Dim] -> ETensor -> Flow
reshape newShape tensor = 
   record (Reshape newShape) $ tensor >>= (reshapeChk newShape)

{- Ensures that the same number of elements exist between the input
and output tensors-}
reshapeChk :: [Dim] -> Tensor -> ETensor
reshapeChk newSize tensor
   | matchSize = Right $ fromDim newSize
   | otherwise = Left  $ printf reshapeMsg (fmtDim newSize) (fmtDim inDims)

   where inDims    = dim tensor
         matchSize = (multiplyAll newSize) == (multiplyAll inDims)

reshapeMsg :: String
reshapeMsg = "Reshape: product of input %s and product of output %s do not match"

{- Records the errors -}
record :: Layer -> ETensor -> Flow
record layer (Right ten) = writer (Right ten, [(layer, dim ten)]) --success
record _ (Left "")       = writer (Left "", [])                   --no op
record _ (Left err)      = writer (Left "", [(Broken err, [])])   --record and clear error

{- Initializing the network flor  -}
input :: [Dim] -> Flow
input dims = start $ fromDim dims

{- Initialize the network flow -}
start :: Tensor -> Flow
start ten = 
   let
      dims = dim ten
   in
      writer (Right ten, [(Input dims, dims)])

{- Convert the flow of the network into a String, either a summary of the 
   flow with dimensions or an error message -}
check :: Flow -> String
check flow = 
   let
      (output, layers)    = runWriter flow
      fmtLayers           = unlines $ map fmtPair layers
      fmtOut              = case output of
                               Right tensor -> show tensor
                               Left err      -> err
      fmtPair (layer, ds) =  printf "%-60s %s" (show layer) (fmtDim ds)
      numParams           = allParameters (map fst layers)
   in
      unlines ["Final Output: " ++ fmtOut,
               printf "Number of parameters: %s" (show numParams),
               "",
               printf "%-60s %s" "Layers:" "Output Dimensions",
               fmtLayers
              ]

{- Format the dimensions of the tensor -}
fmtDim :: [Dim] -> String
fmtDim dims = intercalate "x" (map show dims)


{- Returns the number of parameters across all the layers -}
allParameters :: [Layer] -> Dim
allParameters = foldr D.add (lit 0) . map parameters

--NOTE: the use of strings here is gross, add RNN data type
{- Returns the number of parameters in the layer -}
parameters :: Layer -> Dim

--linear
parameters (Linear inDim outDim) = multiply inDim outDim

--'Vanilla' RNN
parameters (RNN "RNN" inDim hiddenDim bi) = 
   let
      input  = multiply inDim hiddenDim
      hidden = multiply hiddenDim hiddenDim
      total  = addAll [input, hidden, hiddenDim, hiddenDim]
   in
      if bi then multiply (lit 2) total else total

--GRU
parameters (RNN "GRU" inDim hiddenDim bi) = 
   let
      --3 input transformations
      inGates = multiplyAll [(lit 3), inDim, hiddenDim]

      --3 hidden "memory" transformations
      memGates = multiplyAll [(lit 3), hiddenDim, hiddenDim]

      --3 bias per input and memory
      bias = multiply (lit 6) hiddenDim

      total = addAll [inGates, memGates, bias] 
   in
      if bi then multiply (lit 2) total else total 

--LSTM
parameters (RNN "LSTM" inDim hiddenDim bi) =
   let
      --4 input transformations
      inGates = multiplyAll [(lit 4), inDim, hiddenDim]

      --4 hidden "memory" transformations
      memGates = multiplyAll [(lit 4), hiddenDim, hiddenDim]

      --4 bias per input and memory
      bias = multiply (lit 8) hiddenDim

      total = addAll [inGates, memGates, bias] 
   in
      if bi then multiply (lit 2) total else total 
      

--Layer that produces last vector, same as RNN
parameters (RNNLast name inDim hiddenDim bi _) = parameters (RNN name inDim hiddenDim bi)

--1D convolution
parameters (Conv1d inChannels outChannels kernelSize _ _) = 
   let
      trans = multiplyAll [inChannels, outChannels, lit kernelSize]
   in
      addAll [trans, outChannels]

--2D convolution
parameters (Conv2d inChannels outChannels kernelSize _ _) = 
   let
      trans = multiplyAll [inChannels, outChannels, lit kernelSize, lit kernelSize]
   in
      addAll [trans, outChannels]

--Sequential layer, add up all sub layers
parameters (Sequential layers) = foldr D.add (lit 0) (map parameters layers)

parameters _ = lit 0
