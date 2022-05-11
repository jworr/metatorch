module Layer (
   Layer(..),
   Flow,
   start,
   generate,
   record,
   crossEnt,
   linear,
   act,
   permute,
   squeeze
)
where

import Control.Monad.Writer
import Data.List
import Data.Either (isRight)
import Text.Printf (printf)

import Tensor (Tensor(..), ETensor, Dim, dim, fromDim, isScalar)

--TODOs basics: product, sum, squeeze, dim sum, reshape
--TODOs 2D Conv, Max/Avg pool, concat, MSE, BCE
--TODO "join" for two different flows

data Layer = Linear Dim Dim

           --name, hidden layer, bidirectional
           | RNN String Dim Bool

           --name, hidden layer, bi, batch
           | RNNLast String Dim Bool Bool

           | Permute [Int] 
           | Squeeze Int
           | Activation String
           | CELoss Dim
           | Broken String

instance Show Layer where

   show (Linear d1 d2)    = printf "Linear %s %s" d1 d2
   show (RNN name d bi)   = printf "%s%s %s" (if bi then "bi-" else "") name d
   
   show (RNNLast name d bi ba) = printf "Last %s%s %s%s" biMsg name d baMsg
      where biMsg = if bi then "bi-" else ""
            baMsg = if ba then " batch first" else ""

   show (Activation name) = name
   show (CELoss d)        = printf "Cross Entropy %s" d
   show (Broken msg)      = printf "Error: %s" msg
   show (Permute ord)     = printf "Permute: %s" (intercalate ","$ map show ord)
   show (Squeeze d)       = printf "Squeeze: %d" d

--principle type, an operation in the flow of network
type Flow = Writer [Layer] ETensor


{- Applies the Loss layer in the flow of the network -}
--consider switching back to ETensor from Tensor that is:
--crossEnt :: Dim -> ETensor -> ETensor -> Flow
crossEnt :: Dim -> Tensor -> ETensor -> Flow
crossEnt numClasses target input = log $ input >>= loss
   where log = record (CELoss numClasses)
         loss = crossEntChk numClasses target

{- Applies cross entropy loss -}
crossEntChk :: Dim -> Tensor -> Tensor -> ETensor
crossEntChk numClasses target input
  
   --input (C) target ()
   | inDim == 1 && tarDim == 0 && isNc inputDims = out

   --input (N, C) target (N)
   | inDim == 2 && tarDim == 1 && sameFirst && (isNc $ tail inputDims) = out

   --input (N, C, d1, d2, ..) target (N, d1, d2, ...)
   | inDim >= 3 && tarDim >= 2 && sameFirst && sameEnd && (isNc $ tail inputDims) = out

   --error
   | otherwise = Left $ msg (fmtDim inputDims) (fmtDim targetDims) numClasses

   where inputDims  = dim input
         targetDims = dim target
         inDim      = length inputDims
         tarDim     = length targetDims
         out        = Right target
         sameFirst  = head inputDims == head targetDims
         sameEnd    = (tail $ tail inputDims) == tail targetDims
         isNc d     = numClasses == head d
         msg        = printf crossEntError

crossEntError = unlines ["Cross Ent: input %s does not match target %s with %s classes",
   "Should be one of the following:",
   "input (C) target ()",
   "input (N, C) target (N)",
   "input (N, C, d1, d2,...) target (N, d1, d2,...)"]

       
{- Applies a linear NN layer -}
linearChk :: Dim -> Dim -> Tensor -> ETensor
linearChk inSize outSize tensor
   | notScalar && (last size == inSize)  = Right $ fromDim (init size ++ [outSize])
   | otherwise                           = Left $ msg (show tensor) inSize
   where notScalar = not $ isScalar tensor
         size = dim tensor
         msg = printf "Linear Layer: last dimension of %s does not match input size %d"


{- Applies the linear layer in the flow of the network -}
linear :: Dim -> Dim -> ETensor -> Flow
linear inSize outSize eTen = log $ eTen >>= (linearChk inSize outSize)
   where log = record $ Linear inSize outSize

{- Applies an activation function -}
act :: String -> ETensor -> Flow
act name = record $ Activation name

{- Squeezes out a dimension of the tensor -}
squeeze :: Int -> ETensor -> Flow
squeeze sDim tensor = log $ tensor >>= (squeezeChk sDim)
   where log = record $ Squeeze sDim

{- Applies and checks that the squeeze opertion can be applied to the tensor-}
squeezeChk :: Int -> Tensor -> ETensor
squeezeChk sDim tensor 
   | inRange && isOne  = Right $ fromDim newDims
   | otherwise         = Left $ msg
   
   where dims = dim tensor
         inRange = sDim < (length dims) && sDim >= 0
         isOne = (dims !! sDim) == "1" --TODO fix hack!

         --split at the index given
         (pre, post) = splitAt sDim dims

         --the dimension to drop will be the first of "post"
         newDims     = pre ++ (tail post)

         msg = printf "Squeeze: cannot remove %d from %s" sDim (fmtDim dims)
     

{- Perumutes the dimensions of the tensor -}
permute :: [Int] -> ETensor -> Flow
permute order tensor = log $ tensor >>= (permuteChk order)
   where log = record $ Permute order

{- Applies a permuation of the tensors dimension -}
permuteChk :: [Int] -> Tensor -> ETensor
permuteChk order tensor 
   | match && valid = Right $ fromDim newDims
   | otherwise      = Left $ msg
   
   where 
         dims = dim tensor
         inDims = length $ dims
         match = length order == inDims
         valid = (maximum order) < inDims && (minimum order) == 0
         msg = printf "Permute: %s does not match input shape %s" ord (fmtDim dims)
         ord = intercalate "," $ map show order
         newDims = map (dims !!) order

{- Records the errors -}
record :: Layer -> ETensor -> Flow
record layer (Right ten) = writer (Right ten, [layer])    --success
record _ (Left "")       = writer (Left "", [])           --no op
record _ (Left err)      = writer (Left "", [Broken err]) --record and clear error

{- Initialize the network flow -}
start :: Tensor -> Flow
start ten = writer (Right ten, [])

{- Convert the flow of the network into a String to be printed -}
generate :: Flow -> String
generate flow = printf "Final Output: %s\n\nLayers:\n%s" fmtOut fmtLayer

   where (output, layers) = runWriter flow
         fmtLayer = unlines $ map show layers
         fmtOut   = case output of
                     Right tensor -> show tensor
                     Left error   -> error

{- Format the dimensions of the tensor -}
fmtDim :: [Dim] -> String
fmtDim = intercalate "x"
