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
   squeeze,
   reshape
)
where

import Control.Monad.Writer
import Data.List
import Text.Printf (printf)

import Dim (Dim, lit, multiplyAll)
import Tensor (Tensor, ETensor, dim, fromDim, isScalar)


data Layer = Linear Dim Dim

           --name, hidden layer, bidirectional
           | RNN String Dim Bool

           --name, hidden layer, bi, batch
           | RNNLast String Dim Bool Bool
           
           --channels, window (kernel) size
           | Conv1d Dim Int

           | Permute [Int] 
           | Squeeze Int
           | Reshape [Dim]
           | Activation String
           | CELoss Dim
           | Broken String

instance Show Layer where

   show (Linear d1 d2)       = printf "Linear %s %s" (show d1) (show d2)
   show (RNN name d True)    = printf "bi-%s %s" name (show d)
   show (RNN name d False)   = printf "%s %s" name (show d)
   
   show (RNNLast name d False False) = printf "Last-%s %s" name (show d)
   show (RNNLast name d True False)  = printf "Last-bi-%s %s" name (show d)
   show (RNNLast name d False True)  = printf "Last %s %s (batch first)" name (show d)
   show (RNNLast name d True True)   = printf "Last-bi-%s %s (batch first)" name (show d)

   show (Conv1d d w)      = printf "Conv1D %s %d" (show d) w

   show (Activation name) = name
   show (CELoss d)        = printf "Cross Entropy %s" (show d)
   show (Broken msg)      = printf "Error: %s" msg
   show (Permute ord)     = printf "Permute: %s" (intercalate "," $ map show ord)
   show (Squeeze d)       = printf "Squeeze: %d" d
   show (Reshape ds)      = printf "Reshape: %s" (fmtDim ds)

--principle type, an operation in the flow of network
type Flow = Writer [(Layer, [Dim])] ETensor


{- Applies the Loss layer in the flow of the network -}
--consider switching back to ETensor from Tensor that is:
--crossEnt :: Dim -> ETensor -> ETensor -> Flow
crossEnt :: Dim -> Tensor -> ETensor -> Flow
crossEnt numClasses target input = wrt $ input >>= loss
   where wrt = record (CELoss numClasses)
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
   | otherwise = Left $ msg (fmtDim inputDims) (fmtDim targetDims) (show numClasses)

   where inputDims  = dim input
         targetDims = dim target
         inDim      = length inputDims
         tarDim     = length targetDims
         out        = Right target
         sameFirst  = head inputDims == head targetDims
         sameEnd    = (tail $ tail inputDims) == tail targetDims
         isNc d     = numClasses == head d
         msg        = printf crossEntError

crossEntError :: String
crossEntError = unlines ["Cross Ent: input %s does not match target %s with %s classes",
   "Should be one of the following:",
   "input (C) target ()",
   "input (N, C) target (N)",
   "input (N, C, d1, d2,...) target (N, d1, d2,...)"]

       
{- Applies a linear NN layer -}
linearChk :: Dim -> Dim -> Tensor -> ETensor
linearChk inSize outSize tensor
   | notScalar && (last size == inSize)  = Right $ fromDim (init size ++ [outSize])
   | otherwise                           = Left $ msg (fmtDim size) (show inSize)
   where notScalar = not $ isScalar tensor
         size = dim tensor
         msg = printf "Linear Layer: last dimension of %s does not match input size %s"


{- Applies the linear layer in the flow of the network -}
linear :: Dim -> Dim -> ETensor -> Flow
linear inSize outSize eTen = wrt $ eTen >>= (linearChk inSize outSize)
   where wrt = record $ Linear inSize outSize

{- Applies an activation function -}
act :: String -> ETensor -> Flow
act name = record $ Activation name

{- Squeezes out a dimension of the tensor -}
squeeze :: Int -> ETensor -> Flow
squeeze sDim tensor = wrt $ tensor >>= (squeezeChk sDim)
   where wrt = record $ Squeeze sDim

{- Applies and checks that the squeeze opertion can be applied to the tensor-}
squeezeChk :: Int -> Tensor -> ETensor
squeezeChk sDim tensor 
   | inRange && isOne  = Right $ fromDim newDims
   | otherwise         = Left $ msg
   
   where dims = dim tensor
         inRange = sDim < (length dims) && sDim >= 0
         isOne = (dims !! sDim) == lit 1

         --split at the index given
         (pre, post) = splitAt sDim dims

         --the dimension to drop will be the first of "post"
         newDims     = pre ++ (tail post)

         msg = printf "Squeeze: cannot remove dimension %d from %s" sDim (fmtDim dims)
     

{- Perumutes the dimensions of the tensor -}
permute :: [Int] -> ETensor -> Flow
permute order tensor = wrt $ tensor >>= (permuteChk order)
   where wrt = record $ Permute order

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

{- Reshapes the given tensor -}
reshape :: [Dim] -> ETensor -> Flow
reshape newShape tensor = log $ tensor >>= (reshapeChk newShape)
   where log = record $ Reshape newShape

{- Ensures that the same number of elements exist between the input
and ouput tensors-}
reshapeChk :: [Dim] -> Tensor -> ETensor
reshapeChk newSize tensor
   | matchSize = Right $ fromDim newSize
   | otherwise = Left  $ printf reshapeMsg (fmtDim newSize) (fmtDim inDims)
   where inDims    = dim tensor
         matchSize = (multiplyAll newSize) == (multiplyAll inDims)

reshapeMsg = "Reshape: product of input %s and product of output %s do not match"

{- Records the errors -}
record :: Layer -> ETensor -> Flow
record layer (Right ten) = writer (Right ten, [(layer, dim ten)]) --success
record _ (Left "")       = writer (Left "", [])                   --no op
record _ (Left err)      = writer (Left "", [(Broken err, [])])   --record and clear error

{- Initialize the network flow -}
start :: Tensor -> Flow
start ten = writer (Right ten, [])

{- Convert the flow of the network into a String to be printed -}
generate :: Flow -> String
generate flow = printf "Final Output: %s\n\n%-30s %s\n%s" fmtOut "Layers:" "Output Dimensions:" fmtLayers

   where (output, layers)    = runWriter flow
         fmtLayers           = unlines $ map fmtPair layers
         fmtOut              = case output of
                                  Right tensor -> show tensor
                                  Left error   -> error
         fmtPair (layer, ds) =  printf "%-30s %s" (show layer) (fmtDim ds)

{- Format the dimensions of the tensor -}
fmtDim :: [Dim] -> String
fmtDim dims = intercalate "x" (map show dims)
