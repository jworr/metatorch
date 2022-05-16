module Layer.Cnn
(
   conv1d,
   maxPool1d,
   avgPool1d
)
where

import Text.Printf (printf)

import Tensor (Tensor, ETensor, dim, fromDim)
import Layer(Layer(..), Flow, record)
import Dim (Dim, add, sub, divBy, lit)

{- Applies a 1D Convolutional layer -}
conv1d :: Dim -> Dim -> Int -> Int -> ETensor -> Flow
conv1d inChl outChl window stride input 
   = record (Conv1d outChl window stride) $
      input >>= (conv1dChk "Conv" inChl outChl window stride)

{-- Applies max pooling --}
maxPool1d :: Dim -> Int -> Int -> ETensor -> Flow
maxPool1d = pooling1d "Max"

{-- Applies max pooling --}
avgPool1d :: Dim -> Int -> Int -> ETensor -> Flow
avgPool1d = pooling1d "Avg"

{- Applies a Pooling Layer -}
pooling1d :: String -> Dim -> Int -> Int -> ETensor -> Flow
pooling1d name channels window stride input
   = record (Pool1d name window stride) $
      input >>= (conv1dChk name channels channels window stride)

{- Checks that CNN is applied to the time series -}
conv1dChk :: String -> Dim -> Dim -> Int -> Int -> Tensor -> ETensor
conv1dChk name inChl outChl window stride tensor
   | has 2 && inChl == (head size)        = Right $ newDim size
   | has 3 && inChl == (head $ tail size) = Right $ newDim size
   | otherwise                            = Left  $ printf erMsg name (show inChl)

   where has   = hasDims tensor
         size  = dim tensor
         newDim [_, len]    = fromDim [outChl, outputLen len window stride]
         newDim [n, _, len] = fromDim [n, outChl, outputLen len window stride]
         newDim d           = fromDim d

erMsg :: String
erMsg = unlines ["%s1d: input channels %s do not match either have:",
   "shape 3 (batch, channel in, length) or shape 2 (channel in, length)"]


{- Computes the output length -}
outputLen :: Dim -> Int -> Int -> Dim
outputLen len window stride = (numerator `divBy` stride) `add` (lit 1)

   --Length - (window - 1) - 1 from the docs, simplified because of no
   --padding or dilation
   where numerator = len `sub` (lit window)

{- Returns true if the tensor has a specified number of dimensions -}
hasDims :: Tensor -> Int -> Bool
hasDims tensor target = (length $ dim tensor) == target
