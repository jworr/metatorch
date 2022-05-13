module Layer.Cnn
(
   conv1d
)
where

import Control.Monad.Writer
import Data.List
import Text.Printf (printf)

import Tensor (Tensor, ETensor, dim, fromDim)
import Layer(Layer(..), Flow, record)
import Dim (Dim, sub, lit)


conv1d :: Dim -> Dim -> Int -> ETensor -> Flow
conv1d inChl outChl window input = log $ input >>= (conv1dChk inChl outChl window)
   where log = record $ Conv1d outChl window

{- Checks that CNN is applied to the time series -}
conv1dChk :: Dim -> Dim -> Int -> Tensor -> ETensor
conv1dChk inChl outChl window tensor
   | has 2 && inChl == (head size)          = Right $ newDim size
   | has 3 && inChl == (head $ tail size)   = Right $ newDim size
   | otherwise                              = Left  $ printf c1Msg (show inChl)

   where has   = hasDims tensor
         size  = dim tensor
         newDim [_, len]    = fromDim [outChl, len `sub` (lit $ window - 1)]
         newDim [n, _, len] = fromDim [n, outChl, len `sub` (lit $ window - 1)]


c1Msg = unlines ["Conv1d: input channels %d do not match either have:",
   "shape 3 (batch, channel in, length) or shape 2 (channel in, length)"]


{- Returns true if the tensor has a specified number of dimensions -}
hasDims :: Tensor -> Int -> Bool
hasDims tensor target = (length $ dim tensor) == target
