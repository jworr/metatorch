module Layer.Rnn
(
   rnn,
   biRnn,
   lastRnn
)
where

import Control.Monad.Writer
import Data.List
import Data.Either (isRight)
import Text.Printf (printf)

import Tensor (Tensor(..), ETensor, Dim, dim, fromDim, isScalar)
import Layer(Layer(..), Flow, record)


{- Models a single directional RNN, outputs at each timestep -}
rnn :: String -> Dim -> ETensor -> Flow
rnn name hidden input = log $ input >>= (rnnChk name hidden)
   where log = record $ RNN name hidden False

{- Checks that the  -}
rnnChk :: String -> Dim -> Tensor -> ETensor
rnnChk name hidden input
   | has 3     = out ((init size) ++ [hidden])
   | has 2     = out [head size, hidden]
   | otherwise = Left $ printf rnnChkMsg name

   where has n = (length $ dim input) == n
         size  = dim input
         out d = Right $ fromDim d

rnnChkMsg = unlines ["%s Layer: input must be 2 (length, H)",
   "or (batch, length, h) or (length, batch, h)"]

--TODO add layers
{- Models a single directional RNN, that outputs the last timesteps vector -}
lastRnn :: String -> Bool -> Dim -> ETensor -> Flow
lastRnn name batchFst hidden input = log $ 
   input >>= (lastRnnChk name batchFst "1" hidden) --TODO fix HACK

   where log = record $ RNNLast name hidden False batchFst

{- checks and applies a RNN that outputs the last timesteps vector -}
lastRnnChk :: String -> Bool -> Dim -> Dim -> Tensor -> ETensor
lastRnnChk name batchFst fstOut hidden input
   | has 3     = out [fstOut, batch, hidden]
   | has 2     = out [fstOut, hidden]
   | otherwise = Left $ printf rnnChkMsg name

   where dims = dim input
         has n = (length dims) == n
         out d = Right $ fromDim d
         batch = if batchFst then head dims else head $ tail dims

{- Models a Bi-Directional RNN, outputs at each timestep -}
biRnn :: String -> Dim -> ETensor -> Flow
biRnn name hidden input = log $ input >>= (rnnChk name bi)
   where log = record $ RNN name bi True
         --TODO fix this hack!!!!
         bi = "2*" ++ hidden

--TODO bidirectional, first/last
