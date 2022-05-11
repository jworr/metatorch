module Layer.Rnn
(
   rnn,
   lstm,
   gru,
   rnnBi,
   lstmBi,
   gruBi,
   rnnLast,
   lstmLast,
   gruLast,
   rnnBiLast,
   lstmBiLast,
   gruBiLast
)
where

import Control.Monad.Writer
import Data.List
import Data.Either (isRight)
import Text.Printf (printf)

import Tensor (Tensor, ETensor, dim, fromDim, isScalar)
import Layer(Layer(..), Flow, record)
import Dim (Dim, multiply, lit)

rnn  = genRnn "RNN"
lstm = genRnn "LSTM"
gru  = genRnn "GRU"

rnnBi  = genBiRnn "RNN"
lstmBi = genBiRnn "LSTM"
gruBi  = genBiRnn "GRU"

rnnLast  = genLastRnn "RNN"
lstmLast = genLastRnn "LSTM"
gruLast  = genLastRnn "GRU"

rnnBiLast  = genBiLastRnn "RNN"
lstmBiLast = genBiLastRnn "LSTM"
gruBiLast  = genBiLastRnn "GRU"


{- Models a single directional RNN, outputs at each timestep -}
genRnn :: String -> Dim -> ETensor -> Flow
genRnn name hidden input = log $ input >>= (rnnChk name hidden)
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

{- Models a single directional RNN, that outputs the last timesteps vector -}
genLastRnn :: String -> Bool -> Dim -> ETensor -> Flow
genLastRnn name batchFst hidden input = log $ 
   input >>= (lastRnnChk name batchFst (lit 1) hidden) 

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
genBiRnn :: String -> Dim -> ETensor -> Flow
genBiRnn name hidden input = log $ input >>= (rnnChk name bi)
   where log = record $ RNN name bi True
         bi = multiply (lit 2) hidden

{- Models a Bi-Directional RNN that outputs two "last" vectors -}
genBiLastRnn :: String -> Bool -> Dim -> ETensor -> Flow
genBiLastRnn name batchFst hidden input = log $ 
   input >>= (lastRnnChk name batchFst (lit 2) hidden) 

   where log = record $ RNNLast name hidden True batchFst
