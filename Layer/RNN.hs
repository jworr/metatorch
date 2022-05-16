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

module Layer.RNN
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

import Text.Printf (printf)

import Tensor (Tensor, ETensor, dim, fromDim)
import Layer(Layer(..), Flow, record)
import Dim (Dim, multiply, lit)

rnn :: Dim -> ETensor -> Flow
rnn  = genRnn "RNN"

lstm :: Dim -> ETensor -> Flow
lstm = genRnn "LSTM"

gru :: Dim -> ETensor -> Flow
gru = genRnn "GRU"

rnnBi :: Dim -> ETensor -> Flow
rnnBi = genBiRnn "RNN"

lstmBi :: Dim -> ETensor -> Flow
lstmBi = genBiRnn "LSTM"

gruBi :: Dim -> ETensor -> Flow
gruBi = genBiRnn "GRU"

rnnLast:: Bool -> Dim -> ETensor -> Flow
rnnLast = genLastRnn "RNN"

lstmLast :: Bool -> Dim -> ETensor -> Flow
lstmLast = genLastRnn "LSTM"

gruLast :: Bool -> Dim -> ETensor -> Flow
gruLast = genLastRnn "GRU"

rnnBiLast :: Bool -> Dim -> ETensor -> Flow
rnnBiLast = genBiLastRnn "RNN"

lstmBiLast :: Bool -> Dim -> ETensor -> Flow
lstmBiLast = genBiLastRnn "LSTM"

gruBiLast :: Bool -> Dim -> ETensor -> Flow
gruBiLast  = genBiLastRnn "GRU"


{- Models a single directional RNN, outputs at each time step -}
genRnn :: String -> Dim -> ETensor -> Flow
genRnn name hidden input = wrt $ input >>= (rnnChk name hidden)
   where wrt = record $ RNN name hidden False

{- Checks that the RNN is applied to the give time series -}
rnnChk :: String -> Dim -> Tensor -> ETensor
rnnChk name hidden input
   | has 3     = out ((init size) ++ [hidden])
   | has 2     = out [head size, hidden]
   | otherwise = Left $ printf rnnChkMsg name

   where has n = (length $ dim input) == n
         size  = dim input
         out d = Right $ fromDim d

rnnChkMsg :: String
rnnChkMsg = unlines ["%s Layer: input must be 2 (length, H)",
   "or (batch, length, h) or (length, batch, h)"]

{- Models a single directional RNN, that outputs the last time steps vector -}
genLastRnn :: String -> Bool -> Dim -> ETensor -> Flow
genLastRnn name batchFst hidden input = wrt $ 
   input >>= (lastRnnChk name batchFst (lit 1) hidden) 

   where wrt = record $ RNNLast name hidden False batchFst

{- checks and applies a RNN that outputs the last time steps vector -}
lastRnnChk :: String -> Bool -> Dim -> Dim -> Tensor -> ETensor
lastRnnChk name batchFst fstOut hidden input
   | has 3     = out [fstOut, batch, hidden]
   | has 2     = out [fstOut, hidden]
   | otherwise = Left $ printf rnnChkMsg name

   where dims = dim input
         has n = (length dims) == n
         out d = Right $ fromDim d
         batch = if batchFst then head dims else head $ tail dims

{- Models a Bi-Directional RNN, outputs at each time step -}
genBiRnn :: String -> Dim -> ETensor -> Flow
genBiRnn name hidden input = wrt $ input >>= (rnnChk name bi)
   where wrt = record $ RNN name bi True
         bi = multiply (lit 2) hidden

{- Models a Bi-Directional RNN that outputs two "last" vectors -}
genBiLastRnn :: String -> Bool -> Dim -> ETensor -> Flow
genBiLastRnn name batchFst hidden input = wrt $ 
   input >>= (lastRnnChk name batchFst (lit 2) hidden) 

   where wrt = record $ RNNLast name hidden True batchFst
