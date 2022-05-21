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

rnn :: Dim -> Dim -> ETensor -> Flow
rnn  = genRnn "RNN"

lstm :: Dim -> Dim -> ETensor -> Flow
lstm = genRnn "LSTM"

gru :: Dim -> Dim -> ETensor -> Flow
gru = genRnn "GRU"

rnnBi :: Dim -> Dim -> ETensor -> Flow
rnnBi = genBiRnn "RNN"

lstmBi :: Dim -> Dim -> ETensor -> Flow
lstmBi = genBiRnn "LSTM"

gruBi :: Dim -> Dim -> ETensor -> Flow
gruBi = genBiRnn "GRU"

rnnLast:: Bool -> Dim -> Dim -> ETensor -> Flow
rnnLast = genLastRnn "RNN"

lstmLast :: Bool -> Dim -> Dim -> ETensor -> Flow
lstmLast = genLastRnn "LSTM"

gruLast :: Bool -> Dim -> Dim -> ETensor -> Flow
gruLast = genLastRnn "GRU"

rnnBiLast :: Bool -> Dim -> Dim -> ETensor -> Flow
rnnBiLast = genBiLastRnn "RNN"

lstmBiLast :: Bool -> Dim -> Dim -> ETensor -> Flow
lstmBiLast = genBiLastRnn "LSTM"

gruBiLast :: Bool -> Dim -> Dim -> ETensor -> Flow
gruBiLast  = genBiLastRnn "GRU"


{- Models a single directional RNN, outputs at each time step -}
genRnn :: String -> Dim -> Dim -> ETensor -> Flow
genRnn name inputDim hidden input = 
   record (RNN name inputDim hidden False) $ input >>= (rnnChk name inputDim hidden)

{- Checks that the RNN is applied to the give time series -}
rnnChk :: String -> Dim -> Dim -> Tensor -> ETensor
rnnChk name inputFeats hidden input
   | hasN input 3 && matchF  = out ((init size) ++ [hidden])
   | hasN input 2 && matchF  = out [head size, hidden]
   | otherwise               = Left $ rnnMsg name (rnnFeats input) inputFeats

   where size       = dim input
         out d      = Right $ fromDim d
         matchF     = correctFeats inputFeats input 
         

{- Models a single directional RNN, that outputs the last time steps vector -}
genLastRnn :: String -> Bool -> Dim -> Dim -> ETensor -> Flow
genLastRnn name batchFst inputDim hidden input = 
   record (RNNLast name inputDim hidden False batchFst) $
      input >>= (lastRnnChk name batchFst (lit 1) inputDim hidden)


{- checks and applies a RNN that outputs the last time steps vector -}
lastRnnChk :: String -> Bool -> Dim -> Dim -> Dim -> Tensor -> ETensor
lastRnnChk name batchFst fstOut inputFeats hidden input
   | hasN input 3 && matchF  = out [fstOut, batch, hidden]
   | hasN input 2 && matchF  = out [fstOut, hidden]
   | otherwise               = Left $ rnnMsg name (rnnFeats input) inputFeats

   where dims       = dim input
         out d      = Right $ fromDim d
         batch      = if batchFst then head dims else head $ tail dims
         matchF     = correctFeats inputFeats input

{- Models a Bi-Directional RNN, outputs at each time step -}
genBiRnn :: String -> Dim -> Dim -> ETensor -> Flow
genBiRnn name inputFeats hidden input = 
   record (RNN name inputFeats bi True) $ input >>= (rnnChk name inputFeats bi)
      
      where bi = multiply (lit 2) hidden

{- Models a Bi-Directional RNN that outputs two "last" vectors -}
genBiLastRnn :: String -> Bool -> Dim -> Dim -> ETensor -> Flow
genBiLastRnn name batchFst inputFeats hidden input = 
   record (RNNLast name inputFeats hidden True batchFst) $
      input >>= (lastRnnChk name batchFst (lit 2) inputFeats hidden)

{- Determines if the tensor has the expected number of dimensions -}
hasN :: Tensor -> Int -> Bool
hasN tensor expected = expected == (length $ dim tensor)

{- Determines if the tensor (RNN input) has the correct number of features -}
correctFeats :: Dim -> Tensor -> Bool
correctFeats expected tensor = (rnnFeats tensor) == expected

rnnFeats :: Tensor -> Dim
rnnFeats = last . dim

{- Creates a RNN error message -}
rnnMsg :: String -> Dim -> Dim -> String
rnnMsg name input expected = printf msg name (show input) (show expected)

   where msg = unlines ["%s Layer: input must be 2 (length, H)",
                        "or (batch, length, h) or (length, batch, h)",
                        "Given features %s must match expected features %s"]
