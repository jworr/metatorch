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

import Metatorch (Tensor(..), Flow, input, relu, linear, crossEnt, 
   check, permute, reshape, mean, lstm, gruBi, lstmBiLast, conv2d, maxPool2d, lit, var)
import Metatorch.Dim (multiply, sub)
import Metatorch.Generate (generate)

--define variables and constants
l = var "l"
w = var "w"
k = var "k"
h = var "h"
d = var "d"
_2h = lit 2 `multiply` h
n = var "n"
h_2 = h `sub` _2

_2  = lit 2
_4  = lit 4
_5  = lit 5
_10 = lit 10

--per "word" classifier for 10 classes
token :: Flow
token = input [l, k]
      >>= lstm k h
      >>= linear h _10
      >>= crossEnt _10 (Vector l)

--sequence summarization, bi-directional LSTM, last vectors used for prediction
--one prediction per sequence, 2 classes
summarize :: Flow
summarize = input [n, l, k]
          >>= lstmBiLast True k h
          >>= reshape [n, _2h]
          >>= linear _2h _2
          >>= crossEnt _2 (Vector n)

--batched per "word" classifier for 5 classes
batchToken :: Flow
batchToken = input [n, l, k]
           >>= gruBi k h
           >>= linear _2h _5
           >>= permute [0, 2, 1]
           >>= crossEnt _5 (Matrix n l)

--Conv1d applied to a batch of sequences and makes a prediction
--for each sequence in the batch
conv :: Flow
conv = input [n, k, l, w]
     >>= conv2d 5 1 2 k h
     >>= maxPool2d 2 2 1 h
     >>= conv2d 3 1 1 h h
     >>= maxPool2d 2 2 1 h
     >>= mean 3
     >>= mean 2
     >>= linear h _4
     >>= crossEnt _4 (Vector n)

--multi-layer perceptron for 4 classes example 
mlp :: Flow
mlp = input [n, k]
      >>= linear k d
      >>= relu
      >>= linear d _4
      >>= crossEnt _4 (Vector n)


main :: IO ()
main = do
   
   putStrLn "---MLP---"
   putStrLn $ check mlp

   putStrLn "---Per Token classifier---"
   putStrLn $ check token
   
   putStrLn "---Sequence Summarization---"
   putStrLn $ check summarize

   putStrLn "---Per token classifier, batched---"
   putStrLn $ check batchToken

   putStrLn "---Sequence + Conv for predicting 4 classes---"
   putStrLn $ check conv

   putStrLn "--Code--"
   putStrLn $ generate False summarize

   return ()
