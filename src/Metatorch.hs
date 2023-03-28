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

module Metatorch
(
   --from Metatorch.Dim
   Dim(..), lit, var, multiply, sub, add,

   --from Metatorch.Layer
   check, Flow, input, relu, linear, crossEnt, mseLoss, permute, reshape, mean, 
   squeeze, unsqueeze, maximize, sigmoid, tanhAct,

   --from Metatorch.Layer.CNN
   conv1d, maxPool1d, avgPool1d, conv2d, maxPool2d, avgPool2d,

   --from Metatorch.Layer.RNN
   rnn, lstm, gru, rnnBi, lstmBi, gruBi, rnnLast, lstmLast, gruLast, rnnBiLast,
   lstmBiLast, gruBiLast,

   --from Metatorch.Layer.Embedding
   embedding,

   --from Metatorch.Tensor
   Tensor(..),

   --from Metatorch
   evalModel
)
where

import System.Environment (getArgs)

import Metatorch.Generate (generate)
import Metatorch.Layer (check, Flow, input, relu, linear, crossEnt, permute, 
   reshape, mean, squeeze, unsqueeze, maximize, mseLoss, sigmoid, tanhAct)
import Metatorch.Layer.CNN (conv1d, maxPool1d, avgPool1d, conv2d, maxPool2d, 
   avgPool2d)
import Metatorch.Layer.RNN (rnn, lstm, gru, rnnBi, lstmBi, gruBi, rnnLast, 
   lstmLast, gruLast, rnnBiLast, lstmBiLast, gruBiLast)
import Metatorch.Layer.Embedding (embedding)

import Metatorch.Tensor (Tensor(..))
import Metatorch.Dim (Dim(..), lit, var, multiply, sub, add)

--constants for CLI
usage = "[gen [tabs]]"

{- Runs and evalutes the model based on command-line arguments -}
evalModel :: Flow -> IO ()
evalModel flow = getArgs >>= putStrLn . evalFlow flow

evalFlow :: Flow -> [String] -> String
evalFlow model []              = check model
evalFlow model ["gen"]         = generate True model
evalFlow model ["gen", "tabs"] = generate False model
evalFlow _ _                   = usage
