
import Tensor (Tensor(..))
import Layer (Flow, start, act, linear, crossEnt, generate, permute, reshape)
import Layer.Rnn (lstm, gruBi, lstmBiLast)
import Layer.Cnn (conv1d, maxPool1d)
import Dim (lit, var, multiply, add, sub, divBy)

--define variables and constants
l = var "l"
k = var "k"
h = var "h"
d = var "d"
_2h = lit 2 `multiply` h
n = var "n"
h_2 = h `sub` _2
l' = l `sub` _2
l'' = ((l' `sub` _2) `divBy` 2) `add` (lit 1)

_2  = lit 2
_4  = lit 4
_5  = lit 5
_10 = lit 10

--per "word" classifier for 10 classes
token :: Flow
token = (start $ Matrix l k)
      >>= lstm h
      >>= linear h _10
      >>= crossEnt _10 (Vector l)

--sequence summarization, bi-directional LSTM, last vectors used for prediction
--one prediction per sequence, 2 classes
summarize :: Flow
summarize = (start $ Tensor n l k)
          >>= lstmBiLast True h
          >>= reshape [n, _2h]
          >>= linear _2h _2
          >>= crossEnt _2 (Vector n)

--batched per "word" classifier for 5 classes
batchToken :: Flow
batchToken = (start $ Tensor n l k)
           >>= gruBi h
           >>= linear _2h _5
           >>= permute [0, 2, 1]
           >>= crossEnt _5 (Matrix n l)

--Conv1d applied to a sequence
conv = (start $ Tensor n k l)
     >>= conv1d k h 3 1
     >>= maxPool1d h 2 2
     >>= permute [0, 2, 1]
     >>= linear h _4
     >>= permute [0, 2, 1]
     >>= crossEnt _4 (Matrix n l'')

--multi-layer perceptron for 4 classes example 
mlp :: Flow
mlp = (start $ Matrix n k) 
      >>= linear k d
      >>= act "ReLu"
      >>= linear d _4
      >>= crossEnt _4 (Vector n)


main :: IO ()
main = do
   
   putStrLn "---MLP---"
   putStrLn $ generate mlp

   putStrLn "---Per Token classifier---"
   putStrLn $ generate token
   
   putStrLn "---Sequence Summarization---"
   putStrLn $ generate summarize

   putStrLn "---Per token classifier, batched---"
   putStrLn $ generate batchToken

   putStrLn "---Sequence + Conv for predicting 4 classes---"
   putStrLn $ generate conv

   return ()
