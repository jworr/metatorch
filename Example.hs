
import Tensor (Tensor(..))
import Layer (Flow, start, act, linear, crossEnt, generate, permute, squeeze)
import Layer.Rnn (lstm, gruBi, lstmLast)
import Dim (lit, var, multiply)

--define variables and constants
l = var "l"
k = var "k"
h = var "h"
d = var "d"
_2h = lit 2 `multiply` h
n = var "n"

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

--sequence summarization, one prediction per sequence, 4 classes
summarize :: Flow
summarize = (start $ Tensor n l n)
          >>= lstmLast True h
          >>= squeeze 0
          >>= linear h _2
          >>= crossEnt _2 (Vector n)

--batched per "word" classifier for 5 classes
batchToken :: Flow
batchToken = (start $ Tensor n l k)
           >>= gruBi h
           >>= linear _2h _5
           >>= permute [0, 2, 1]
           >>= crossEnt _5 (Matrix n l)

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

   return ()
