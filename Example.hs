
import Tensor (Tensor(..))
import Layer (Flow, start, act, linear, crossEnt, generate, permute, squeeze)
import Layer.Rnn (rnn, biRnn, lastRnn)

_2h = "test"


--per "word" classifier for 10 classes
token :: Flow
token = (start $ Matrix "l" "k")
      >>= rnn "LSTM" "h"
      >>= linear "h" "10"
      >>= crossEnt "10" (Vector "l")

--sequence summarization, one prediction per sequence, 4 classes
summarize :: Flow
summarize = (start $ Tensor "n" "l" "k")
          >>= lastRnn "LSTM" True "h"
          >>= squeeze 0
          >>= linear "h" "2"
          >>= crossEnt "2" (Vector "n")

--batched per "word" classifier for 5 classes
batchToken :: Flow
batchToken = (start $ Tensor "n" "l" "k")
           >>= biRnn "GRU" "h"
           >>= linear "2*h" "5"
           >>= permute [0, 2, 1]
           >>= crossEnt "5" (Matrix "n" "l")

--multi-layer perceptron for 4 classes example 
mlp :: Flow
mlp = (start $ Matrix "n" "k") 
      >>= linear "k" "d"
      >>= act "ReLu"
      >>= linear "d" "4"
      >>= crossEnt "4" (Vector "n")


main :: IO ()
main = do
   
   putStrLn "---MLP---"
   putStrLn $ generate mlp

   putStrLn "---Per Token classifier---"
   putStrLn $ generate token
   
   putStrLn "---Sequence Summarization---"
   putStrLn $ generate summarize

   putStrLn "---Batched per token classifier---"
   putStrLn $ generate batchToken

   return ()
