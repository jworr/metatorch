module Dim
(
)
where

import Data.Char (isDigit)
import Text.Printf (printf)

data Dim = Dim Int [String]

instance Show Dim where

   show (Dim num [])   = show num
   show (Dim num [v])  = printf "%d%s" num v
   show (Dim num vars) = printf "%d(%s)" num (unwords vars)

lit :: Int -> Dim
lit x = Dim x []

var :: String -> Dim
var v = Dim 1 [v]

multiply :: Dim -> Dim -> Dim
multiply (Dim x vars) (Dim y others) = Dim (x * y) (vars ++ others)

