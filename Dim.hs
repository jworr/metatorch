module Dim
(
   Dim,
   lit,
   var,
   multiply
)
where

import Data.Char (isDigit)
import Text.Printf (printf)

--TODO the Eq instance should be manually implemented to make sure multiplication
--commutative
data Dim = Dim Int [String]
           deriving Eq

instance Show Dim where

   show (Dim num [])   = show num
   show (Dim 1 [v])    = printf "%s" v
   show (Dim num [v])  = printf "%d%s" num v
   show (Dim 1 vars)   = printf "%s" (unwords vars)
   show (Dim num vars) = printf "%d(%s)" num (unwords vars)

{- Makes a dimension from an Int -}
lit :: Int -> Dim
lit x = Dim x []

{- Makes a dimension from a String -}
var :: String -> Dim
var v = Dim 1 [v]

{- Multiplys two dimensions together -}
multiply :: Dim -> Dim -> Dim
multiply (Dim x vars) (Dim y others) = Dim (x * y) (vars ++ others)

