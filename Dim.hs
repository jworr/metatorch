module Dim
(
   Dim,
   lit,
   var,
   multiply,
   multiplyAll
)
where

import Data.Char (isDigit)
import Text.Printf (printf)
import Data.List (foldr, sort)

data Dim = Dim Int [String]

instance Show Dim where

   show (Dim num [])   = show num
   show (Dim 1 [v])    = printf "%s" v
   show (Dim num [v])  = printf "%d%s" num v
   show (Dim 1 vars)   = printf "%s" (unwords vars)
   show (Dim num vars) = printf "%d(%s)" num (unwords vars)

instance Eq Dim where

   (Dim l lds) == (Dim r rds) = (l == r) && (sort lds) == (sort rds)

{- Makes a dimension from an Int -}
lit :: Int -> Dim
lit x = Dim x []

{- Makes a dimension from a String -}
var :: String -> Dim
var v = Dim 1 [v]

{- Multiplys two dimensions together -}
multiply :: Dim -> Dim -> Dim
multiply (Dim x vars) (Dim y others) = Dim (x * y) (vars ++ others)

{- Multiply all the dimension together -}
multiplyAll :: [Dim] -> Dim
multiplyAll = foldr multiply (lit 1)
