module Dim
(
   Dim,
   lit,
   var,
   multiply,
   multiplyAll,
   sub
)
where

import Text.Printf (printf)
import Data.List (sort)

data Dim = Dim Int [Diff]
         deriving Ord

data Diff = Diff Dim Dim
          | Var String
          deriving (Ord, Eq)

instance Show Diff where

   show (Var str)       = str
   show (Diff dim dim2) = printf "(%s - %s)" (fmt left) (fmt right)
      
      where left    = show dim
            right   = show dim2
            fmt str
               | null str         = ""
               | length str == 1  = str
               | otherwise        = printf "(%s)" str


instance Show Dim where

   show (Dim 1 [])     = show 1
   show (Dim num [])   = show num
   show (Dim 1 [v])    = printf "%s" (show v)
   show (Dim num [v])  = printf "%d%s" num (show v)
   show (Dim 1 vars)   = printf "%s" (unwords $ map show vars)
   show (Dim num vars) = printf "%d(%s)" num (unwords $ map show vars)

instance Eq Dim where

   (Dim l lds) == (Dim r rds) = (l == r) && (sort lds) == (sort rds)

getVars :: Dim -> [Diff]
getVars (Dim _ vars) = vars

getCoef :: Dim -> Int
getCoef (Dim coef _) = coef

{- Makes a dimension from an Int -}
lit :: Int -> Dim
lit x = Dim x []

{- Makes a dimension from a String -}
var :: String -> Dim
var v = Dim 1 [Var v]

{- Multiplys two dimensions together -}
multiply :: Dim -> Dim -> Dim
multiply (Dim x vars) (Dim y others) = Dim (x * y) (vars ++ others)

{- Multiply all the dimension together -}
multiplyAll :: [Dim] -> Dim
multiplyAll = foldr multiply (lit 1)

{- Subtract two dimensions -}
sub :: Dim -> Dim -> Dim
sub (Dim left []) (Dim right [])        = Dim (left - right) []
sub left right
   | (getVars left) == (getVars right)  = Dim ((getCoef left) - (getCoef right)) (getVars left)
   | otherwise                          = Dim 1 $ [Diff left right]
