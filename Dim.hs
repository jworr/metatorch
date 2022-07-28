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

module Dim
(
   Dim,
   lit,
   var,
   multiply,
   multiplyAll,
   sub,
   add,
   divBy,
   hasVars,
   dimVars,
   addPrefix
)
where

import Data.List (intercalate, sort)
import Text.Printf (printf)

--TODO/NOTE: addition and subtraction is NOT commutative!!!!

--numerator, denominator, sum of expressions
data Dim = Dim Int Int [Sum]
         deriving Ord

data Sum = Sum Dim Dim
         | Diff Dim Dim
         | Var String
         deriving (Ord, Eq)

instance Show Sum where

   show (Var str)         = str
   show (Sum left right)  = printf "(%s + %s)" (show left) (show right)
   show (Diff left right) = printf "(%s - %s)" (show left) (show right)
      

instance Show Dim where

   show (Dim 1 1 [])          = show 1
   show (Dim num 1 [])        = show num
   show (Dim num denom [])    = printf "(%d/%d)" num denom
   show (Dim 1 1 [v])         = show v
   show (Dim num 1 [v])       = printf "%d*%s" num (show v)
   show (Dim num denom [v])   = printf "(%d/%d)*%s" num denom (show v)
   show (Dim 1 1 vars)        = unwords $ map show vars
   show (Dim num 1 vars)      = printf "%d*(%s)" num (showVars vars)
   show (Dim num denom vars)  = printf "(%d/%d)*(%s)" num denom (showVars vars)

instance Eq Dim where

   (Dim ln ld lds) == (Dim rn rd rds) =  (ln == rn) 
                                      && (ld == rd)
                                      && (sort lds) == (sort rds)

showVars :: [Sum] -> String
showVars vars = intercalate "*" $ map show vars

{- Adds a prefix to all the variable names -}
addPrefix :: String -> Dim -> Dim
addPrefix prefix (Dim num denom vars) = Dim num denom $ map (addSumPrefix prefix) vars

addSumPrefix :: String -> Sum -> Sum
addSumPrefix prefix (Var var)            = Var (prefix ++ var)
addSumPrefix prefix (Sum left right)     = Sum (addPrefix prefix left) (addPrefix prefix right)
addSumPrefix prefix (Diff left right)    = Diff (addPrefix prefix left) (addPrefix prefix right)

{- Returns all the individual variables used in the dimension -}
dimVars :: Dim -> [String]
dimVars (Dim _ _ sums) = concatMap sumVars sums

sumVars :: Sum -> [String]
sumVars (Var var)         = [var]
sumVars (Sum left right)  = (dimVars left) ++ (dimVars right)
sumVars (Diff left right) = (dimVars left) ++ (dimVars right)

{- Determines if the dimension contains variables -}
hasVars :: Dim -> Bool
hasVars = not . null . getVars

getVars :: Dim -> [Sum]
getVars (Dim _ _ vars) = vars

getCoef :: Dim -> [Int]
getCoef (Dim num denom _) = [num, denom]

{- Makes a dimension from an Int -}
lit :: Int -> Dim
lit x = Dim x 1 []

{- Makes a dimension from a String -}
var :: String -> Dim
var v = Dim 1 1 [Var v]

{- Multiples two dimensions together -}
multiply :: Dim -> Dim -> Dim
multiply (Dim x xd vars) (Dim y yd others) = Dim (x * y) (xd * yd) (vars ++ others)

{- Multiply all the dimension together -}
multiplyAll :: [Dim] -> Dim
multiplyAll = foldr multiply (lit 1)

{- Subtract two dimensions -}
sub :: Dim -> Dim -> Dim
sub (Dim 1 1 [Diff x (Dim c 1 [])]) (Dim k 1 []) = Dim 1 1 [Diff x (lit $ c + k)]

sub (Dim 1 1 [Sum x (Dim c 1 [])]) (Dim k 1 [])
   | c - k == 0 = x
   | c > k      = Dim 1 1 [Sum x (lit $ c - k)]
   | otherwise  = Dim 1 1 [Diff x (lit $ k - c)]

sub left right
   | lVars == rVars && (denom left) == (denom right) = Dim ((num left) - (num right)) (denom left) lVars 
   | otherwise                                       = Dim 1 1 [Diff left right]

   where num = head . getCoef
         denom = head . tail. getCoef
         lVars = getVars left
         rVars = getVars right


{- Adds two dimensions -}
add :: Dim -> Dim -> Dim
add (Dim 1 1 [Sum x (Dim c 1 [])]) (Dim k 1 []) = Dim 1 1 [Sum x (lit $ c + k)]

add (Dim 1 1 [Diff x (Dim c 1 [])]) (Dim k 1 [])
   | k - c == 0 = x
   | c > k      = Dim 1 1 [Diff x (lit $ c - k)]
   | otherwise  = Dim 1 1 [Sum x (lit $ k - c)]

add left right
   | lVars == rVars && (denom left) == (denom right) = Dim ((num left) + (num right)) (denom left) lVars 
   | otherwise                                       = Dim 1 1 [Sum left right]

   where num = head . getCoef
         denom = head . tail. getCoef
         lVars = getVars left
         rVars = getVars right

{- divide a dimension by a constant factor -}
divBy :: Dim -> Int -> Dim
divBy (Dim value 1 []) divisor
   | value `mod` divisor == 0 = Dim (value `div` divisor) 1 []
   | otherwise                = Dim value divisor []
divBy (Dim num denom vars) divisor = Dim num (denom * divisor) vars
