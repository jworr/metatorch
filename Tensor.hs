module Tensor
(
   Tensor(..),
   ETensor,
   dim,
   fromDim,
   isScalar
)
where

import Dim (Dim)

data Tensor = Scalar
            | Vector Dim
            | Matrix Dim Dim 
            | Tensor Dim Dim Dim 
            | NTensor [Dim]
            deriving (Show, Eq)


type ETensor = Either String Tensor


{- Returns the dimensions of the tensor -}
dim :: Tensor -> [Dim]
dim Scalar            = []
dim (Vector d)        = [d]
dim (Matrix d1 d2)    = [d1, d2]
dim (Tensor d1 d2 d3) = [d1, d2, d3]
dim (NTensor d)        = d

{- Makes a Tensor based on the given dimensions-}
fromDim :: [Dim] -> Tensor
fromDim []            = Scalar
fromDim (d:[])        = Vector d
fromDim (d1:d2:[])    = Matrix d1 d2
fromDim (d1:d2:d3:[]) = Tensor d1 d2 d3
fromDim d             = NTensor d

{- Is true if the tensor is a scalar value -}
isScalar :: Tensor -> Bool
isScalar Scalar = True
isScalar _      = False
