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

module Metatorch.Tensor
(
   Tensor(..),
   ETensor,
   dim,
   fromDim,
   isScalar
)
where

import Metatorch.Dim (Dim)

data Tensor = Scalar
            | Vector Dim
            | Matrix Dim Dim 
            | Tensor Dim Dim Dim 
            | Tensor4D Dim Dim Dim Dim
            | NTensor [Dim]
            deriving (Show, Eq)


type ETensor = Either String Tensor


{- Returns the dimensions of the tensor -}
dim :: Tensor -> [Dim]
dim Scalar                 = []
dim (Vector d)             = [d]
dim (Matrix d1 d2)         = [d1, d2]
dim (Tensor d1 d2 d3)      = [d1, d2, d3]
dim (Tensor4D d1 d2 d3 d4) = [d1, d2, d3, d4]
dim (NTensor d)            = d

{- Makes a Tensor based on the given dimensions-}
fromDim :: [Dim] -> Tensor
fromDim []               = Scalar
fromDim (d:[])           = Vector d
fromDim (d1:d2:[])       = Matrix d1 d2
fromDim (d1:d2:d3:[])    = Tensor d1 d2 d3
fromDim (d1:d2:d3:d4:[]) = Tensor4D d1 d2 d3 d4
fromDim d                = NTensor d

{- Is true if the tensor is a scalar value -}
isScalar :: Tensor -> Bool
isScalar Scalar = True
isScalar _      = False
