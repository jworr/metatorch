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

module Layer.CNN
(
   conv1d,
   maxPool1d,
   avgPool1d,
   conv2d,
   maxPool2d,
   avgPool2d,
)
where

import Text.Printf (printf)

import Tensor (Tensor, ETensor, dim, fromDim)
import Layer(Layer(..), Flow, record)
import Dim (Dim, add, sub, divBy, lit)

{- Applies a 1D Convolutional layer -}
conv1d :: Dim -> Dim -> Int -> Int -> ETensor -> Flow
conv1d inChl outChl window stride input 
   = record (Conv1d outChl window stride) $
      input >>= (conv1dChk "Conv" inChl outChl window stride)

{-- Applies max pooling --}
maxPool1d :: Dim -> Int -> Int -> ETensor -> Flow
maxPool1d = pooling1d "Max"

{-- Applies max pooling --}
avgPool1d :: Dim -> Int -> Int -> ETensor -> Flow
avgPool1d = pooling1d "Avg"

{- Applies a 1D Pooling Layer -}
pooling1d :: String -> Dim -> Int -> Int -> ETensor -> Flow
pooling1d name channels window stride input
   = record (Pool1d name window stride) $
      input >>= (conv1dChk name channels channels window stride)

{- Checks that CNN is applied to the time series -}
conv1dChk :: String -> Dim -> Dim -> Int -> Int -> Tensor -> ETensor
conv1dChk name inChl outChl window stride tensor
   | has 2 && inChl == (head size)        = Right $ newDim size
   | has 3 && inChl == (head $ tail size) = Right $ newDim size
   | otherwise                            = Left  $ printf d1Err name (show inChl)

   where has   = hasDims tensor
         size  = dim tensor
         newDim [_, len]    = fromDim [outChl, outputLen len window stride]
         newDim [n, _, len] = fromDim [n, outChl, outputLen len window stride]
         newDim d           = fromDim d

d1Err :: String
d1Err = unlines ["%s1d: input channels %s do not match either have:",
   "shape 3 (batch, channel in, length) or shape 2 (channel in, length)"]



{-- Applies max pooling --}
maxPool2d :: Dim -> Int -> Int -> ETensor -> Flow
maxPool2d = pooling2d "Max"

{-- Applies max pooling --}
avgPool2d :: Dim -> Int -> Int -> ETensor -> Flow
avgPool2d = pooling2d "Avg"

{- Applies a 2D Pooling Layer -}
pooling2d :: String -> Dim -> Int -> Int -> ETensor -> Flow
pooling2d name channels window stride input
   = record (Pool2d name window stride) $
      input >>= (conv2dChk name channels channels window stride)

{- Applies a 2D CNN to a tensor-}
conv2d :: Dim -> Dim -> Int -> Int -> ETensor -> Flow
conv2d inChl outChl window stride input
   = record (Conv2d outChl window stride) $
      input >>= (conv2dChk "Conv" inChl outChl window stride)

{- Checks that CNN is applied to the time series -}
conv2dChk :: String -> Dim -> Dim -> Int -> Int -> Tensor -> ETensor
conv2dChk name inChl outChl window stride tensor
   | has 3 && inChl == (head size)        = Right $ newDim size
   | has 4 && inChl == (head $ tail size) = Right $ newDim size
   | otherwise                            = Left  $ printf d2Err name (show inChl)

   where has   = hasDims tensor
         size  = dim tensor
         newDim [_, len, width]    = fromDim [outChl, out len, out width]
         newDim [n, _, len, width] = fromDim [n, outChl, out len, out width]
         newDim d                  = fromDim d
         out l = outputLen l window stride

d2Err :: String
d2Err = unlines ["%s2d: input channels %s do not match either have:",
   "shape 3 (batch, channel in, length, width) or shape 2 (channel in, length, width)"]

{- Computes the output length -}
outputLen :: Dim -> Int -> Int -> Dim
outputLen len window stride = (numerator `divBy` stride) `add` (lit 1)

   --Length - (window - 1) - 1 from the docs, simplified because of no
   --padding or dilation
   where numerator = len `sub` (lit window)

{- Returns true if the tensor has a specified number of dimensions -}
hasDims :: Tensor -> Int -> Bool
hasDims tensor target = (length $ dim tensor) == target
