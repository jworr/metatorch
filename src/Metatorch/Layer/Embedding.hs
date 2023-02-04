{-
   Copyright 2023 J. Walker Orr

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

module Metatorch.Layer.Embedding (embedding) where

import Metatorch.Tensor (Tensor, ETensor, dim, fromDim)
import Metatorch.Layer(Layer(..), Flow, record)
import Metatorch.Dim (Dim)

{- Models an embedding, produces a vector per input -}
embedding :: Dim -> Dim -> ETensor -> Flow
embedding vocab embSize input =
   record (Embedding vocab embSize) $ input >>= (embeddingChk embSize)


{- check that the embeding layer matches its input -}
embeddingChk :: Dim -> Tensor -> ETensor
embeddingChk embSize tensor = Right . fromDim $ dim tensor ++ [embSize]
