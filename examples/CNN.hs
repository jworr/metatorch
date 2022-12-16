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

import Metatorch

--define variables and constants
l = var "l"
w = var "w"
k = var "k"
h = var "h"
n = var "n"
_4  = lit 4


--Conv2d applied to a batch of sequences and makes a prediction
--for each sequence in the batch
conv :: Flow
conv = input [n, k, l, w]
     >>= conv2d 5 1 2 k h
     >>= maxPool2d 2 2 1 h
     >>= conv2d 3 1 1 h h
     >>= maxPool2d 2 2 1 h
     >>= mean 3
     >>= mean 2
     >>= linear h _4
     >>= crossEnt _4 (Vector n)


main :: IO ()
main = evalModel conv
