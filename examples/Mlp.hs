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
k = var "k"
d = var "d"
n = var "n"

_4  = lit 4


--multi-layer perceptron for 4 classes example 
mlp :: Flow
mlp = input [n, k]
      >>= linear k d
      >>= relu
      >>= linear d _4
      >>= crossEnt _4 (Vector n)


main :: IO ()
main = evalModel mlp
