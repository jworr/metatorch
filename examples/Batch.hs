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
k = var "k"
h = var "h"
_2h = lit 2 `multiply` h
n = var "n"

_5  = lit 5


--batched per "word" classifier for 5 classes
batchToken :: Flow
batchToken = input [n, l, k]
           >>= gruBi k h
           >>= linear _2h _5
           >>= permute [0, 2, 1]
           >>= crossEnt _5 (Matrix n l)


main :: IO ()
main = evalModel batchToken 
