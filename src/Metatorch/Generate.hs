module Metatorch.Generate (generate) where

import Control.Monad.Writer

import Text.Printf (printf)
import qualified Data.Map.Strict as M
import Data.Char (toLower)
import Data.List

import Metatorch.Layer (Layer(..), Flow)
import Metatorch.Dim (Dim, dimVars, hasVars, addPrefix)

type LayerNames = [(Layer, String)]
type LayerCount = M.Map LayerType Int
type LayerType = String
type Model = [Layer]

{- Generates Python3/Pytorch code based on a list of layers -}
generate :: Bool -> Flow -> String
generate spaces network = case model network of
   Left  err    -> err
   Right layers -> unlines code
   
      where names      = assignNames $ groupBlocks layers
            prefix     = if spaces then spaceIndent else tabIndent

            --all the layers that need to be declared
            declLayers = filter (not . isFunctional . fst) names

            --get all the unique variable names from all the layers
            attributes = nub . concatMap dimVars $ concatMap layerDims layers

            --constructor for the model
            construct  = generateInit prefix attributes declLayers

            --forward method
            forward    = generateForward prefix attributes names

            --training method
            train = generateTraining prefix

            --all the code
            code       = ["from random import shuffle",
                          "from time import time",
                          "import torch as t",
                          "import torch.nn as nn",
                          "from torch.optim import Adam",
                          "",
                          "class Model(nn.Module):",
                          "",
                          construct,
                          forward,
                          train]


{- Gets the model from the network flow -}
model :: Flow -> Either String Model
model flow = 
   let
      (output, layersWithDim)  = runWriter flow
      layers                   = map fst layersWithDim
   in
      case output of 
         Right _  -> Right layers
         Left msg -> Left $ "Could not generate code " ++ msg


{- Generates the constructor (init) for the model -}
generateInit :: String -> [String] -> LayerNames -> String
generateInit prefix vars namedLayers = 
   let 
      --check if there is a loss layer
      lossFun = find (isLoss . fst) namedLayers

      --declaration of the constructor
      initDecl     = case lossFun of 

                        -- there is a loss layer, no need for an arg
                        Just _ -> printf (prefix ++ "def __init__(self, %s):")

                        -- make the loss function an argument
                        _ -> printf (prefix ++ "def __init__(self, loss_function, %s):")

      --function to apply indenting
      indent s     = prefix ++ prefix ++ s

      --format the arguments to the constructor
      args         = intercalate ", " vars

      --call to super constructor
      super        = indent "super().__init__()\n"


      --declare all the attributes like hidden layer size
      attrDecls    = map (indent . genAttrDecl) vars

      --triple indentation for sequential layers
      triple       = prefix ++ prefix ++ prefix

      --declare all the layers
      layerDecls   = map (indent . (uncurry (genDecl triple))) namedLayers
   in
      unlines $ [initDecl args, super] ++ attrDecls ++ layerDecls


{- Creates an attribute per each dimension variable -}
genAttrDecl :: String -> String
genAttrDecl var = printf "self.%s = %s" var var


{- Generates the forward method for the model -}
generateForward :: String -> [String] -> LayerNames -> String
generateForward prefix vars namedLayers = 
   let
      indent s     = prefix ++ prefix ++ s
      steps        = filter (callable . fst) namedLayers
      calls        = unlines $ map (indent . (uncurry $ genCall vars)) steps
   in
      unlines $ [prefix ++ "def forward(self, tensor):",
                 indent "\"\"\"",
                 indent "Applies the model to the given tensor",
                 indent "\"\"\"",
                 indent . genSize . fst $ head namedLayers,
                 calls,
                 indent "return tensor"]


{- Generates calls of layers -}
genCall :: [String] -> Layer -> String -> String
genCall _ (RNN _ _ _ _) name             = printf "tensor, _ = self.%s(tensor)" name
genCall _ (RNNLast "LSTM" _ _ _ _) name  = printf "_, (tensor, _) = self.%s(tensor)" name
genCall _ (RNNLast _ _ _ _ _) name       = printf "_, tensor = self.%s(tensor)" name
genCall _ (Average index) _              = printf "tensor = t.mean(tensor, %d)" index
genCall _ (Max index) _                  = printf "tensor = t.max(tensor, %d)[0]" index
genCall _ (Permute dims) _               = 
   let
      --note this assumes a constant or single variable dimension
      fmtIndex ds = intercalate ", " (map show ds)
   in
      printf "tensor = tensor.permute(%s)" (fmtIndex dims)

genCall _ (Squeeze index) _              = printf "tensor = tensor.squeeze(%d)" index
genCall attrs (Reshape dims) _           = 
   let
      --determines if the dimensions has any class attribute variables
      hasAttrs d    = any (\ v -> v `elem` attrs) (dimVars d)
      fmtDim (d:[]) = show (fmtVar d) ++ ","
      fmtDim ds     = intercalate ", " $ map (show . fmtVar) ds
      fmtVar v      = if hasAttrs v then addPrefix "self." v else v
   in
      printf "tensor = t.reshape(tensor, (%s))" (fmtDim dims)

genCall _ _ name                         = printf "tensor = self.%s(tensor)" name


{- Generates a layer's declaration i.e. inside the class constructor -}
genDecl :: String -> Layer -> String -> String
genDecl prefix layer name = printf "self.%s = %s" name (declare prefix layer)
  
{- Generates the Pytorch object instantiation -}
declare :: String -> Layer -> String
declare _ (RNN name inFeat outFeat bi) = 
   let
      py = "nn.%s(%s, %s, bidirectional=%s)"
   in
      printf py name (show inFeat) (show outFeat) (show bi)

declare _ (RNNLast name inFeat outFeat bi batch) = 
   let
      py = "nn.%s(%s, %s, bidirectional=%s, batch_first=%s)"
   in
      printf py name (show inFeat) (show outFeat) (show bi) (show batch)

declare _ (Conv1d inF outF window stride pad) = 
   let
      py = "nn.Conv1d(%s, %s, %d, %d, %d)" 
   in
      printf py (show inF) (show outF) window stride pad

declare _ (Pool1d name window stride pad) = 
   printf "nn.%s1d(%d, %d, %d)" name window stride pad

declare _ (Conv2d inF outF window stride pad) = 
   let
      py = "nn.Conv2d(%s, %s, %d, %d, %d)"
   in
      printf py (show inF) (show outF) window stride pad

declare _ (Pool2d name window stride pad) = 
   printf "nn.%s2d(%d, %d, %d)" name window stride pad

declare _ (Embedding vocab emb) = 
   printf "nn.Embedding(%s, %s)" (show vocab) (show emb)

declare _ (Activation name) = printf "nn.%s()" name
declare _ (Linear inF outF) = 
   printf "nn.Linear(%s, %s)" (show inF) (show outF)

declare _ (CELoss _) = "nn.CrossEntropyLoss()"

declare prefix (Sequential layers) = 
   let
      names = map (declare prefix) layers
      args  = intercalate (",\n" ++ prefix) names
   in
      printf "nn.Sequential(%s)" args

declare _ _ = ""

{- Generates the line that unpacks the size of the given tensor -}
genSize :: Layer -> String
genSize (Input dims) = 
   let
      genInput di
         | hasVars di   = show di   --use the variable name
         | otherwise    = "_"       --constants are replaced with _

      needsSize = any hasVars dims
      assignment = intercalate ", " (map genInput dims)
   in
      if needsSize then
         printf "%s = tensor.size()" assignment
      else
         ""

genSize _ = ""

{- Generates the model's training method -}
generateTraining :: String -> String
generateTraining prefix =
   let
      --indents the line
      tab n s = concat (replicate n prefix) ++ s
   in
      unlines $ map (tab 1) 
               ["def fit(self, train_data, dev_data, num_epochs, learning_rate, reg):",
                tab 1 "\"\"\"",
                tab 1 "Trains the model, the data collections are iterables of (inst, target) tuples",
                tab 1 "\"\"\"",
                tab 1 "params = [p for p in self.parameters() if p.requires_grad]",
                tab 1 "optim = Adam(params, learning_rate, weight_decay=reg)",
                tab 1 "self.train()",
                tab 1 "epoch = 1",
                tab 1 "",
                tab 1 "# for a fixed number of epochs, train the model",
                tab 1 "while epoch < num_epochs:",
                tab 2 "print(\"Epoch %d |\" % epoch, end=\"\")",
                tab 2 "start_time = time()",
                tab 2 "total_loss = 0.0",
                tab 2 "",
                tab 2 "shuffle(train_data)",
                tab 2 "",
                tab 2 "# for each training example, make a prediction, measure the loss, and update",
                tab 2 "for inst, target in train_data:",
                tab 3 "self.zero_grad()",
                tab 3 "pred = self(inst.cuda())",
                tab 3 "loss = self.loss_function(pred, target.cuda())",
                tab 3 "total_loss += loss.cpu().data.item()",
                tab 3 "loss.backward()",
                tab 3 "optim.step()",
                tab 3 "",
                tab 2 "print(\" Loss: %7.2f |\" % (total_loss / len(train_data)), end=\"\") ",
                tab 2 "print(\" Time: %8.2f\" % (time() - start_time), \"seconds\")",
                tab 2 "epoch += 1",
                tab 2 "",
                tab 1 "self.eval()"
               ]

{- assigns attribute names to each layer -}
assignNames :: Model -> LayerNames
assignNames network = reverse $ assignHelper M.empty network []

assignHelper :: LayerCount -> Model -> LayerNames -> LayerNames
assignHelper _ [] names                      = names
assignHelper oldCounts (layer:rest) oldNames = 
  let 
      --increment the counts
      counts = M.insertWith (+) (layerType layer) 1 oldCounts         

      --update the names
      names  = (layer, makeName counts layer) : oldNames
  in
     assignHelper counts rest names

{- Makes a name for the layer based on how many other layers exist previously -}
makeName :: LayerCount -> Layer -> String
makeName counts layer = 
   let 
      --lookup the count of the layer type, should never be "Nothing"
      cnt  = case M.lookup (layerType layer) counts of
                 Just count -> count
                 _          -> 0

      name = if isLoss layer then
               "loss_function"
            else
               lower $ layerType layer
   in
      if cnt > 1 then (printf "%s%d" name cnt) else name

{- Produces the layer's type, i.e. a name of the layer and its type -}
layerType :: Layer -> LayerType
layerType (Linear _ _)            = "Linear"
layerType (RNN name _ _ _)        = name
layerType (RNNLast name _ _ _ _)  = name
layerType (Conv1d _ _ _ _ _)      = "Conv1d"
layerType (Pool1d name _ _ _)     = printf "%s1d" name
layerType (Conv2d _ _ _ _ _)      = "Conv2d"
layerType (Pool2d name _ _ _)     = printf "%s2d" name
layerType (Average _)             = "mean"
layerType (Max _)                 = "max"
layerType (Permute _)             = "permute"
layerType (Squeeze _)             = "squeeze"
layerType (Reshape _)             = "reshape"
layerType (Activation name)       = name
layerType (CELoss _)              = "CrossEntropyLoss"
layerType (Input _)               = "Input"
layerType (Broken _)              = "Broken"
layerType (Sequential _)          = "Sequential"
layerType (Embedding _ _)         = "Embedding"


{- Groups up the layers into Sequential blocks -}
groupBlocks :: Model -> Model
groupBlocks = 
   let

      isReg layer = not (isFunctional layer) && not (isLoss layer) && not (isRNN layer)
      bothFunctional l k = (isReg l) == (isReg k)
   in
      --group up layers into functional and not functional chunks
      --make non-functional chunks into a sequential layers
      concatMap makeSequential . groupBy bothFunctional


{- Makes a seqential layer, assume all the layers in the model block
are functional or not -}
makeSequential :: Model -> Model
makeSequential []                       = []
makeSequential [single]                 = [single]
makeSequential (first:rest)
   | isFunctional first || isRNN first  = (first:rest)
   | otherwise                          = [Sequential (first:rest)]

{- Determines if the layer is functional or not -}
isFunctional :: Layer -> Bool
isFunctional (Average _) = True
isFunctional (Max _)     = True
isFunctional (Permute _) = True
isFunctional (Squeeze _) = True
isFunctional (Reshape _) = True
isFunctional (Input _)   = True
isFunctional (Broken _)  = True
isFunctional _           = False

{- Determines if the layer is an RNN -}
isRNN (RNN _ _ _ _)       = True
isRNN (RNNLast _ _ _ _ _) = True
isRNN _                   = False

{- Determines if the layer is a loss function or not -}
callable :: Layer -> Bool
callable (Input _)  = False
callable (Broken _) = False
callable (CELoss _) = False
callable _          = True

{- Determines if the layer is a loss function -}
isLoss :: Layer -> Bool
isLoss (CELoss _) = True
isLoss _          = False

{- Produces the dimensions use by the layer -}
layerDims :: Layer -> [Dim]
layerDims (Linear inDim outDim)                 = [inDim, outDim]
layerDims (RNN _ inDim outDim _)                = [inDim, outDim]
layerDims (RNNLast _ inDim outDim _ _)          = [inDim, outDim]
layerDims (Conv1d inDim outDim _ _ _)           = [inDim, outDim]
layerDims (Conv2d inDim outDim _ _ _)           = [inDim, outDim]
layerDims (Reshape dims)                        = dims
layerDims (CELoss dims)                         = [dims]
layerDims (Input dims)                          = dims
layerDims (Embedding vocab dims)                = [vocab, dims]
layerDims _                                     = []

lower :: String -> String
lower = map toLower

spaceIndent :: String
spaceIndent = "    "

tabIndent :: String
tabIndent   = "\t"
