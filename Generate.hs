module Generate(
   generate
)
where

import Control.Monad.Writer

import Text.Printf (printf)
import qualified Data.Map.Strict as M
import Data.Char (toLower)
import Data.List

import Layer (Layer(..), Flow)
import Dim (Dim, dimVars, hasVars, addPrefix)

type LayerNames = [(Layer, String)]
type LayerCount = M.Map LayerType Int
type LayerType = String
type Model = [Layer]

{- Generates Python3/Pytorch code based on a list of layers -}
generate :: Bool -> Flow -> String
generate spaces network = case model network of
   Left  err    -> err
   Right layers -> unlines $ preamble ++ [construct] ++ [forward]
   
      where names      = assignNames layers
            prefix     = if spaces then spaceIndent else tabIndent
            declLayers = filter (not . isFunctional . fst) names
            attributes = nub $ concatMap dimVars $ concatMap layerDims $ map fst declLayers
            construct  = generateInit prefix attributes declLayers
            forward    = generateForward prefix attributes names
            preamble   = ["import torch as t",
                          "import torch.nn as nn",
                          "",
                          "class Model(nn.Module):",
                          ""]

{- Gets the model from the network flow -}
model :: Flow -> Either String Model
model flow = case output of 
   Right _  -> Right layers
   Left msg -> Left $ "Could not generate code " ++ msg

   where (output, layersWithDim)  = runWriter flow
         layers                   = map fst layersWithDim


{- Generates the constructor (init) for the model -}
generateInit :: String -> [String] -> LayerNames -> String
generateInit prefix vars namedLayers = 
   unlines $ [initDecl args, super] ++ attrDecls ++ layerDecls

   where initDecl     = printf (prefix ++ "def __init__(self, %s):")
         indent s     = prefix ++ prefix ++ s
         super        = indent "super().__init__()\n"
         args         = intercalate ", " vars
         attrDecls    = map (indent . genAttrDecl) vars
         layerDecls   = map (indent . (uncurry genDecl)) namedLayers

{- Creates an attribute per each dimension variable -}
genAttrDecl :: String -> String
genAttrDecl var = printf "self.%s = %s" var var


--TODO add an "size" expansion
{- Generates the forward method for the model -}
generateForward :: String -> [String] -> LayerNames -> String
generateForward prefix vars namedLayers = 
   unlines $ [forwardPre, indent . genSize . fst $ head namedLayers] 
             ++ calls ++ [indent "", indent "return tensor"]

   where forwardPre   = prefix ++ "def forward(self, tensor):\n"
         indent s     = prefix ++ prefix ++ s
         steps        = filter (callable . fst) namedLayers
         calls        = map (indent . (uncurry $ genCall vars)) steps

{- Generates calls of layers -}
genCall :: [String] -> Layer -> String -> String
genCall _ (RNN _ _ _ _) name             = printf "tensor, _ = self.%s(tensor)" name
genCall _ (RNNLast "LSTM" _ _ _ _) name  = printf "_, (tensor, _) = self.%s(tensor)" name
genCall _ (RNNLast _ _ _ _ _) name       = printf "_, tensor = self.%s(tensor)" name
genCall _ (Average index) _              = printf "tensor = t.mean(tensor, %d)" index
genCall _ (Permute dims) _               = printf "tensor = tensor.permute(%s)" (fmtIndex dims)

   --note this assumes a constant or single variable dimension
   where fmtIndex ds = intercalate ", " (map show ds)

genCall _ (Squeeze index) _              = printf "tensor = tensor.squeeze(%d)" index
genCall attrs (Reshape dims) _           = printf "tensor = t.reshape(tensor, (%s))" (fmtDim dims)

   where fmtDim ds  = intercalate ", " $ map (show . fmtVar) ds
         fmtVar v   = if hasAttrs v then addPrefix "self." v else v
         hasAttrs d = any (\ v -> v `elem` attrs) (dimVars d)

genCall _ _ name                         = printf "tensor = self.%s(tensor)" name


{- Generates a layer's declaration -}
genDecl :: Layer -> String -> String
genDecl layer name = printf "self.%s = %s" name (declare layer)
  
{- Generates the Pytorch object instantiation -}
declare :: Layer -> String
declare  (RNN name inFeat outFeat bi) = 
   printf py name (show inFeat) (show outFeat) (show bi)
      where py = "nn.%s(%s, %s, bidirectional=%s)"

declare (RNNLast name inFeat outFeat bi batch) = 
   printf py name (show inFeat) (show outFeat) (show bi) (show batch)
      where py = "nn.%s(%s, %s, bidirectional=%s, batch_first=%s)"

declare (Conv1d inF outF window stride) = 
   printf py (show inF) (show outF) window stride
      where py = "nn.Conv1d(%s, %s, %d, %d)" 

declare (Pool1d name window stride) = printf py name window stride
   where py = "nn.%s1d(%d, %d)"

declare (Conv2d inF outF window stride) = printf py (show inF) (show outF) window stride
   where py = "nn.Conv2d(%s, %s, %d, %d)"

declare (Pool2d name window stride) = printf py name window stride
   where py = "nn.%s2d(%d, %d)"

declare (Activation name) = printf "nn.%s()" name
declare (Linear inF outF) = printf py (show inF) (show outF)
   where py = "nn.Linear(%s, %s)"

declare _ = ""

{- Generates the line that unpacks the size of the given tensor -}
genSize :: Layer -> String
genSize (Input dims) = printf "%s = tensor.size()" assignment

   where genInput di
            | hasVars di   = show di
            | otherwise    = "_"

         assignment = intercalate ", " (map genInput dims)

genSize _ = ""

{- assigns attribute names to each layer -}
assignNames :: Model -> LayerNames
assignNames network = reverse $ assignHelper M.empty network []

assignHelper :: LayerCount -> Model -> LayerNames -> LayerNames
assignHelper _ [] names                      = names
assignHelper oldCounts (layer:rest) oldNames = assignHelper counts rest names
        
         --increment the counts
   where counts = M.insertWith (+) (layerType layer) 1 oldCounts         

         --update the names
         names  = (layer, makeName counts layer) : oldNames

{- Makes a name for the layer based on how many other layers exist previously -}
makeName :: LayerCount -> Layer -> String
makeName counts layer = if cnt > 1 then (printf "%s%d" name cnt) else name
  
         --lookup the count of the layer type, should never be "Nothing"
   where cnt  = case M.lookup (layerType layer) counts of
                 Just count -> count
                 _          -> 0

         name = lower $ layerType layer

{- Produces the layer's type -}
layerType :: Layer -> LayerType
layerType (Linear _ _)            = "Linear"
layerType (RNN name _ _ _)        = name
layerType (RNNLast name _ _ _ _)  = name
layerType (Conv1d _ _ _ _)        = "Conv1d"
layerType (Pool1d name _ _)       = printf "%s1d" name
layerType (Conv2d _ _ _ _)        = "Conv2d"
layerType (Pool2d name _ _)       = printf "%s2d" name
layerType (Average _)             = "mean"
layerType (Permute _)             = "permute"
layerType (Squeeze _)             = "squeeze"
layerType (Reshape _)             = "reshape"
layerType (Activation name)       = name
layerType (CELoss _)              = "CrossEntropyLoss"
layerType (Input _)               = "Input"
layerType (Broken _)              = "Broken"


{- Determines if the layer is functional or not -}
isFunctional :: Layer -> Bool
isFunctional (Average _) = True
isFunctional (Permute _) = True
isFunctional (Squeeze _) = True
isFunctional (Reshape _) = True
isFunctional (Input _)   = True
isFunctional (Broken _)  = True
isFunctional (CELoss _)  = True
isFunctional _           = False

{- Determines if the layer is a loss function or not -}
callable :: Layer -> Bool
callable (Input _)  = False
callable (Broken _) = False
callable (CELoss _) = False
callable _          = True

{- Produces the dimensions use by the layer -}
layerDims :: Layer -> [Dim]
layerDims (Linear inDim outDim)                 = [inDim, outDim]
layerDims (RNN _ inDim outDim _)                = [inDim, outDim]
layerDims (RNNLast _ inDim outDim _ _)          = [inDim, outDim]
layerDims (Conv1d inDim outDim _ _)             = [inDim, outDim]
layerDims (Conv2d inDim outDim _ _)             = [inDim, outDim]
layerDims (Reshape dims)                        = dims
layerDims (CELoss dims)                         = [dims]
layerDims (Input dims)                          = dims
layerDims _                                     = []

lower :: String -> String
lower = map toLower

spaceIndent :: String
spaceIndent = "    "

tabIndent :: String
tabIndent   = "\t"
