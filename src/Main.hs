
module Main where

import Network.Layer
import Network.Network
import Network.Neuron
import Network.Trainer

import Numeric.LinearAlgebra
import System.Random

main :: IO ()
main = do

  g <- newStdGen
  let l   = LayerDefinition sigmoidNeuron 2 connectFully
      l'  = LayerDefinition sigmoidNeuron 2 connectFully
      l'' = LayerDefinition sigmoidNeuron 1 connectFully

  let n = createNetwork normals g [l, l', l'']

  let t = BackpropTrainer 3 quadraticCost quadraticCost'

  let dat = [(fromList [0, 1], fromList [1]),
             (fromList [1, 1], fromList [0]),
             (fromList [1, 0], fromList [1]),
             (fromList [0, 0], fromList [0])]

  let n' = trainNTimes n t online dat 10000

  putStrLn "==> XOR predictions: "
  putStr "(0,0): "
  print $ predict (fromList [0, 0]) n'
  putStr "(1,0): "
  print $ predict (fromList [1, 0]) n'
  putStr "(0,1): "
  print $ predict (fromList [0, 1]) n'
  putStr "(1,1): "
  print $ predict (fromList [1, 1]) n'

  putStrLn "\n==> Trained network: "
  print $ map layerToShowable $ layers n'

{-
  saveNetwork "xor.ann" n'

  putStrLn "==> Network saved and reloaded: "
  n'' <- loadNetwork "xor.ann" [l, l', l'']

  print $ predict (fromList [0, 0]) n''
  print $ predict (fromList [1, 0]) n''
  print $ predict (fromList [0, 1]) n''
  print $ predict (fromList [1, 1]) n''
-}

