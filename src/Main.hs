
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

  let t = BackpropTrainer (3 :: Float) quadraticCost quadraticCost'

  let dat = [(fromList [0, 1], fromList [1]), (fromList [1, 1], fromList [0]), (fromList [1, 0], fromList [1]), (fromList [0, 0], fromList [0])]

  let n' = trainNTimes n t online dat 10000

  putStrLn "==> XOR predictions: "
  print $ predict (fromList [0, 0]) n'
  print $ predict (fromList [1, 0]) n'
  print $ predict (fromList [0, 1]) n'
  print $ predict (fromList [1, 1]) n'

  saveNetwork "xor.ann" n'

  putStrLn "==> Network saved and reloaded: "
  n'' <- loadNetwork "xor.ann" [l, l', l'']

  print $ predict (fromList [0, 0]) n''
  print $ predict (fromList [1, 0]) n''
  print $ predict (fromList [0, 1]) n''
  print $ predict (fromList [1, 1]) n''

