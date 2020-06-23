from __future__ import division
import numpy as np
import sys, os, argparse
from os.path import isfile, join, isdir
sys.path.insert(0, './caffe/python')
import caffe
parser = argparse.ArgumentParser(description='Training dds nets.')
parser.add_argument('--gpu', type=int, help='gpu ID', default=0)
parser.add_argument('--solver', type=str, help='solver', default='./model/nice/solver_bottom_up.prototxt')
parser.add_argument('--weights', type=str, help='base model', default='./snapshot/VGG_ILSVRC_16_layers.caffemodel')
args = parser.parse_args()
assert isfile(args.weights) and isfile(args.solver)

caffe.set_mode_gpu()
caffe.set_device(args.gpu)

solver = caffe.SGDSolver(args.solver)
solver.net.copy_from(args.weights)
solver.solve()

