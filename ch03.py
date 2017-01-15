#!/usr/bin/env python
# -*- coding: utf-8 -*-
import unittest
import numpy as np
import matplotlib.pylab as plt


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def ReLU(x):
    u"""Rectified Linear Unit"""
    return np.maximum(0, x)


class Test(unittest.TestCase):

    def testSigmoid(self):
        x = np.array([-1.0, 1.0, 2.0])
        self.assertEqual(np.array([0.26894142, 0.73105858, 0.88079708]).all(),
                         sigmoid(x).all())

    def testSigmoidGraph(self):
        x = np.arange(-5.0, 5.0, 0.1)
        y = sigmoid(x)
        plt.plot(x, y)
        plt.ylim(-0.1, 1.1)  # y軸の範囲を指定
        #  plt.show()


    def testNumpy(self):
        A = np.array([1, 2, 3, 4])
        self.assertEqual(1, np.ndim(A))
        self.assertEqual((4,), A.shape)
        self.assertEqual(4, A.shape[0])


if __name__ == '__main__':
    unittest.main()
