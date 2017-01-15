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
        B = np.array([[1, 2], [3, 4], [5, 6]])
        self.assertEqual(2, np.ndim(B))
        self.assertEqual((3, 2), B.shape)

    def testDot1(self):
        u"""2x2の行列と2x2の行列の内積"""
        A = np.array([[1, 2], [3, 4]])
        self.assertEqual((2, 2), A.shape)
        B = np.array([[5, 6], [7, 8]])
        self.assertEqual((2, 2), B.shape)
        self.assertEqual(np.array([[19, 22], [43, 50]]).all(),
                         np.dot(A, B).all())


    def testDot2(self):
        u"""2x3の行列と3x2の行列の内積"""
        A = np.array([[1, 2, 3], [4, 5, 6]])
        self.assertEqual((2, 3), A.shape)
        B = np.array([[1, 2], [3, 4], [5, 6]])
        self.assertEqual((3, 2), B.shape)
        self.assertEqual(np.array([[22, 28], [49, 64]]).all(),
                         np.dot(A, B).all())


    def testDot3(self):
        u"""3x2の行列と2x1の行列の内積"""
        A = np.array([[1, 2], [3, 4], [5, 6]])
        self.assertEqual((3, 2), A.shape)
        B = np.array([7, 8])
        self.assertEqual((2,), B.shape)
        self.assertEqual(np.array([23, 53, 83]).all(),
                         np.dot(A, B).all())



if __name__ == '__main__':
    unittest.main()
