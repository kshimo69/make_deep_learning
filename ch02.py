#!/usr/bin/env python
# -*- coding: utf-8 -*-
import unittest
import numpy as np


def AND1(x1, x2):
    w1, w2, theta = 0.5, 0.5, 0.7
    tmp = x1 * w1 + x2 * w2
    if tmp <= theta:
        return 0
    elif tmp > theta:
        return 1


def AND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.7
    tmp = np.sum(w * x) + b
    if tmp <= 0:
        return 0
    else:
        return 1


def NAND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([-0.5, -0.5])
    b = 0.7
    tmp = np.sum(w * x) + b
    if tmp <= 0:
        return 0
    else:
        return 1


def OR(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.2
    tmp = np.sum(w * x) + b
    if tmp <= 0:
        return 0
    else:
        return 1


def XOR(x1, x2):
    u"""XORはNANDとORのANDを取れば実現できる

    x1 NAND x2 -> s1
    x1 OR   x2 -> s2
    s1 AND  s2 -> XOR

    x1 x2 | s1 s2 | y
    ------+-------+---
    0  0  | 1  0  | 0
    1  0  | 1  1  | 1
    0  1  | 1  1  | 1
    1  1  | 0  1  | 0
    """
    s1 = NAND(x1, x2)
    s2 = OR(x1, x2)
    y = AND(s1, s2)
    return y


class Test(unittest.TestCase):

    def testAND1(self):
        self.assertEqual(AND1(0, 0), 0)
        self.assertEqual(AND1(1, 0), 0)
        self.assertEqual(AND1(0, 1), 0)
        self.assertEqual(AND1(1, 1), 1)

    def testAND2(self):
        self.assertEqual(AND(0, 0), 0)
        self.assertEqual(AND(1, 0), 0)
        self.assertEqual(AND(0, 1), 0)
        self.assertEqual(AND(1, 1), 1)

    def testNAND(self):
        self.assertEqual(NAND(0, 0), 1)
        self.assertEqual(NAND(1, 0), 1)
        self.assertEqual(NAND(0, 1), 1)
        self.assertEqual(NAND(1, 1), 0)

    def testOR(self):
        self.assertEqual(OR(0, 0), 0)
        self.assertEqual(OR(1, 0), 1)
        self.assertEqual(OR(0, 1), 1)
        self.assertEqual(OR(1, 1), 1)

    def testXOR(self):
        self.assertEqual(XOR(0, 0), 0)
        self.assertEqual(XOR(1, 0), 1)
        self.assertEqual(XOR(0, 1), 1)
        self.assertEqual(XOR(1, 1), 0)


if __name__ == '__main__':
    unittest.main()
