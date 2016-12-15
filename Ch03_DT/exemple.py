# -*- coding: utf-8 -*-
"""
Created on Thu Dec 15 15:50:57 2016

@author: Vanguard
"""

import trees
import treePlotter

fr = open('lenses.txt')
lenses = [inst.strip().split('\t') for inst in fr.readlines()]
lensesLabels=['age','prescript','astigmatic','tearRate']
lensesTree = trees.createTree(lenses,lensesLabels)
treePlotter.createPlot(lensesTree)
