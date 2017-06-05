from training_pipeline.lr_w_search.model_utils import *


from random import uniform
import matplotlib.pyplot as plt
import numpy as np

class lossHistory:

    def __init__(self):
        self.train_loss_history = []
        self.test_loss_history = []

    def generateLoss(self):
        self.train_loss_history.append(uniform(0.0, 10.0))
        self.test_loss_history.append(uniform(0.0, 10.0))

def test_plotAndSaveData(lossHistory):

    for i in range(10):
        lossHistory.generateLoss()

    plotAndSaveData(lossHistory,"Loss",4e-7,0.0032,"Loss History")


def test_makeDirs():
    lr = 10 ** np.random.uniform(-3.0, -6.0)
    ws = 10 ** np.random.uniform(-2.0, -4.0)

    dir_name = '{lr:%8f}-{ws:%6f}' % (lr, ws)
    dir_path = makeDir(dir_name)
    print(dir_path)

def test_plotAndSaveSession():
    final_accuracies = []
    learning_rates = []
    weight_scales = []

    for i in range(200):
        lr = 10 ** np.random.uniform(-3.0, -6.0)
        ws = 10 ** np.random.uniform(-2.0, -4.0)
        acc = uniform(0.0, 100.0)
        learning_rates.append(lr)
        weight_scales.append(ws)
        final_accuracies.append(acc)

    plotAndSaveSession(learning_rates,weight_scales,final_accuracies)

test_plotAndSaveSession()