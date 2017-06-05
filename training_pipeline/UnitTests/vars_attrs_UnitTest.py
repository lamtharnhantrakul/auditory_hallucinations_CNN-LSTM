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

lossHistory = lossHistory()

for i in range(10):
    lossHistory.generateLoss()

param_history = (vars(lossHistory))

for param in param_history:
    param_values = param_history[param]
    epochs = np.arange(len(param_values))  # epochs is 1,2,3...[num items in param values]
    plt.plot(epochs, param_values)

plt.xlabel('epochs')
plt.ylabel('loss')
# plt.show()
plt.draw()
lrate = 4e-6
weight = 0.005
#file_name = 'lr:{lrate:%6f}-ws:{weight:%6f}'.format(lrate,weight)
file_name = '{lr:%s}-{ws:%s}.png' % (str(lrate), str(weight))
plt.savefig('../graphs/training_history/' + file_name)
#plt.savefig('../training_pipeline/graphs/training_history/test_graph.png')
plt.close()

