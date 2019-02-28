# test_pyplot.py
import numpy as np
import matplotlib.pyplot as plt

# x = [0,5,9,10,15]
# y = [0,1,2,3,4]
pruned_percents = [0, 6.25, 12.5, 18.75, 25, 31.25, 37.5, 43.75, 50, 56.25, 62.5, 68.75, 75, 81.25, 87.5, 93.75]
pruned_accuracies = [0.73554, 0.67904, 0.49366, 0.40764, 0.31172, 0.36386, 0.28626, 0.24244, 0.21532, 0.1707, 0.09952, 0.06316, 0.06822, 0.02472, 0.02286, 0.01426]
finetuned_accuracies = [0.73554, 0.62758, 0.59692, 0.54154, 0.49476, 0.44424, 0.41248, 0.38450, 0.33348, 0.27684, 0.21748, 0.17024, 0.11616, 0.08536, 0.05434, 0.02102]
plt.title("Finetune vs Pruned Accuracy")
plt.xlabel("Pruned percents")
plt.ylabel("Validation Accuracy")
plt.plot(pruned_percents, pruned_accuracies, label='pruned')
plt.plot(pruned_percents, finetuned_accuracies, label='finetune')
# plt.xlim((0,100))
# plt.ylim((0,1.))
plt.xticks(np.arange(0, 100, 10.0))
plt.yticks(np.arange(0, 1, 0.1))
plt.legend()
plt.show()