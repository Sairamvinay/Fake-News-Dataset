#this function is supposed to be called inside lstm.py file

import lstm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

##for acc graphs
#error_list = []
#for i in lstm.acc_list:
#    error_list.append(1-i)
#plt.plot(error_list)
#plt.title('model accuracy')
#plt.ylabel('test error')
#plt.xlabel('epoch')
#plt.legend(['test'], loc='upper left')
#plt.show()


def graphs_nn(error_list,epochs):
    plt.plot(epochs, error_list)
    plt.title('model accuracy')
    plt.ylabel('test error')
    plt.xlabel('epoch')
    plt.legend(['test'], loc='upper left')
    plt.show()

def main():
    error_list = []
    for i in lstm.acc_list:
        error_list.append(1-i)
    graphs_nn(error_list,30)
#epochs are not fixed at 30 here, but since this is local variable, so indeally change it to lstm._.epochs
