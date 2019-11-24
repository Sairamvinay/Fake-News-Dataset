#this function is supposed to be called inside lstm.py file
import matplotlib.pyplot as plt

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


def graphs_nn(loss, val_loss, accuracy, val_accuracy):
    epochs = range(1, len(loss) + 1)
    plt.figure(1)
    plt.plot(epochs, loss, label='train')
    plt.plot(epochs, val_loss, label='test')
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend()
    plt.figure(2)
    
    plt.plot(epochs, accuracy, label='train')
    plt.plot(epochs, val_accuracy, label='test')
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.legend()
    plt.show()

# def main():
#     error_list = []
#     for i in lstm.acc_list:
#         error_list.append(1-i)
#     graphs_nn(error_list,30)
#epochs are not fixed at 30 here, but since this is local variable, so indeally change it to lstm._.epochs
