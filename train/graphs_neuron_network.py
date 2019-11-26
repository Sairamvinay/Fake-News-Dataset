import matplotlib.pyplot as plt

# graph of neural network model of both loss and accuracy 
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
