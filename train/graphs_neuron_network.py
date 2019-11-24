import lstm
import matplotlib.pyplot as plt


#for acc graphs
acc_list = lstm.acc_list
plt.plot(acc_list)
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['test'], loc='upper left')
plt.show()
