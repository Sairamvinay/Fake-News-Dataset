import lstm
import matplotlib.pyplot as plt


#for acc graphs
error_list = []
for i in lstm.acc_list:
    error_list.append(1-i)
plt.plot(error_list)
plt.title('model accuracy')
plt.ylabel('test error')
plt.xlabel('epoch')
plt.legend(['test'], loc='upper left')
plt.show()
