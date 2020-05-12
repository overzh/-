import numpy, os
from User_draw import user_draw
from matplotlib import pyplot
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Activation
from keras.optimizers import SGD

(x_train,y_train),(x_test,y_test) = mnist.load_data()

x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)

y_train = np_utils.to_categorical(y_train, 10)
y_test = np_utils.to_categorical(y_test, 10)

run_fit = False

if os.path.exists('NN_model.json'):
    #從設定檔建立模組
    NN = model_from_json(open('NN_model.json').read())
    NN.load_weights('NN_weight.h5')
else:
    run_fit = True

    #建立模組
    NN = Sequential()

    #第一層
    NN.add(Dense(500, input_dim=784))
    NN.add(Activation('sigmoid'))

    #第二層
    NN.add(Dense(250))
    NN.add(Activation('sigmoid')) 

    #輸出層
    NN.add(Dense(10))
    NN.add(Activation('softmax'))

#進行編譯
#loss      損失函數
#optimizer 優化方法
NN.compile(loss='mse', optimizer=SGD(lr=0.05), metrics = ['accuracy'])

#進行訓練
if run_fit:
    #batch_size 一次訓練幾筆資料
    #epochs     一共訓練幾次
    NN.fit(x_train, y_train, batch_size=100, epochs=3)

    #儲存結果
    NN_json = NN.to_json()
    open('NN_model.json','w').write(NN_json)
    NN.save_weights('NN_weight.h5',)

user_draw = user_draw.user_draw(28, 28)
user_draw.show()

if 236 in user_draw.user_table:
    x_test = user_draw.user_table.reshape(1, 784)

predictions = NN.predict_classes(x_test)

if len(x_test) < 5:
    pick_num = len(x_test)
else:
    pick_num =  5

pick = numpy.random.randint(1, len(x_test)+1, pick_num)

for i in range(pick_num):
    #pyplot.subplot(1, pick_num, i+1)
    pyplot.imshow(x_test[pick[i]-1].reshape(28,28), cmap='Greys')
    pyplot.title(predictions[pick[i]-1])
    #pyplot.axis('off')
    pyplot.show()