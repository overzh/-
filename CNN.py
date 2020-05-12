import numpy, os
from User_draw import user_draw
from matplotlib import pyplot
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D
from keras.optimizers import SGD

(x_train,y_train),(x_test,y_test) = mnist.load_data()

x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28, 1)

y_train = np_utils.to_categorical(y_train, 10)
y_test = np_utils.to_categorical(y_test, 10)

run_fit = False

if os.path.exists('CNN_model.json'):
    #從設定檔建立模組
    CNN = model_from_json(open('CNN_model.json').read())
    CNN.load_weights('CNN_weight.h5')
else:
    run_fit = True

    #建立模組
    CNN = Sequential()

    #第一層 Convolution, Pooling
    CNN.add(Conv2D(10, (3,3), padding='same', input_shape=(28, 28, 1)))
    CNN.add(Activation('relu'))

    CNN.add(MaxPooling2D(pool_size=(2, 2)))

    #第二層 Convolution, Pooling
    CNN.add(Conv2D(20, (3,3), padding='same'))
    CNN.add(Activation('relu'))

    CNN.add(MaxPooling2D(pool_size=(2, 2)))

    # #第三層 Convolution, Pooling
    # CNN.add(Conv2D(128, (3,3), padding='same'))
    # CNN.add(Activation('relu'))

    # CNN.add(MaxPooling2D(pool_size=(2, 2)))

    #將二維向量拉成一維
    CNN.add(Flatten())

    #第四層 Dense
    CNN.add(Dense(200))
    CNN.add(Activation('relu'))

    #輸出層
    CNN.add(Dense(10))
    CNN.add(Activation('softmax'))

#進行編譯
#loss      損失函數
#optimizer 優化方法
CNN.compile(loss='mse', optimizer=SGD(lr=0.05), metrics = ['accuracy'])

#進行訓練
if run_fit:
    #batch_size 一次訓練幾筆資料
    #epochs     一共訓練幾次
    CNN.fit(x_train, y_train, batch_size=100, epochs=1)

    #儲存結果
    CNN_json = CNN.to_json()
    open('CNN_model.json','w').write(CNN_json)
    CNN.save_weights('CNN_weight.h5',)

#測試
user_draw = user_draw.user_draw(28, 28)
user_draw.show()
if 236 in user_draw.user_table:
    x_test = user_draw.user_table.reshape(1, 28, 28, 1)

predictions = CNN.predict_classes(x_test)

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