import numpy, os
from matplotlib import pyplot
from keras.datasets import imdb
from keras.models import Sequential, model_from_json, load_model
from keras.layers import Dense, Activation, Embedding, LSTM
from keras.preprocessing import sequence

# 讀入資料 限制常見10000字
(x_train,y_train),(x_test,y_test) = imdb.load_data(num_words=10000)

#資料處理
x_train = sequence.pad_sequences(x_train, maxlen=100)
x_test = sequence.pad_sequences(x_test, maxlen=100)

print(x_train[0])

run_fit = False

#if os.path.exists('RNN_model.json'):
if os.path.exists('RNN.h5'):
    #從設定檔建立模組
    #RNN = model_from_json(open('RNN_model.json').read())
    #RNN.load_weights('RNN_weight.h5')
    RNN = load_model('RNN.h5')
else:
    run_fit = True

    #建立模組
    RNN = Sequential()

    RNN.add(Embedding(10000, 128))

    RNN.add(LSTM(120))

    #RNN.add(Dense(1))
    #RNN.add(Activation('sigmoid')) 
    RNN.add(Dense(1, activation='sigmoid'))

#進行編譯
#loss      損失函數
#optimizer 優化方法
RNN.compile(loss='binary_crossentropy', optimizer='adam', metrics = ['accuracy'])

#進行訓練
if run_fit:
    #batch_size 一次訓練幾筆資料
    #epochs     一共訓練幾次
    RNN.fit(x_train, y_train, batch_size=50, epochs=1)

    #儲存結果
    #RNN_json = RNN.to_json()
    #open('RNN_model.json','w').write(RNN_json)
    #RNN.save_weights('RNN_weight.h5',)
    RNN.save('RNN.h5')

#測試