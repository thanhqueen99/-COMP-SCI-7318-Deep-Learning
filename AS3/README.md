# Bitcoin RNN Crypto Prediction

[![forthebadge made-with-python](http://ForTheBadge.com/images/badges/made-with-python.svg)](https://www.python.org/)

The goal of this project is to build up the cryptocurrency project and mainly focus on three types of pair coins which is BTCUSDT, MATICBUSDT, ETHUSDT respectively.

**Summary**
1. Prerequisite
2. Scraping Dataset from Binance stock
3. Build up the model RNN

# 1. Environment set up
You need to have python installation before
```
git clone https://github.com/thanhqueen99/-COMP-SCI-7318-Deep-Learning.git
cd AS3/
pip install -r requirements.txt
python src/main.py
```

# 2. Data creation from binance stock
Whenever you run ```python src/main.py``` the tools/scraping_data/scraping_bitcoin_RNN.py is called when executed.</br>
The dataset will create 3 types pair of coins as defined at `src/model_training.py`. If you want to create your own dataset, go to `src/model_training.py` and modifying the pair in pair list types.
- `PAIR`: list type -- Concurrency pair `BTCUSDT; MATICBUSDT; ETHUSDT` by default.
- `INTERVAL`: int type -- The interval time candle fetch , `1h` by default.
- `INTERVAL_MS`: int type -- The interval equivalent in `int(60 * 60 * 1e6)` by default.

To generate the dataset we set the interval time `start: Date - end: Date`  
- `start`: date type -- Start fetching date.
- `end`: date type -- End fetching date.
- `path`: str type -- Path to save Dataframe in Excel format
In order to set more dataset you modify the content `start and end date` that you want to create the dataset.

# 3. Build up the RNN model
The tensorflow has been used to build the predictive binance stock model. Defined the ModelBuilder class which has a various implementation for the pre-processing dataset used for model. Normalization dataset technique used in this project. <br/>

`ModelBuilder` is a class which can simply create and train a cryptocurrency based on RNN LSTM techniques. If you want to create and add more layer in order to do go to `src/cryto_model_build.py` contribute it.
- `Params`:
```
1. SEQ_LEN = 100
2. TRAIN_SPLIT = 0.9
3. BATCH_SIZE = 64
4. EPOCHS = 50
5. VALIDATION_SPLIT = 0.15
6. PATH_IN = PATH_EXCEL (Set excel path dataset)
7. PATH_OUT = PATH_OUT (Set Model where saved)
8. LOADED = LOAD (Set the Model is trained or not if True load_model and continue train else create model and start train from beginning)
9. MODEL = SEQUENTIAL() else LOAD MODEL if existed
10. SCALER = MinMaxScaler()
11. TRAINING_DATA = None
12. HISTORY, DF = NONE, NONE
13. X_TRAIN, Y_TRAIN, X_TEST, Y_TEST = NONE, NONE, NONE, NONE
14. LOSS, ACC = NONE, NONE
```
- `Function used`:
```
1. open_excel() : Public
2. _normalization() : Private
3. _create_model() : Private
4. build_model() : Public
5. train_model() : Public
6. evaluate_model(assert_dir, pair) : Public
7. test_prediction(assert_dir, pair) : Public
8. future_prediction(assert_dir, pair) : Public
9. save_model() : Public
```
- The model saved in /saved_model whenever finish the training process.