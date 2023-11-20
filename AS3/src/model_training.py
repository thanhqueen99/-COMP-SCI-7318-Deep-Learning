from crypto_model_build import *
import plotly.graph_objects as go
from datetime import datetime, date
import sys
# Set sys module at top of the file
file_location = os.path.abspath(__file__)  # Get current file abspath
root_directory = os.path.dirname(file_location)  # Get root dir
sys.path.append(os.path.join(root_directory, '..'))
from tools.scraping_data.scraping_bitcoin_RNN import *

dataset_dir = os.path.join(root_directory, '..', '..' '/dataset')
assets_dir = os.path.join(dataset_dir, 'assets')
PAIR = ["BTCUSDT", "MATICBUSD", "ETHUSDT"]


# index 0: BTCUSDT, index 1: MATICBUSD, index 2: ETHUSDT

def delete_scraping_data():
    # Remove all existed dataset file and to renew ones
    files = os.listdir(dataset_dir)
    for file in files:
        file_path = os.path.join(dataset_dir, file)
        if os.path.isfile(file_path):
            os.remove(file_path)


def read_all_scraping_data(start, end, pair):
    dfs = []
    files = os.listdir(dataset_dir)
    if len(files) > 0:
        for file in files:
            file_path = os.path.join(dataset_dir, file)
            if os.path.isfile(file_path) and pair in file:
                df = pd.read_excel(file_path)
                dfs.append(df)
    else:
        generate_dataset(start, end)
    return dfs, file_path


# Generating the figure to visualize for stock respectively [BTCUSDT; MATICUSDT; ETHUSDT] with index 0, 1, 2
def go_figure(df, pair) -> pd.DataFrame:
    fig_plot = go.Figure(data=go.Ohlc(x=df['Open time'].apply(lambda x: datetime.datetime.fromtimestamp(x / 1000)),
                                      open=df['Open price'],
                                      high=df['High price'],
                                      low=df['Low price'],
                                      close=df['Close price']))
    # fig_plot.show()
    pair_dir = os.path.join(assets_dir, pair)
    if not os.path.exists(pair_dir):
        os.mkdir(pair_dir)
    image_file = os.path.join(pair_dir, "{}.png".format(pair))
    fig_plot.write_image(image_file)


def scraping_data_export(start, end, pair) -> pd.DataFrame:
    request_num = len([k for k in range(start.get_timestamp(), end.get_timestamp(), INTERVAL_MS)])
    print(f'Request Data: {request_num}')

    # Generate the newest datasets
    excel_file_gen = f'{pair}_{start.get_timestamp()}_{end.get_timestamp()}.xlsx'
    dataset_file_dir = os.path.join(dataset_dir, excel_file_gen)
    create_dataset(start, end, pair, dataset_file_dir)


def generate_dataset(start, end):
    # Delete all the duplicated datasets and ready for a new datasets
    delete_scraping_data()
    for pair in PAIR:
        scraping_data_export(start, end, pair)


def model_train_evaluate(model, pair):
    model.train_model()
    model.evaluate_model(assets_dir, pair)
    model.save_model()


def model_prediction(model, pair):
    model.test_prediction(assets_dir, pair)
    model.future_prediction(assets_dir, pair)
