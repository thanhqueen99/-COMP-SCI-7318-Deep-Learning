from model_training import *

file_location = os.path.abspath(__file__)  # Get current file abspath
root_directory = os.path.dirname(file_location)  # Get root dir
# Params
today = date.today()
# Scraping data back two year count from present
_start = Date((today.year - 2), today.month, today.day, 0)
_end = Date(today.year, today.month, today.day, 0)


def main():
    # Setting up model BTCUSDT Build, Train, Evaluate and Save the [BTCUSDT;MATICUSDT;ETHUSDT] model
    for pair in PAIR:
        # Read all scraping data, if is not existed create a new ones
        dfs, df_dirs = read_all_scraping_data(_start, _end, pair)
        # Plot the image dataset and saved into dataset/assets/<PAIR TYPE>
        go_figure(dfs[0], pair)
        model_path = os.path.join(root_directory, '..', 'saved_model', pair)
        # Check if the model is existed in saved model folder, otherwise build model in scratch
        load_model = os.path.exists(model_path)
        print("Model Builder: {}, load_model: {}".format(pair, load_model))
        model = ModelBuilder(df_dirs, pair, load_model)
        # Load model when the model is already created otherwise create a new ones
        if not load_model:
            model.build_model()
        model_train_evaluate(model, pair)
        model_prediction(model, pair)


if __name__ == "__main__":
    main()
