import os
import json
import pandas as pd
import dill

#os.environ['PROJECT_PATH'] = 'C:/Users/Admin/airflowe_33'
path = os.environ.get('PROJECT_PATH', '.')
#PATH = path


def get_latest_model() -> object:
    models_dir = f'{path}/data/models'
    model_files = os.listdir(models_dir)
    if model_files:
        latest_model_file = max(model_files)
        model_path = os.path.join(models_dir, latest_model_file)
        with open(model_path, 'rb') as f:
            model = dill.load(f)
        return model
    return None


def get_predicts() -> pd.DataFrame:
    preds = {
        'car_id': [],
        'pred': [],
    }
    test_cars = os.listdir(f'{path}/data/test')
    model = get_latest_model()

    for car_id in test_cars:
        with open(f'{path}/data/test/{car_id}', 'rb') as file:
            car = json.load(file)

        df = pd.DataFrame(car, index=[0])
        y = model.predict(df)
        preds['car_id'].append(car_id.split('.')[0])
        preds['pred'].append(y[0])
    return pd.DataFrame(preds)


def save_predictions(predictions: pd.DataFrame, predictions_path: str) -> None:
    predictions.to_csv(os.path.join(predictions_path, 'predictions.csv'), index=False)


def predict():
    predictions = get_predicts()
    save_predictions(predictions, f'{path}/data/predictions')


if __name__ == '__main__':
    predict()
