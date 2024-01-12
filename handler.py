import os
import pickle
import pandas as pd
from flask import Flask, request, Response
from cars_pred.CarsPricePred import CarsPricePred

# Loading ML model
model = pickle.load(open('model/model_car_price_pred.pkl', 'rb')) # deploy
# model = pickle.load(open('C:/Users/andre/repos/cursos_ds/9_car_price_prediction/model/model_car_price_pred.pkl', 'rb'))


# initialize API - ( End point ativo... então ela terá o end point ativo e o modelo em memória esperando requisição)
app = Flask(__name__)
@app.route('/cars-price-predict', methods=['POST'])

def rossmann_predict():
    car_json = request.get_json()

    if car_json: # there is data
        if isinstance(car_json, dict): # unique example
            df_car = pd.DataFrame(car_json, index=[0])
        else: # multiple example
            df_car = pd.DataFrame(car_json, columns=car_json[0].keys())
            #[0].keys() para usar como colunas as primeiras chaves no df, mas pega o valor da primeira sim. 

        # Instantiate Rossmann class
        pipeline = CarsPricePred()

        # -----------------------------------------------
        # Não necessário nesse caso.
        # data cleaning
        # df1 = pipeline.data_cleaning( df_car )
        # feature engineering
        # df2 = pipeline.feature_engineering( df1 )
        # -----------------------------------------------

        # data preparation
        df_to_pred = pipeline.data_preparation(df_car)

        # prediction
        df_response = pipeline.get_prediction(model, df_car, df_to_pred)

        return df_response

    else:
        return Response('{}', status=200, mimetype='application/json')

if __name__ == '__main__':
    port = os.environ.get('PORT', 5000)
    app.run(host='0.0.0.0', port=port)