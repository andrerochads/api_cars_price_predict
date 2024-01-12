import pickle
import pandas as pd
import numpy as np
import math


class CarsPricePred(object):
    def __init__(self):
        # self.home_path='C:/Users/andre/repos/cursos_ds/9_car_price_prediction/' self.home_path + teste local
        self.ano_de_fabricacao_scaler = pickle.load(open('parameter/ano_de_fabricacao.pkl', 'rb'))
        self.ano_modelo_scaler =        pickle.load(open('parameter/ano_modelo.pkl', 'rb'))
        self.cilindradas_scaler =       pickle.load(open('parameter/cilindradas.pkl', 'rb'))
        self.hodometro_scaler =         pickle.load(open('parameter/hodometro.pkl', 'rb'))
        self.cidade_vendedor_encoder =  pickle.load(open('parameter/cidade_vendedor_tgt_encoder.pkl', 'rb'))
        self.cambio_encoder =           pickle.load(open('parameter/cambio_tgt_encoder.pkl', 'rb'))
        self.combustivel_encoder =      pickle.load(open('parameter/combustivel_tgt_encoder.pkl', 'rb'))
        self.cor_encoder =              pickle.load(open('parameter/cor_tgt_encoder.pkl', 'rb'))
        self.modelo_encoder =           pickle.load(open('parameter/modelo_tgt_encoder.pkl', 'rb'))
        self.marca_encoder =            pickle.load(open('parameter/marca_tgt_encoder.pkl', 'rb'))
        self.tipo_encoder =             pickle.load(open('parameter/tipo_tgt_encoder.pkl', 'rb'))

    def data_preparation(self, df):

        ## Rescaling and Encoding
        df['ano_de_fabricacao'] = self.ano_de_fabricacao_scaler.transform(df[['ano_de_fabricacao']].values)
        df['ano_modelo'] = self.ano_modelo_scaler.transform(df[['ano_modelo']].values)
        df['cilindradas'] = self.cilindradas_scaler.transform(df[['cilindradas']].values)
        df['hodometro'] = self.hodometro_scaler.transform(df[['hodometro']].values)
        df['cidade_vendedor'] = self.cidade_vendedor_encoder.transform(df[['cidade_vendedor']])
        df['cambio'] = self.cambio_encoder.transform(df[['cambio']])
        df['combustivel'] = self.combustivel_encoder.transform(df[['combustivel']])
        df['cor'] = self.cor_encoder.transform(df[['cor']])
        df['modelo'] = self.modelo_encoder.transform(df[['modelo']])
        df['marca'] = self.marca_encoder.transform(df[['marca']])
        df['tipo'] = self.tipo_encoder.transform(df[['tipo']])
        df['blindado'] = df['blindado'].apply(lambda x: 1 if x == 'Sim' else 0)

        return df    

    def get_prediction(self, model, df_car, df_to_pred):

        # prediction
        price_pred = model.predict(df_to_pred)

        # join pred into the original data
        df_car['preco_sugerido'] = np.expm1(price_pred)

        return df_car.to_json(orient='records')