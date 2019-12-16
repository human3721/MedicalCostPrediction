import sys
import os
import uuid
import json
import tensorflow as tf
from keras import models 
from keras.preprocessing.sequence import pad_sequences
import pandas as pd
import numpy as np
import falcon
from io import StringIO
import logging
import requests
import gunicorn 
import pickle as pkl
from waitress import serve

path = "char_cnn"

class Model:
    """
    Prediction Service to make predictions from local model
    """

    def __init__(self, model_dir):
        self.model = None
        self.graph = None
        self.path = model_dir

    def get_model(self):
        """
        Retrieves model artifacts
        """
        if not self.model:
            logging.info("Loading model")
            try:
                model_name = 'model.h5'
                model = models.load_model(os.path.join(self.path, model_name))

                model._make_predict_function()
                model.summary()

                self.model = model
                self.graph = tf.get_default_graph()
            except Exception as e:
                logging.error("Failed to load model due to error: {}".format(str(e)))

        return self.model, self.graph

    def predict(self, X):
        """
        Predicts X using model
        """
        clf, gph = self.get_model()
        with gph.as_default():
            return clf.predict(X)
        
class Cache:
    labels = open('data/labels.txt').read().splitlines()
    model = Model(path)
    # calling tokenizer
    with open(os.path.join(path,'tokenizer.pkl'), 'rb') as f:
        tokenizer = pkl.load(f)
    desc_length = 200
    
class Ping(object):
    """
    /ping for Health Check
    """

    def on_get(self, req, resp):
        health = Model(path).get_model() is not None
        resp.body = json.dumps("\n", ensure_ascii=False)
        resp.status = falcon.HTTP_200 if health else falcon.HTTP_404
        
class Invocation(object):
    """
    /invocations to process a prediction
    """
    
    def on_post(self, req, resp):
        try:
            data = req.stream.read().decode("utf-8")
            data_df = pd.read_csv(StringIO(data), names=["description"])
            X = self.prepare_input(
                data_df, Cache.desc_length, Cache.tokenizer, Cache.labels
            )
            
            y_pred, y_conf = self.predict(X)
            comment = "\n\n Predicted cost range in $: \n Very-Low in (23.8, 121.9) \n Low in (121.9, 660.3) \n Moderate-Low in (660.3, 1580.8) \n Moderate-High in (1580.8, 2647.1) \n High in (2647.1, 4573.3) \n Very-High in (4573.3, 10269.4)"
            resp.body = "The predicted cost range is " + str(y_pred) + "\n The prediction confidence % is " + str(np.round(y_conf*100,1)) + comment
            resp.context_type = "text/plain"
            resp.status = falcon.HTTP_200

        except Exception as e:
            resp.status = falcon.HTTP_500
            resp.body = str(e)
            resp.context_type = "text/plain"
            
    def prepare_input(self, data_df, desc_length, tokenizer, labels):
        desc_input = pad_sequences(tokenizer.texts_to_sequences([str(k).lower() for k in data_df.description]),
            maxlen = desc_length, dtype=int,
            padding='post', truncating='post', value=0
        )
        x = {
            'desc_input': desc_input
        }
        return x
    
    
    def predict(self, X):
        y_raw = Cache.model.predict(X)
        y_pred = np.argmax(y_raw, axis=-1)
        y_conf = np.asarray([y_raw[i,p] for i,p in enumerate(y_pred)])
        y_pred_labels = [Cache.labels[i] for i in y_pred]
        
        return y_pred_labels, y_conf
    
api = falcon.API()

# Defining App Routes
api.add_route("/ping", Ping())
api.add_route("/invocations", Invocation())


serve(api, host='127.0.0.1', port=8080)