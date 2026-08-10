FROM tensorflow/serving:latest

COPY serving_model/heart-failure-model /models/heart-failure-model

ENV MODEL_NAME=heart-failure-model
