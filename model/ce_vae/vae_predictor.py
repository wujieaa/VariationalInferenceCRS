from tqdm import tqdm

import torch

def predict(prediction_score, topK, matrix_Train):
    matrix_Train=1-matrix_Train
    prediction_score=prediction_score*matrix_Train
    prediction=prediction_score.topk(topK,dim=-1).indices
    return prediction



def predict_keyphrase(prediction_score, topK):
    return prediction_score.topk(topK,dim=-1).indices


