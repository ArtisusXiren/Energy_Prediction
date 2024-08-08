import joblib
import numpy as np
import os
def load_model():
    path=r"C:\Users\ArtisusXiren\Desktop\Enefit\Enefit\myapp"
    xgb_path=os.path.join(path,'xgb_model.pkl')
    random_path=os.path.join(path,'random.pkl')
    xgb=joblib.load(xgb_path)
    random=joblib.load(random_path)
    return xgb,random
def predict(model,input_data):
    input_data=np.array(input_data).reshape(1,-1)
    return model.predict(input_data)
    