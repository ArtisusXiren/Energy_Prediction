from django.shortcuts import render
from .ml_utils import load_model,predict
from django.http import JsonResponse
xgb,random=load_model()
def predict_view(request):
    if request.method == 'POST':
        input_data=request.POST.get('input_data')
        if input_data:
            input_data=list(map(float,input_data.split(',')))
            xgb_prediction=predict(xgb,input_data)
            random_prediction=predict(random,input_data)
            return JsonResponse({
                'xgb_prediction':xgb_prediction.tolist(),
                'random_prediction':random_prediction.tolist()
            })
    return render(request,'predict.html')
# Create your views here.
