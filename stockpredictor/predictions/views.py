import io
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import base64
from datetime import date, timedelta
from django.shortcuts import render
from django.http import HttpResponse
from keras.models import Sequential
from keras.layers import Dense, LSTM
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

model_dict = {}

def create_model():
    model = Sequential()
    model.add(LSTM(128, return_sequences=True, input_shape=(3, 1)))
    model.add(LSTM(64, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def load_data(company_symbol):
    today = date.today()
    end_date = today.strftime("%Y-%m-%d")
    start_date = (today - timedelta(days=5000)).strftime("%Y-%m-%d")
    data = yf.download(company_symbol, start=start_date, end=end_date, progress=False)
    data["Date"] = data.index
    data = data[["Date", "Open", "High", "Low", "Close"]]
    data.reset_index(drop=True, inplace=True)
    x = data[["Open", "High", "Low"]].values
    y = data["Close"].values
    y = y.reshape(-1, 1)
    xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=42)
    model = create_model()
    xtrain = xtrain.reshape((xtrain.shape[0], xtrain.shape[1], 1))
    xtest = xtest.reshape((xtest.shape[0], xtest.shape[1], 1))
    model.fit(xtrain, ytrain, batch_size=1, epochs=10)
    return model, xtest, ytest

def plot_predictions(ytest, ypred):
    plt.figure(figsize=(14, 5))
    plt.plot(ytest, color='blue', label='Actual Stock Price')
    plt.plot(ypred, color='red', label='Predicted Stock Price')
    plt.title('Stock Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.legend()

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    return buf

def predict_stock(request):
    if request.method == 'POST':
        company_symbol = request.POST['company']
        open_price = float(request.POST['open_price'])
        high_price = float(request.POST['high_price'])
        low_price = float(request.POST['low_price'])

        if company_symbol not in model_dict:
            model, xtest, ytest = load_data(company_symbol)
            model_dict[company_symbol] = (model, xtest, ytest)
        else:
            model, xtest, ytest = model_dict[company_symbol]

        features = np.array([[open_price, high_price, low_price]])
        features = features.reshape((features.shape[0], features.shape[1], 1))
        predicted_close = model.predict(features)
        
 
        ypred = model.predict(xtest[-100:])
        ytest_last_100 = ytest[-100:]
        buf = plot_predictions(ytest_last_100, ypred)
     
        plot_image = base64.b64encode(buf.getvalue()).decode('utf-8')
 
        mse = mean_squared_error(ytest_last_100, ypred)
        r2 = r2_score(ytest_last_100, ypred)

     
        request.session['actual_values'] = ytest_last_100.tolist()
        request.session['predicted_values'] = ypred.tolist()
        request.session['mse'] = mse
        request.session['r2'] = r2

        return render(request, 'predictions/result.html', {
            'company_symbol': company_symbol,
            'open_price': open_price,
            'high_price': high_price,
            'low_price': low_price,
            'predicted_close': predicted_close[0][0],
            'plot_image': plot_image, 
            'mse': mse,
            'r2': r2
        })
    return render(request, 'predictions/index.html')

def prediction_values(request):
    actual_values = request.session.get('actual_values', [])
    predicted_values = request.session.get('predicted_values', [])
    mse = request.session.get('mse', 0)
    r2 = request.session.get('r2', 0)
    
    values = list(zip(actual_values, predicted_values))
    
    return render(request, 'predictions/values.html', {
        'values': values,
        'mse': mse,
        'r2': r2
    })
