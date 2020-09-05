# Rolling forecast aplying an ARIMA model
import pandas as pd
from statsmodels.tsa.arima_model import ARIMA
from matplotlib import pyplot
from sklearn.metrics import mean_squared_error

data = pd.read_excel('Prices.xlsx', sheet_name='Prices', nrows=200)
prices = list(data['Price'][0:25*7])

size = int(len(prices) * 0.66)
train, test = prices[0:size], prices[size:len(prices)]
history = [x for x in train]
predictions = list()
for t in range(len(test)):
	model = ARIMA(history, order=(3,0,0))
	model_fit = model.fit(disp=0)
	output = model_fit.forecast()
	yhat = output[0]
	predictions.append(yhat)
	obs = test[t]
	history.append(obs)
	print('predicted=%f, expected=%f' % (yhat, obs))
error = mean_squared_error(test, predictions)
print('Test MSE: %.3f' % error)
# plot
pyplot.plot(test, label='Real')
pyplot.plot(predictions, color='red', label='Prediction')
pyplot.legend()
pyplot.xlabel("Time (h)")
pyplot.ylabel("Price (€/MWh)")
pyplot.show()