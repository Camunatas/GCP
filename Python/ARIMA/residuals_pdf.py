# Script to obtain the ARMA fit residual error line plot and the pdf function for the ARIMA model
import pandas as pd
from statsmodels.tsa.arima_model import ARIMA
from matplotlib import pyplot

data = pd.read_excel('Prices.xlsx', sheet_name='Prices', nrows=200)
prices = list(data['Price'][0:25*7])

# fit model
model = ARIMA(prices, order=(3,0,0))
model_fit = model.fit(disp=0)

model_fit.cov_params
print(model_fit.summary())
# plot residual errors
residuals = pd.DataFrame(model_fit.resid)
residuals.plot()
pyplot.show()
residuals.plot(kind='kde')
pyplot.show()
print(residuals.describe())