# Autocorrelation plot script. An autocorrelation plot is used to spot randomness, it shows the autocorrelations
# of the data on the model depending on the number of lags. This randomness is ascertained by
# computing autocorrelations for data values at varying time lags. If random, such autocorrelations should be near zero
# for any and all time-lag separations. If non-random, then one or more of the autocorrelations will
# be significantly non-zero.
# After obtaining the autocorrelation plot, the lag number when the positive correlation ends is a good canditate to
# be the AR parameter on the model. Autocorrelations that fall within the band marked by the dotted blue lines
# are deemed not to be signifficantly different from 0.
import pandas as pd
from matplotlib import pyplot
from pandas.plotting import autocorrelation_plot

data = pd.read_excel('Prices.xlsx', sheet_name='Prices', nrows=200)
prices = list(data['Price'][0:25*7])

autocorrelation_plot(prices)
pyplot.show()