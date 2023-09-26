import yfinance as yf
import sklearn
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import precision_score
from sklearn.ensemble import RandomForestClassifier # randomize parameters
TSLA = yf.Ticker("^GSPC") 
TSLA = TSLA.history(period = "max")
TSLA.index
TSLA.plot.line(y = "Open", use_index = True)
TSLA.plot.line(y = "Close", use_index = True)
del TSLA["Dividends"]
del TSLA["Stock Splits"]
TSLA["Tomorrow"] = TSLA["Close"].shift(-1)
TSLA["Target"] = (TSLA["Tomorrow"] > TSLA["Close"]).astype(int)
TSLA = TSLA.loc["1990-01-01":].copy()
model = RandomForestClassifier(n_estimators = 100, min_samples_split = 100, random_state = 1)

train = TSLA.iloc[:-100]
test = TSLA.iloc[-100:]

predictors = ["Close", "Volume", "Open", "High", "Low"]
model.fit(train[predictors], train["Target"])
preds = model.predict(test[predictors])
preds = pd.Series(preds, index = test.index)
precision_score(test["Target"], preds)
combined = pd.concat([test["Target"], preds], axis = 1)
combined.plot()

def predict(train, test, predictors, model):
     model.fit(train[predictors], train["Target"])
     preds = model.predict(test[predictors])
     preds = pd.Series(preds, index = test.index, name = "Predictions")
     combined = pd.concat([test["Target"], preds], axis = 1)
     return combined

def backtest(data, model, predictors, start = 4000, step = 1):
        all_predictions = []

        for i in range(start, data.shape[0], step):
               train = data.iloc[0:i].copy()
               test = data.iloc[i:(i+step)].copy()
               predictions = predict(train, test, predictors, model)
               all_predictions.append(predictions)
        return pd.concat(all_predictions)

predictions = backtest(TSLA, model, predictors)

#print(predictions["Predictions"].value_counts())
#print(precision_score(predictions["Target"], predictions["Predictions"]))

horizons = [2, 5, 60, 250, 1000]

new_predictors = []
for horizon in horizons:
        rolling_averages = TSLA.rolling(horizon).mean()
        ratio_column = f'Close_Ratio_{horizon}'
        TSLA[ratio_column] = TSLA["Close"] / rolling_averages["Close"]
        trend_column = f'Trend_{horizon}'
        TSLA[trend_column] = TSLA.shift(1).rolling(horizon).sum()["Target"]
        new_predictors += [ratio_column, trend_column]

print(TSLA)

TSLA = TSLA.dropna()
model = RandomForestClassifier(n_estimators = 200, min_samples_split = 50, random_state = 1)
def predict(train, test, predictors, model):
     model.fit(train[predictors], train["Target"])
     preds = model.predict_proba(test[predictors])[:,1]
     preds[preds >= .6] = 1
     preds[preds < .6] = 0
     preds = pd.Series(preds, index = test.index, name = "Predictions")
     combined = pd.concat([test["Target"], preds], axis = 1)
     return combined

print(predictions["Predictions"].value_counts())

predictions = backtest(TSLA, model, new_predictors)

print(precision_score(predictions["Target"], predictions["Predictions"]))

TSLA.plot.line(y = "Trend_1000", use_index = True)

plt.show()