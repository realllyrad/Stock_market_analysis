# Stock_market_analysis
Analysis of Stock Market Data with Conventional Machine Learning Models and Deep Learning Models

Project Report: Analysis of Stock Market Data with Conventional Machine Learning Models and Deep Learning Models

1. Introduction
The objective of this project was to analyse stock market data using both conventional machine learning models and deep learning models. The dataset used was the NIFTY 50 stock market index at NSE, obtained from Kaggle. The primary focus was on predicting the open prices using various parameters such as 'Prev Close', 'Open', 'High', 'Low', 'Last', 'Close', and 'VWAP'. Additionally, the project aimed to understand the stock market, study the significance of different parameters, compare companies over the years, observe seasonality, cyclicity, and trends.

2. Data Preprocessing and Exploratory Data Analysis (EDA)
We perform EDA on the data for the company ADANIPORTS. The first step involved cleaning the stock market data by removing null rows and columns. On checking we see the column trades have 866 null rows. We then decide to eliminate the column trades from the data. We proceed to analyse each column and its significance. 

Close: The closing price of a stock represents the final price at which a stock was traded on a particular day. It is typically the last traded price before the market closes for the day. The close price is widely used by investors and analysts to assess the performance of a stock over a specific time period.

Open: The opening price of a stock signifies the price at which the first trade occurred for the day. It is the initial price at the beginning of the trading session. The opening price is an important indicator as it can provide insights into the market sentiment and investor demand at the start of the trading day.

High: The high price indicates the highest trading price reached by a stock during a particular trading day. It reflects the maximum price at which the stock was bought or sold within that time frame. The high price is useful for identifying the intraday peak or resistance levels of a stock.

Low: The low price represents the lowest trading price reached by a stock during a specific trading day. It indicates the minimum price at which the stock was traded during that time period. The low price is valuable for identifying the intraday troughs or support levels of a stock.

Date: The date parameter signifies the specific day to which the stock market data corresponds. It is important for tracking the historical performance of a stock over time and conducting analysis or comparisons across different dates.

Symbol: The symbol refers to the unique ticker symbol assigned to a particular stock. Ticker symbols are used to identify and differentiate stocks listed on stock exchanges. For example, the symbol for Apple Inc. is "AAPL."

Series: The series parameter is used in some stock markets to categorize different types of securities or market segments. It can indicate whether a stock belongs to the equity segment, debt segment, or any other specific classification. The series parameter is more commonly used in countries like India.

Last: The last price represents the latest traded price of a stock during a trading session. It provides information on the most recent price at which the stock was bought or sold.

VWAP (Volume Weighted Average Price): VWAP is a trading indicator that calculates the average price at which a stock has been traded throughout the day, weighted by the volume of each trade. It is calculated by dividing the total value of all trades by the total volume traded. VWAP is commonly used by institutional investors and algorithmic traders to assess the average price at which they executed their trades relative to the overall market.

Volume: The volume parameter represents the total number of shares or contracts traded for a particular stock on a given day. It provides insights into the level of market activity and liquidity for a stock. Higher trading volumes generally indicate increased investor interest and can suggest the presence of significant market moves.

Turnover: The turnover parameter signifies the total value of all trades conducted for a stock on a specific day. It is calculated by multiplying the volume of shares traded by the corresponding price for each trade. Turnover is a useful metric to assess the overall market activity and the monetary value associated with the trading of a stock.

Trades: The trades parameter refers to the total number of individual trades executed for a stock on a given day. It represents the number of times the stock has changed hands during the trading session. The number of trades can provide insights into the level of market participation and the intensity of buying or selling pressure.

Deliverable Volume: Deliverable volume represents the number of shares or contracts that were actually delivered or transferred between buyers and sellers at the end of a trading day. It indicates the portion of the total trading volume that resulted in the actual exchange of ownership. 

We decided to eliminate Series from the data as the information was redundant. 
EDA techniques were applied to gain insights into the dataset. We plotted the line plot of ADANI PORTS closing prices with time. We see that there is no visible trend, with fluctuations over the years, however a sharp decreasing trend is seen initially, followed by a gradual increase. We can see seasonality but no cyclicity over the years.


The correlation between different parameters was analysed using a heatmap. We notice that the parameters 'Prev Close', 'Open', 'High', 'Low', 'Last', 'Close' and 'VWAP' are highly correlated while the rest are mostly independent. We see that Volume, turnover and deliverable volume are weakly correlated, which could have been undertsood with the defination
We compare the closing prices for 3 companies. We notice no sharp trend but seasonal fluctuations in all 3 companies. Some companies showed cyclicity while others didnt.

3. Conventional Machine Learning Models
3.1 ARIMA
ARIMA (Autoregressive Integrated Moving Average) was employed for time series forecasting. It was applied on the parameters ‘Open’ and ‘Date’. The final performance was studied using MAPE(Mean Absolute Percentage Error). The model parameters (p, d, q) were determined using ACF (Autocorrelation Function) and PACF (Partial Autocorrelation Function).



We can conclude that the AR term must be 1 and MA terms must be zero, the values for ACF at lag 1 are positive and PACF drops suddenly.
 Differencing was performed to obtain the value of d. We can see that the differenced values are stagnant. We can conclude that I should be 1

The model was implemented after re-indexing the data and splitting 60% for training. ARIMA yielded a mean absolute percentage error (MAPE) of 12.27%, outperforming XGBoost and Linear Regression. However, it was applied to only one dataset as different datasets may have different parameter values (p, d, q), hence wasn’t applied on other companies together. Performance could be further improved by using multivariable ARIMA.

3.2 Linear Regression
Further linear regression was implemented as the machine learning model. We used parameters 'Prev Close', 'Open', 'High', 'Low', 'Last', 'Close' and 'VWAP', and all parameters were standardised using a standard scaler. Our target variable was taken to be ‘Open’. We divided the data into test and train using TimeSeriesSplit using the number of splits to be 5. Finally we evaluated the MAPE.It achieved an accuracy of 12.98% over Adani Ports and an average accuracy of 8.21% over all files.

3.2 XGBoost
Next, XGBoost was implemented with 1000 estimators. We used parameters 'Prev Close', 'Open', 'High', 'Low', 'Last', 'Close' and 'VWAP'. Our target variable was taken to be ‘Open’. We divided the data into test and train using TimeSeriesSplit using the number of splits to be 5. Finally we evaluated the MAPE. It achieved an accuracy of 13.15% over Adani Ports. The average accuracy over all 50 files was 10.07%

3.4 Concatenated Data
Linear regression and XGBoost was performed on concatenated data. We used parameters 'Prev Close', 'Open', 'High', 'Low', 'Last', 'Close' and 'VWAP', and all parameters were standardised using a standard scaler. Our target variable was taken to be ‘Open’. We divided it into training and test sets such that each symbol had 80% of data in the test set and the remaining 20% in the training set. One Hot Encoder was used for the symbol column. Surprisingly good results were obtained with a MAPE of 2.74% with linear regression and 3.37% with XGBoost This improvement may be attributed to the difference in the size of data used for training the model.

4. Deep Learning Models
4.1 Basic Neural Network (DNN)
A basic neural network model was trained on 90% of the data, resulting in an improved MAPE of 1.40% and a mean absolute error (MAE) of 8.53%. The DNN model had a single layered neural network with 100 epochs.

4.2 RNNs, LSTMs, Attention, and Transformers
Sequential models such as RNNs (Recurrent Neural Networks) and LSTMs (Long Short-Term Memory) were studied to capture dependencies over time. Attention mechanisms were introduced to enable the model to focus on different parts of the input sequence. Transformers, which heavily utilise attention, became the preferred architecture for various NLP tasks.

5. Temporal Fusion Transformers (TFT)
TFT, a specific DNN architecture designed for time series forecasting, was considered for further study. TFT incorporates static covariate encoders, gating mechanisms, a sequence-to-sequence layer, and a temporal self-attention decoder. It achieves high performance and offers interpretability, enabling identification of globally-important variables, persistent temporal patterns, and significant events. The implementation of TFT was pending at the time of writing this report.

6. Conclusion and Future Work
In conclusion, this project analysed stock market data using both conventional machine learning models and deep learning models. Various models such as linear regression, XGBoost, ARIMA, and neural networks were implemented and evaluated for accuracy. Deep learning models, including RNNs, LSTMs, attention mechanisms, and transformers, were studied. The project identified areas for further improvement, including feature engineering, modelling using Shapley values, and the implementation of TFT.
