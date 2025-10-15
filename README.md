# QuantLab: A Research Environment for Quantitative Trading Strategies

QuantLab is an open-source Python project designed to facilitate the development, testing, and analysis of quantitative trading strategies. It provides a structured environment for loading market data, applying technical indicators, and visualizing results through interactive charts.

For now I have implemented a very basic shell that allows you to load data, add indicators, and plot charts. chart.py also has a demo for how to use the functions directly. 

To run the cli, just run `python quantlab.py` from the project root. To exit press CTRL+C.

A cli example is as follows:

```shell
    quantlab$ mode list
    quantlab$ mc
    quantlab$ tAAPL
    quantlab$ i sma:20
    quantlab$ i ema:50
    quantlab$ output --chart "aapl_candles_with_indicators.html"
    quantlab$ output --dataframe "aapl_data_with_indicators.csv"
    quantlab$ exit
```

