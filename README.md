# QuantLab: A Research Environment for Quantitative Trading Strategies (WIP)

### This project is very early in its development (I am literally the only developer right now) so please be patient as I build out the functionality.
### I plan to make this open source and welcome contributions from the community.
### I will spend time writing documentation and tests as I build out the functionality. Right now I am focused on getting the core functionality working.
### If you have any questions or suggestions, please feel free to open an issue or contact me directly at _riddhi dot mandal at mail dot utoronto dot ca_

QuantLab is an open-source Python project designed to facilitate the development, testing, and analysis of quantitative trading strategies. It provides a structured environment for loading market data, applying technical indicators, and visualizing results through interactive charts.

For now I have implemented a very basic shell that allows you to load data, add indicators, and plot charts. chart.py also has a demo for how to use the functions directly. 

To run the cli, just run `python quantlab.py` from the project root.

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

