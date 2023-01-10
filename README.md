# About

Exponentially Weighted Moving Average (EWMA) method is used to forecast the future daily volatility for the next trading day. 

Using the forecasted volatility, "Volatility Risk Premium (VRP)" is also estimated. There are several ways by which VRP can be defined. However, in thos code, VRP is defined is the ratio of the forecasted volatility to the at-the-money implied volatility. VRP is plotted along with its 10th, 25th, 50th, 75th, and 90th percentile values historically.

Important Note: VRP in itself is NOT a sufficient metric to finalise trade decisions.
