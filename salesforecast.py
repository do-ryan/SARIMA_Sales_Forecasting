import plotly.plotly as py
import plotly
import plotly.graph_objs as go
import numpy as np
from StringIO import StringIO
import pandas as pd
import scipy as sp
plotly.tools.set_credentials_file(username='doryan', api_key='YQY9JEivlbDJFQBlIG4k')
from statsmodels.tsa.statespace.sarimax import SARIMAX

def scatter(data1, data2, label, filename):

    trace1 = go.Scatter(
        x=label,
        y=data1[:-3],
        mode='lines+markers',
        name='Historic sales',
        marker = dict(
            color = 'green'
        )
    )

    trace2 = go.Scatter(
        x=label,
        y=data2,
        mode='lines',
        name='Sales budget',
        line = dict(
            width = 1.5
        ),
        marker = dict(
            color='orange'
        )
    )

    trace3 = go.Scatter(
        x=label[-4:],
        y=data1[-4:],
        mode = 'lines+markers',
        name = 'Forecasted sales',
        marker = dict(
            color = 'red'
        )
    )

    layout = go.Layout(
        autosize=False,
        width=1100,
        height=700,
        title = dict(text="Laptop Sales Forecast for 2019 Q1", font=dict(size=36)),
        xaxis = dict(title = "Monthly"),
        yaxis = dict(title = "Sales per month ($)"),
    )

    fig = go.Figure(data=[trace1, trace2, trace3], layout=layout)

    py.plot(fig, filename=filename)

def barplot(x, y1, y2, filename):
    trace1 = go.Bar(
        x=x,
        y=y1,
        name='Forecasted Sales',
        marker = dict(
            color='red'
        )
    )
    trace2 = go.Bar(
        x=x,
        y=y2,
        name=' Sales Budget',
        marker = dict(
            color='orange'
        )
    )

    layout = go.Layout(
        barmode='group',
        width = 1100,
        height = 700,
        title = dict(text="Laptop Sales Forecast vs Budget For 2019 Q1", font = dict(size=36)),
        xaxis = dict(title="Month"),
        yaxis = dict(title="Sales ($)"),
    )

    fig = go.Figure(data=[trace1, trace2], layout=layout)

    py.plot(fig, filename=filename)

def main():
    sales_budget_data = np.genfromtxt("monthlysales.csv", delimiter=",")[1:] # omit header
    sales_budget_data = sales_budget_data[:, ~np.all(np.isnan(sales_budget_data), axis=0)] # eliminate NaN columns

    sales_budget_labels = np.genfromtxt("monthlysales.csv", delimiter=',', usecols=0, dtype=str)[1:sales_budget_data.shape[0]-8]

    sales = np.transpose(sales_budget_data)[0]
    sales = np.expand_dims(sales[~np.isnan(sales)],axis=1) # result in historic sales
    budget = np.transpose(sales_budget_data)[1] # all budget
    # scatter(sales, budget, sales_budget_labels, "historical_data") # plot original data

    model = SARIMAX(sales, order=(3,1,3), seasonal_order=(1,0,0,3), enforce_stationarity=False, enforce_invertibility=False)
    model_fit = model.fit(disp=False)
    K = 3
    forecast = np.expand_dims(model_fit.forecast(K), axis=1)

    sales = np.append(sales, forecast, axis=0)
    scatter(sales, budget, sales_budget_labels, "forecast")

    barplot(sales_budget_labels[-3:], forecast, budget[-12:-9], "forecast_budget_comparison")

    return

if __name__ == "__main__":
    main()