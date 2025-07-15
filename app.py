import dash
from dash import html, dcc
import pandas as pd
import plotly.graph_objs as go
from prophet import Prophet
import requests

# Scarica dati da CoinMetrics: market cap, realized cap, prezzo
def fetch_coinmetrics():
    url = "https://api.coinmetrics.io/v4/timeseries/asset-metrics"
    params = {
        "assets": "btc",
        "metrics": "CapMrktCurUSD,CapRealUSD,PriceUSD,AdrActCnt,TxCnt,TxTfrValAdjUSD,DiffMean",
        "frequency": "1d",
        "start": "2021-01-01",
        "format": "csv"
    }
    r = requests.get(url, params=params)
    from io import StringIO
df = pd.read_csv(StringIO(r.text))
    df['date'] = pd.to_datetime(df['time'], utc=True)
    df['mvrv'] = df['CapMrktCurUSD'] / df['CapRealUSD']
    df = df.rename(columns={
        "PriceUSD": "price",
        "AdrActCnt": "active_addresses",
        "TxCnt": "tx_count",
        "TxTfrValAdjUSD": "tx_volume",
        "DiffMean": "difficulty"
    })
    return df[['date', 'price', 'mvrv', 'active_addresses', 'tx_count', 'tx_volume', 'difficulty']].dropna()

# Dati reali
df = fetch_coinmetrics()

# Previsione con Prophet
def forecast_column(df, col_name):
    prophet_df = df[["date", col_name]].rename(columns={"date": "ds", col_name: "y"})
    model = Prophet()
    model.fit(prophet_df)
    future = model.make_future_dataframe(periods=180)
    forecast = model.predict(future)
    return forecast

forecast_mvrv = forecast_column(df, "mvrv")
forecast_price = forecast_column(df, "price")

# Dash layout
app = dash.Dash(__name__)
app.layout = html.Div([
    html.H1("Crypto Macro Dashboard (BTC) – Dati CoinMetrics"),

    dcc.Graph(figure={
        "data": [
            go.Scatter(x=df["date"], y=df["mvrv"], name="MVRV"),
            go.Scatter(x=df["date"], y=df["price"], name="Prezzo BTC", yaxis="y2")
        ],
        "layout": go.Layout(
            title="MVRV vs Prezzo",
            yaxis=dict(title="MVRV"),
            yaxis2=dict(title="Prezzo", overlaying="y", side="right"),
            xaxis=dict(title="Data")
        )
    }),

    dcc.Graph(figure={
        "data": [
            go.Scatter(x=df["date"], y=df["active_addresses"], name="Indirizzi attivi"),
            go.Scatter(x=df["date"], y=df["tx_count"], name="Transazioni")
        ],
        "layout": go.Layout(title="Attività On-Chain", xaxis=dict(title="Data"))
    }),

    dcc.Graph(figure={
        "data": [
            go.Scatter(x=df["date"], y=df["tx_volume"], name="Volume trasferito (USD)"),
            go.Scatter(x=df["date"], y=df["difficulty"], name="Difficoltà Mining", yaxis="y2")
        ],
        "layout": go.Layout(
            title="Volume vs Difficoltà Mining",
            yaxis=dict(title="Volume"),
            yaxis2=dict(title="Difficoltà", overlaying="y", side="right"),
            xaxis=dict(title="Data")
        )
    }),

    dcc.Graph(figure={
        "data": [
            go.Scatter(x=forecast_mvrv["ds"], y=forecast_mvrv["yhat"], name="MVRV previsto"),
            go.Scatter(x=forecast_price["ds"], y=forecast_price["yhat"], name="Prezzo previsto BTC")
        ],
        "layout": go.Layout(title="Previsioni 6 mesi (MVRV & Prezzo BTC)")
    })
])

server = app.server
