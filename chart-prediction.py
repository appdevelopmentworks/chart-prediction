import datetime
import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np

#sklearn
import sklearn.linear_model
import sklearn.model_selection
#streamlit
import streamlit as st


st.title("株価の予測")
st.text("入力例）^N225, AAPL, 7203.T, JPY=X")
tickerCode = st.text_input('ティッカー:',"^N225")
period = st.slider("予想期間（日）:",min_value=1,max_value=20, value=5)
display_period = st.slider("表示期間（日）:",min_value=30,max_value=300,value=300)
srcBtn = st.button('検索')

def stockPred(ticker, period):
    #株価データ取得
    dfstock = yf.download(ticker)
    #Label列を追加しAdj Close列を30行上にシフトしてコピー
    dfstock['label'] = dfstock['Adj Close'].shift(-period)
    #label列を削除してnumpy配列に変換
    X = np.array(dfstock.drop(['label'], axis='columns'))
    #異常値をはじく
    X = sklearn.preprocessing.scale(X)
    #Label列をnumpy配列に
    y = np.array(dfstock['label'])
    #学習用データとして
    y = y[:-period]
    X = X[:-period]

    #Train（80％）データとtest（20％）データを仕分ける
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.2)
    #線形モデルを選択
    lr = sklearn.linear_model.LinearRegression()
    #学習
    lr.fit(X_train, y_train)
    #予測精度を求める
    accuracy = lr.score(X_test, y_test)
    #予測データ配列を変数に代入
    predicted_data = lr.predict(X[-period:])

    #予想列をNANで初期化
    dfstock['Predict'] = np.nan
    #実データ最終日を取得
    last_date = dfstock.iloc[-1].name
    #1日分を計算
    one_day_td = datetime.timedelta(days=1)
    #データの最終日に１日足して翌日を求める
    next_day = last_date + one_day_td
    #翌日を予測データの日付に
    next_date = next_day

    #予測データの配列をループ
    for data in predicted_data:
        #日付行（index）から最後の列に予測データ代入
        dfstock.loc[next_date] = np.append([np.nan] * (len(dfstock.columns)-1) ,data)
        #次の日付
        next_date += one_day_td
    #予測精度と予測データ入りデータセットを返す
    return accuracy, dfstock


def graph_drawing(data_n, ds):
    #表示日数分のデータセットに代入
    dfDisp = ds.tail(data_n)
    #折れ線グラフで指定
    #グラフ表示
    fig, ax = plt.subplots()
    #タイトルとフォントサイズ設定
    plt.title(tickerCode)
    plt.rcParams['font.size'] = 7
    #グリッド表示
    ax.grid()
    #実データと予測値をそれぞれプロット
    plt.plot(dfDisp.index, dfDisp["Adj Close"],label="RealData")
    plt.plot(dfDisp.index, dfDisp["Predict"],label='Predict')
    #凡例表示
    plt.legend()
    #streamlitでチャート表示
    st.pyplot(fig)

if srcBtn:
    ac, ds = stockPred(tickerCode, period)
    #予測精度と明日の予測値
    st.text("精度：" + str(format(ac,'2f')))
    st.text("明日の終値：" + str(round(ds["Predict"][-period], 2)))
    
    #グラフ描画
    graph_drawing(display_period,ds)
    #予測データセット
    #st.dataframe(ds["Predict"].tail(period))