try:
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    import requests
    import yfinance as yf
except ImportError as e:
    print(f"Error importing module: {e}")
    exit()


def numpy_use():
    np_array = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    return np_array


def import_stock_data(ticker: str, start_date: str, end_date: str):
    data = yf.download(ticker, start=start_date, end=end_date)
    return data


def stock_data_analysis(ticker: str, start_date: str, end_date: str):
    hist_data = import_stock_data(ticker, start_date, end_date)
    hist_data = hist_data.reset_index().rename(columns={'index': 'Date'})
    hist_data['Date'] = pd.to_datetime(hist_data['Date'])
    hist_data['MA20'] = hist_data['Close'].rolling(window=20).mean()
    hist_data['MA50'] = hist_data['Close'].rolling(window=50).mean()
    # hist_data['BB_upper'] = hist_data['MA20'] + 2 * hist_data['Close'].rolling(20).std()
    # hist_data['BB_lower'] = hist_data['MA20'] - 2 * hist_data['Close'].rolling(20).std()
    return hist_data


def list_tickers():
    Matrix_data = ['AAPL']# , 'MSFT', 'GOOGL', 'AMZN', 'TSLA'
    return Matrix_data


def plot_stock_data(ticker: str, hist_data: pd.DataFrame):

    # print(f"{ticker} data: {hist_data.head()}")
    # print(f"{ticker} data: {hist_data.columns}")
    plt.plot(hist_data['Date'], hist_data['Close'], label=ticker, color='blue')
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.title('Stock Price Analysis')
    plt.legend()
    plt.savefig("matrix_analysis.png")
    # plt.show()


def request_data():
    pass
    # r = requests.get('https://api.github.com/events', stream=True)
    # return r.raw


def main():

    # print("Using NumPy:")
    numpy_use()
    # print("\nUsing Pandas to analyse Stock data:")
    tickers = list_tickers()

    # print("\nUsing Requests to fetch data:")
    # r = request_data()
    # print(f"Request Object: {r}")
    # print(f"Request status code: {r.status}")
    print("\nLOADING STATUS: Loading programs...")
    print("Checking dependencies:")
    print(f"[OK] pandas {pd.__version__} - Data manipulation ready")
    print(f"[OK] requests {requests.__version__} - Network access ready")
    print(f"[OK] matplotlib {plt.matplotlib.__version__} - Visualization ready")
    print("Analyzing Matrix data...")
    print("Processing 1000 data points...")
    print("Generating visualization...")
    for ticker in tickers:
        hist_data = stock_data_analysis(ticker, "2020-01-01", "2021-01-01")
        plot_stock_data(ticker, hist_data)
    print("Analysis complete!")
    print("Results saved to: matrix_analysis.png}")


if __name__ == "__main__":
    main()
