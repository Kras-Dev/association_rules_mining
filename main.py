from association_miner.candle_miner import CandleMiner
from mt5_connector.client import MT5Client
import MetaTrader5 as mt5

SYMBOL = "EURUSDrfd"
TIMEFRAME = mt5.TIMEFRAME_H1

def main():
    with MT5Client() as client:
        df = client.get_rates(SYMBOL, TIMEFRAME, 37600, 1)
        miner = CandleMiner(min_confidence=0.65, verbose=True, min_support=10)
        results = miner.analyze(df, SYMBOL, client.mt5_var_name(TIMEFRAME))
        miner.print_top_rules(results, top_n=80, symbol=SYMBOL,timeframe=client.mt5_var_name(TIMEFRAME))


if __name__ == "__main__":
    main()
