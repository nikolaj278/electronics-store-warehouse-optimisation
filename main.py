import pandas as pd
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA

from src.preprocessing import preporocess, take_n


def main():
    print("\n --- Data preprocessing begins--- ")
    df = pd.read_csv("data/kz.csv")
    df = preporocess(df)
    print("\n--- Data preprocessing ends ---")
    # take 100 best selling products
    prod_idx, prod_list, nr_orders_week = take_n(df, 100)
    
    # restrucrute multiindex series to a dataframe for model fit
    df_for_fit = pd.DataFrame({
        "unique_id": nr_orders_week.index.get_level_values(0),
        "ds": nr_orders_week.index.get_level_values(1),
        "y": nr_orders_week.values
    })
    
    print("\n --- Model fitting begins --- ")
    
    # fit ARIMA for every series with AutoARIMA 
    sf = StatsForecast(
        models=[
            AutoARIMA(
                season_length=13,
                start_p=0,
                start_q=0,
                max_d=2
            )
        ],
        freq="W",
        n_jobs=-1
    )
    
    sf.fit(df_for_fit)
    
    print("\n --- Model fitting ends ---")
    
    
    
    print("\n --- Forescting begins --- ")
    forecast = sf.predict(1)
    forecast.to_csv("forecast.csv")
    print("\n --- Forescting ends --- ")


if __name__ == "__main__":
    main()