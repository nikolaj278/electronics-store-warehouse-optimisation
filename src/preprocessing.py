import pandas as pd
import numpy as np

def preporocess(df):
    df["event_time"] = pd.to_datetime(df["event_time"], format="%Y-%m-%d %H:%M:%S UTC")
    df = df.drop("user_id", axis=1)
    mask_1 = ~ df.category_code.isna() & df.category_id.isna()
    df.loc[mask_1, "price"] = df[mask_1].category_code.astype(float)
    
    #drop category_code as it has no more value now
    df = df.drop("category_code", axis=1)
    df["category_id"] = df["category_id"].fillna(0).astype(int)

    
    df["na_brand"] = df.brand.isna()
    prod_na_rate = df.groupby("product_id").na_brand.mean() 
    df = df.drop("na_brand", axis=1)
    # find and fill missing brand values which can be restored
    prod_na_cat = np.array(prod_na_rate[(prod_na_rate > 0) & (prod_na_rate < 1)].index)
    mask_2 = df.product_id.isin(prod_na_cat)
    df_1 = df.loc[mask_2, ["product_id", "brand"]]
    # order df in a way NaN value will be below non-missing so 
    # NaN's could be filled with ffill.
    df_1 = df_1.sort_values(["product_id", "brand"]).ffill()
    brand_idx = np.where(df.columns == "brand")[0][0]
    df.iloc[df_1.index, brand_idx] = df_1.brand
    df["brand"] = df["brand"].fillna("unknown")
    
    df_year = df.event_time.dt.year
    idx_1970 = df[df_year == 1970].index
    df.loc[idx_1970, "event_time"] = np.nan
    df["event_time"] = df["event_time"].ffill()

    df["event_week"] = df.event_time.dt.to_period("W")

    return df


def take_n(df, n):
    nr_orders = df.groupby(["product_id"])["order_id"].count()
    nr_orders = nr_orders.sort_values(ascending=False)
    nr_orders = nr_orders.iloc[:n]
    
    df_100 = df[df["product_id"].isin(nr_orders.index)].copy()
    # paņemam tikai datumu nedēļas perioda beigās
    df_100["period_end"] = df_100.event_week.dt.end_time.dt.date
    # saskaitām pasūtījumu skaitu katram produktam katru nedēļu
    nr_orders_week = df_100.groupby(["product_id", "period_end"]).order_id.count()
    # sadalām datframe vairākos dataframes, kur katrā ir dati tikai par vienu produktu
    prod_idx = nr_orders_week.index.get_level_values(0).unique()
    prod_list = {idx: nr_orders_week.loc[idx] for idx in prod_idx}
    # change index label
    for idx, it in prod_list.items():
        it.index.name = "date"
        it.name = "order_nr"

    return prod_idx, prod_list, nr_orders_week 
