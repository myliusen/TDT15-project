import pandas as pd
import numpy as np


def load_yield_data(filepath=r'data/LW_daily.csv') -> pd.DataFrame:
    """Load and preprocess Liu & Wu yield data from CSV file"""
    yield_data = pd.read_csv(filepath, comment='%', index_col=0)
    # rename columns to 1m, 2m, 3m, ..., 360m
    yield_data.columns = [f"{i}m" for i in range(1, 361)]
    # parse dates from 19650101 format to datetime
    yield_data.index = pd.to_datetime(yield_data.index, format='%Y%m%d')
    # Remove all maturities greater than 120 months
    yield_data = yield_data.loc[:, yield_data.columns.str.endswith(('m')) & (yield_data.columns.str[:-1].astype(int) <= 120)]
    # Finally, drop rows with no 10-year yield data
    yield_data = yield_data.dropna(subset=['120m'])
    return yield_data


def apply_transformation(series:pd.Series, tcode:int) -> pd.Series:
    """Apply FRED-MD transformation codes to a series"""
    if series.isnull().all():
        return series  # skip empty series
    try:
        if tcode == 1:
            return series
        elif tcode == 2:
            return series.diff()
        elif tcode == 3:
            return series.diff().diff()
        elif tcode == 4:
            return np.log(series)
        elif tcode == 5:
            return np.log(series).diff()
        elif tcode == 6:
            return np.log(series).diff().diff()
        elif tcode == 7:
            return ((series / series.shift(1)) - 1.0).diff()
        else:
            raise ValueError(f"Unknown tcode: {tcode}")
    except Exception as e:
        print(f"Error transforming series {series.name}: {e}")
        return pd.Series([np.nan] * len(series))


def load_fred_md(start:pd.Timestamp, end:pd.Timestamp, filepath=r'data\2025-09-MD.csv', 
                 mapping_filepath=r'data\MD-mapping.csv') -> pd.DataFrame:
    """Load and transform FRED-MD data according to transformation codes"""
    # Read transformation codes from first row
    transform_row = pd.read_csv(filepath, nrows=1).iloc[0, 1:]  # skip 'sasdate'
    tcodes = transform_row.astype("float").ffill().bfill().astype("int").to_dict()
    
    # Load CSV, skipping the transformation row
    df = pd.read_csv(filepath, parse_dates=['sasdate'], index_col='sasdate', skiprows=[1])
    
    # Load category mapping
    mapping_df = pd.read_csv(mapping_filepath)
    category_map = dict(zip(mapping_df['FRED'], mapping_df['Category']))
    
    # Apply transformations to each column
    df_transformed = df.copy()
    for col in df.columns:
        df_transformed[col] = apply_transformation(df[col], tcodes.get(col, 1))
    
    df_transformed = df_transformed.ffill().bfill()
    df_transformed = df_transformed.loc[start:end]
    
    # Add category information as attributes to the DataFrame (What? Very cool!)
    df_transformed.attrs['categories'] = {col: category_map.get(col, 'Unknown') for col in df_transformed.columns}
    
    return df_transformed

# Copying one function from our project thesis replication study
def calculate_excess_returns(
    yields: pd.DataFrame,
    maturities: list = None,
    horizon: int = 12,
    risk_free_col: str = None,
    period: str = "annual",
) -> pd.DataFrame:
    """Calculate excess bond returns for annual or monthly period.

    Parameters
    ----------
    yields : pd.DataFrame
        Yield data with columns like '1m', '2m', ..., '120m'
    maturities : list, optional
        List of maturities in years (e.g., [2, 3, 4, 5, 7, 10]).
        If None, defaults to [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    horizon : int
        Holding period in months (default 12)
    risk_free_col : str, optional
        Column name for risk-free rate
    period : str
        "annual" or "monthly"
    
    Returns
    -------
    pd.DataFrame
        Excess returns
    """
    # Default to all annual maturities
    if maturities is None:
        maturities = list(range(1, 11))  # [1, 2, 3, ..., 10]
    
    # Select and rename annual maturities (12m -> y1, 24m -> y2, etc.)
    annual_maturities = [f"{12*i}m" for i in range(1, 11)]
    yields_annual = yields[annual_maturities].copy()
    yields_annual.columns = [f"y{i}" for i in range(1, 11)]
    
    excess_returns = pd.DataFrame(index=yields_annual.index)
    
    # Determine risk-free rate column
    if risk_free_col is None:
        if period == "monthly" and 'y1' in yields_annual.columns:
            rf_col = 'y1'  # 1-year yield as proxy for 1-month rate
        else:
            rf_maturity = min(maturities)
            rf_col = f'y{rf_maturity}'
    else:
        rf_col = risk_free_col

    if period == "monthly":
        # 1-month holding period approximation
        for maturity in maturities:
            if maturity < 1:
                continue
            y_col = f'y{maturity}'
            xr_col = f'xr1m{maturity}'
            y_n_t = yields_annual[y_col]
            y_n_t1 = yields_annual[y_col].shift(-1)
            rf_t = yields_annual[rf_col]
            remaining_maturity = maturity - 1/12
            excess_returns[xr_col] = (
                -remaining_maturity * y_n_t1 + maturity * y_n_t - (1/12) * rf_t
            ) / 12
        xr_cols = [c for c in excess_returns.columns if c.startswith('xr1m')]
        if xr_cols:
            excess_returns['xr1m_avg'] = excess_returns[xr_cols].mean(axis=1)
        return excess_returns

    # Annual (generic h-month) holding period
    horizon_years = horizon / 12
    for maturity in maturities:
        if maturity <= horizon_years:
            continue
        y_col = f'y{maturity}'
        xr_col = f'xr{maturity}'
        y_n_t = yields_annual[y_col]
        y_h_t = yields_annual[rf_col]
        remaining_maturity = maturity - horizon_years
        remaining_maturity_int = int(remaining_maturity) if remaining_maturity == int(remaining_maturity) else remaining_maturity
        if remaining_maturity_int in maturities:
            y_nh_col = f'y{remaining_maturity_int}'
            y_nh_th = yields_annual[y_nh_col].shift(-horizon)
        else:
            y_nh_th = yields_annual[y_col].shift(-horizon)
        excess_returns[xr_col] = (
            -(remaining_maturity) * y_nh_th + maturity * y_n_t - horizon_years * y_h_t
        )
    xr_cols = [col for col in excess_returns.columns if col.startswith('xr')]
    if xr_cols:
        excess_returns['xr_avg'] = excess_returns[xr_cols].mean(axis=1)
    return excess_returns