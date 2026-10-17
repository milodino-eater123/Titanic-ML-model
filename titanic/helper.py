def check_missing(df):
    return df.isna().any(axis=1).sum(),len(df)
def clean_missing_results(df):
    return df.dropna(subset=["Survived"])