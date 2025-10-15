import lightgbm as lgb

def build_lightgbm(params):
    """Khởi tạo mô hình LightGBM Regressor"""
    return lgb.LGBMRegressor(**params)
