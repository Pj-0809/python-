import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.outliers_influence import variance_inflation_factor

# 在线读取Carseats原始数据集（ISLR原版400条）
url = "https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/master/csv/ISLR/Carseats.csv"
Carseats = pd.read_csv(url)

# 构建回归模型：Sales ~ Price + Income + Advertising + C(ShelveLoc)
model = ols("Sales ~ Price + Income + Advertising + C(ShelveLoc)", data=Carseats).fit()

# 输出模型完整拟合报告
print(model.summary())

# ========== 计算VIF（多重共线性）【修复版本】 ==========
# 获取模型设计矩阵（包含常数项、哑变量）
X = model.model.exog
var_names = model.model.exog_names

vif_df = pd.DataFrame()
vif_df["变量名"] = var_names
vif_df["VIF"] = [variance_inflation_factor(X, i) for i in range(X.shape[1])]
print("\n===== VIF结果 =====\n", vif_df)
