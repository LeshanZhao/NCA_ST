from matplotlib.ticker import FixedLocator, FuncFormatter
import pandas as pd
import statsmodels.api as sm

import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

import matplotlib.pyplot as plt
import seaborn as sns


from statsmodels.stats.multitest import multipletests

def SLR_cv(data_path, marker_Neuron, n_splits=5):
    # 读取数据
    df = pd.read_csv(data_path)

    # 初始化k-fold交叉验证
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # 创建一个空的DataFrame来存储结果
    results_df = pd.DataFrame(columns=["pvalue", "y_predict_mean", "Rsquare", "Coefficient", "Intercept", "RMSE"])

    # 对每个距离进行线性回归
    time_columns = [col for col in df.columns if col.startswith('t')]  # 找到所有以't'开头的列
    for time in time_columns:
        # y = df['connectivity_num_log']
        y = df[marker_Neuron]
        X = df[[time]]
        X = sm.add_constant(X)  # 添加常数项

        rsquared_list = []
        rmse_list = []
        predictions = []
        
        for train_index, test_index in kf.split(X):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            
            # 构建并拟合模型
            model = sm.OLS(y_train, X_train)
            results = model.fit()
            
            # 预测并计算RMSE
            y_pred = results.predict(X_test)
            rmse = mean_squared_error(y_test, y_pred, squared=False)
            predictions.extend(y_pred)
            rsquared_list.append(results.rsquared)
            rmse_list.append(rmse)
        
        # 存储结果
        results_df.loc[time] = [
            results.pvalues[time],  # p值
            np.mean(predictions),  # 预测值的平均
            np.mean(rsquared_list),  # 平均R^2值
            results.params[time],  # 回归系数
            results.params['const'],  # 截距
            np.mean(rmse_list)  # 平均RMSE
        ]

    return results_df




def SLR_cv_oneY(RK_data_path, df_Exp_Matrix, marker_Neuron, n_splits=5):
    # 读取数据
    df_RK = pd.read_csv(RK_data_path)

    # 初始化k-fold交叉验证
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # 创建一个空的DataFrame来存储结果
    results_df = pd.DataFrame(columns=["pvalue", "y_predict_mean", "Rsquare", "Coefficient", "Intercept", "RMSE", "FDR"])

    # 对每个距离进行线性回归
    time_columns = [col for col in df_RK.columns if col.startswith('t')]  # 找到所有以't'开头的列

    pvalues = []  # 收集所有p值用于FDR调整

    for time in time_columns:
        # y = df_RK['connectivity_num_log']
        y = df_Exp_Matrix[marker_Neuron]
        X = df_RK[[time]]
        X = sm.add_constant(X)  # 添加常数项

        rsquared_list = []
        rmse_list = []
        predictions = []
        
        for train_index, test_index in kf.split(X):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            
            # 构建并拟合模型
            model = sm.OLS(y_train, X_train)
            results = model.fit()
            
            # 预测并计算RMSE
            y_pred = results.predict(X_test)
            rmse = mean_squared_error(y_test, y_pred, squared=False)
            predictions.extend(y_pred)
            rsquared_list.append(results.rsquared)
            rmse_list.append(rmse)

        # 获取最后一个模型的p值和系数
        results = sm.OLS(y, X).fit()
        pvalues.append(results.pvalues[time])

        # 存储结果
        results_df.loc[time] = [
            results.pvalues[time],  # p值
            np.mean(predictions),  # 预测值的平均
            np.mean(rsquared_list),  # 平均R^2值
            results.params[time],  # 回归系数
            results.params['const'],  # 截距
            np.mean(rmse_list),  # 平均RMSE
            None
        ]
        
    # 对p值进行FDR调整
    _, fdr_pvalues, _, _ = multipletests(pvalues, method='fdr_bh')

    # 将FDR调整后的p值加入结果DataFrame
    results_df["FDR"] = fdr_pvalues
    return results_df

def plot_regression_results_pvalue(df, gene_name, plotdir=""):
    # 设置图形样式
    sns.set(style="whitegrid")

    # 创建一个图形
    plt.figure(figsize=(12, 8))

    df['transformed_pvalue'] = -np.log(df['pvalue'])

    # 绘制每个统计量的折线图
    plt.plot(df.index, df["transformed_pvalue"], marker='o', label='-ln(P-value)', linewidth=5, markersize=8)

    # 设置图表标题和坐标轴标签
    plt.title('P-value of Regression by Distance - ' + gene_name, fontsize = 30 ,weight="bold")
    plt.xlabel('Distance', fontsize = 35 ,weight="bold")
    plt.ylabel('-ln (P-values)', fontsize = 35 ,weight="bold")
    plt.xticks(rotation=45, fontsize=30, weight="bold")  # 旋转标签以便阅读
    plt.yticks(fontsize=20, weight="bold") 

    # 添加参考线（p=0.05处)
    significance_line = -np.log(0.05)
    plt.axhline(y=significance_line, color='red', linestyle='--', label='Significance threshold (-ln (0.05))', linewidth=5)

    # 设置Y轴刻度，包括特殊的刻度
    ax = plt.gca()
    existing_ticks = ax.get_yticks()
    new_ticks = np.append(existing_ticks, significance_line)
    ax.yaxis.set_major_locator(FixedLocator(new_ticks))  # 设置新的Y轴刻度

    # 设置特定刻度的标签为红色并标明数值
    labels = [f"{x:.2f}" for x in existing_ticks]
    labels.append(f"$\\text{{-ln(0.05)}}$")  # 特定刻度的标签
    ax.set_yticklabels(labels, color='black')  # 设置所有标签为黑色
    ax.get_yticklabels()[-1].set_color('red')  # 将特定标签设置为红色




    # 添加图例
    plt.legend(fontsize = 25)

    # 显示图表
    # plt.tight_layout()
    if plotdir != "":
        plt.savefig(plotdir+f'pval_{gene_name}.png')
    
    plt.show()
    plt.close()





def plot_regression_results_beta(df, gene_name, plotdir=""):
    
    # 设置图形样式
    sns.set(style="whitegrid")

    # 创建一个图形
    plt.figure(figsize=(12, 8))


    # 绘制每个统计量的折线图
    plt.plot(df.index, df["Coefficient"], marker = "o", label = "Beta", linewidth=5, markersize=10)

    # # 添加标记的颜色：如果 pvalue > 0.05，标记点为绿色 ------------------------------------------------2024.12.13 添加
    # df['color'] = ['green' if p > 0.05 else 'blue' for p in df['pvalue']]
    # # 添加散点以标记特殊点
    # for i, row in df.iterrows():
    #     plt.scatter(i, row["Coefficient"], color=row['color'], zorder=5, s=10)
    # 添加标记的颜色：如果 pvalue > 0.05，标记点为绿色，否则为蓝色 ------------------------------------------------2024.12.13 添加

    # 设置图表标题和坐标轴标签
    plt.title('Beta of Regression by Distance - ' + gene_name, fontsize=30, weight="bold")
    plt.xlabel('Distance', fontsize=35, weight="bold")
    plt.ylabel('Beta', fontsize=35, weight="bold")
    plt.xticks(rotation=45, fontsize=30, weight="bold")  # 旋转标签以便阅读
    plt.yticks(fontsize=20, weight="bold") 

    # 添加参考线（beta=0处)
    reference_line = 0
    plt.axhline(y=reference_line, color='red', linestyle='--', label='beta=0', linewidth=5)
    
    # plt.ylim(-1, 1)
    
    # 添加图例
    plt.legend(fontsize=25)

    # 显示图表
    # plt.tight_layout()
    if plotdir != "":
        plt.savefig(plotdir+f'beta_{gene_name}.png')
    
    plt.show()
    plt.close()


# ------------------------------------------------2024.12.13 添加
def plot_regression_results_beta_piecewise(df, gene_name, plotdir=""):
    # 设置图形样式
    sns.set(style="whitegrid")

    # 创建一个图形
    plt.figure(figsize=(12, 8))
    # 添加标记的颜色：如果 pvalue > 0.05，标记点为灰色，否则为蓝色
    df['color'] = ['gray' if p > 0.05 \
                            or np.isinf(p) \
                            or pd.isna(p) \
                            else 'blue' for p in df['pvalue']]
    

    # 遍历每个数据点并按段绘制
    start_idx = 0
    current_color = 'blue' if df['color'].iloc[0] == "blue" else 'gray'

    for i in range(1, len(df)):
        next_color = 'blue' if df['color'].iloc[i] == "blue" else 'gray'

        plt.plot(df.index[start_idx:i+1], 
                    df["Coefficient"].iloc[start_idx:i+1], 
                    color="blue" if current_color == 'blue' and next_color == 'blue' else 'gray', 
                    linestyle='-' if current_color == 'blue' and next_color == 'blue' else '--', 
                    linewidth= 5 if current_color == 'blue' and next_color == 'blue' else 3,
                    marker="o",
                    markersize=10)
        start_idx = i
        current_color = next_color


    # 绘制最后一段
    plt.plot(df.index[start_idx:], 
             df["Coefficient"].iloc[start_idx:], 
             color=current_color, 
             linestyle='-' if current_color == 'blue' else '--', linewidth=5, 
             marker="o",
             markersize=10)

    # 添加散点以标记特殊点
    for i, row in df.iterrows():
        plt.scatter(i, row["Coefficient"], color=row['color'], zorder=5, s=80)



    # 设置图表标题和坐标轴标签
    plt.title('Beta of Regression by Distance - ' + gene_name, fontsize=30, weight="bold")
    plt.xlabel('Distance', fontsize=35, weight="bold")
    plt.ylabel('Beta', fontsize=35, weight="bold")
    plt.xticks(rotation=45, fontsize=30, weight="bold")
    plt.yticks(fontsize=20, weight="bold")

    # 添加参考线（beta=0处)
    reference_line = 0
    plt.axhline(y=reference_line, color='red', linestyle='--', label='beta=0', linewidth=5)

    # 添加图例
    plt.legend(fontsize=25)

    # 显示图表
    plt.tight_layout()
    if plotdir != "":
        plt.savefig(plotdir+f'beta_{gene_name}.png')

    plt.show()
    plt.close()
