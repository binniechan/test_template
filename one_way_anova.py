import pandas as pd
import numpy as np
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.sandbox.stats.multicomp import MultiComparison
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import probplot
from scipy.stats import kstest
from scipy.stats import normaltest
from scipy.stats import jarque_bera
from scipy.stats import bartlett
from scipy.stats import fligner
from scipy.stats import ks_2samp
from scipy.stats import ttest_ind


# PlantGrowth 데이터셋 불러오기
data_url = "https://vincentarelbundock.github.io/Rdatasets/csv/datasets/PlantGrowth.csv"
data = pd.read_csv(data_url, index_col=0)

# 데이터 출력
print(data.head())

#임의로 구간 나누는법
# data['Ta_gubun'] = pd.cut(data.maxTa, bins=[-5,8,24,36], labels = [0,1,2])
# data = data[data.Ta_gubun.notna()]
# x1 = np.array(data[data.Ta_gubun == 0].CNT)
# x2 = np.array(data[data.Ta_gubun == 1].CNT)
# x3 = np.array(data[data.Ta_gubun == 2].CNT)

# two way anova에서 data.corr()로 corr봐줘야함
#boxplot
# 박스 플롯 생성
plot_data = data[['group', 'weight']]
plt.figure(figsize=(8, 6))
sns.boxplot(x='group', y='weight', data=plot_data)
plt.title('Box plot of Plant Weights by Group')
plt.show()

#정규성 검정 qqplot
f, axes = plt.subplots(2, 2, figsize=(12, 6))
probplot(data['weight'], plot=axes[0][0])
plt.show()



#정규섬 검정 test
#shapiro, kctest, normal test, Jarque_bera - Test
groups = data['group'].unique()
for group in groups:
    group_data = data[data['group'] == group]['weight']
    _, p_value = stats.shapiro(group_data)
    print(f"{group} Shapiro-Wilk p-value: {p_value}")
    _, p_value = kstest(group_data, 'norm')
    print(f"{group} kstest p-value: {p_value}")
    _, p_value = normaltest(group_data)
    print(f"{group} normal test p-value: {p_value}")
    _, p_value = jarque_bera(group_data)
    print(f"{group} jarque_bera test p-value: {p_value}")
    # 이거는 그룹 나눠서 넣어서 전체비교 할때(x1,x2 이렇게 들어가야함)
    # _, p_value = ks_2samp(group_data)
    # print(f"{group} ks_2samp test p-value: {p_value}")

#정규성 가정 x
# print(stats.kruskal(group1, group2, group3))

#등분산성 검정
#levene, Fligner - Test ,Bartlett - Test
grouped_data = [data[data['group'] == group]['weight'] for group in groups]
_, p_value = stats.levene(*grouped_data) #option center = mean, mean, trimmed 가능
print(f"Levene p-value: {p_value}")
_, p_value = bartlett(*grouped_data)
print(f"bartlett p-value: {p_value}")
_, p_value = fligner(*grouped_data)
print(f"fligner p-value: {p_value}")

#등분산성 가정x Welch’s ANOVA
# from pingouin import welch_anova
#
# df = data
# aov = welch_anova(dv='CNT', between='Ta_gubun', data=df)
# aov



#anova
model = ols('weight ~ group', data=data).fit()
anova_result = sm.stats.anova_lm(model, typ=2)
print(anova_result)

# 이원분산분석
# formula = "len ~ C(supp) * C(dose)"
# model = ols(formula, data).fit()
# anova_result = sm.stats.anova_lm(model, typ=2)
# print(anova_result)
# 사후분석
# mc = MultiComparison(data["len"], data["supp"].astype(str) + "-" + data["dose"].astype(str))
# tukey_result = mc.tukeyhsd()
# print(tukey_result.summary())

## Stats Model을 활용한 방법1.
# F, p = stats.f_oneway(posterior_A_samples
#                       , posterior_B_samples
#                       , posterior_C_samples)
# print( 'F-Ratio: {}'.format(F)
#     , 'p-value:{}'.format(p)
#      , sep = '\n')


#사후분석
tukey_result = pairwise_tukeyhsd(data['weight'], data['group'])
print(tukey_result)
mc = MultiComparison(data['weight'], data['group'])
## BONFERRONI 방식
tbl, a1, a2 = mc.allpairtest(ttest_ind, method="bonf")
print(tbl)
tukey_result.plot_simultaneous()
plt.show()




