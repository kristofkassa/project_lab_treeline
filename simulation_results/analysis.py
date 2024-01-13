import numpy as np
import pandas as pd
from scipy.stats import shapiro, anderson, ttest_1samp
import math
import matplotlib.pyplot as plt
import ast 

def arrange_data(raw_df):
    cols = []
    data = []
    if raw_df.iloc[:,0].apply(lambda x: isinstance(x, str)).all(): #if the first column contains only strings
        cols = ast.literal_eval(max(raw_df.iloc[:, 0], key=len)) #keep the longest entry of the first column, parsed as list
    if raw_df.iloc[:,1].apply(lambda x: isinstance(x, str)).all(): #if the second column contains only strings
        data = [ast.literal_eval(l) for l in raw_df[1].to_list()]

    return pd.DataFrame(data, columns = cols)

def load_data_from_excel(file_name):
    # Load data from each sheet into a separate DataFrame
    box_df = arrange_data(pd.read_excel(file_name, sheet_name='box_counting', index_col=0))
    corr_df = arrange_data(pd.read_excel(file_name, sheet_name='correlation', index_col=0))
    ruler_df = arrange_data(pd.read_excel(file_name, sheet_name='ruler', index_col=0))
    avgdist_df = arrange_data(pd.read_excel(file_name, sheet_name='avgdist', index_col=0))

    return box_df, corr_df, ruler_df, avgdist_df

def slope(xs, ys, intercept_needed=False):
    slope, intercept = np.polyfit(xs, ys, 1)
    if intercept_needed:
         return (slope, intercept)
    else:
        return slope

n = 2**10
file_name = f'simulation_results_2_{int(math.log2(n))}.xlsx' 
box_df, corr_df, ruler_df, avgdist_df = load_data_from_excel(file_name)

box_df["dim"] = box_df.apply(lambda row: -slope(box_df.columns.to_list()[2:-4], row.to_list()[2:-4]), axis=1)
corr_df["dim"] = corr_df.apply(lambda row: slope(corr_df.columns.to_list()[2:-3], row.to_list()[2:-3]), axis=1)
ruler_df["dim"] = ruler_df.apply(lambda row: -slope(ruler_df.columns.to_list()[4:-3], row.to_list()[4:-3]), axis=1)
avgdist_df["dim"] = avgdist_df.apply(lambda row: 1/slope(avgdist_df.columns.to_list()[4:-4], row.to_list()[4:-4]), axis=1)

print(f"n=2^{int(math.log2(n))}, sample size={box_df.shape[0]}")
#print(box_df.describe())
#print(corr_df.describe())
#print(ruler_df.describe())
#print(avgdist_df.describe(), '\n')

print("corr", shapiro(corr_df['dim']), anderson(corr_df['dim']).statistic, anderson(corr_df['dim']).critical_values[-1])
print("rul", shapiro(ruler_df['dim']), anderson(ruler_df['dim']).statistic, anderson(ruler_df['dim']).critical_values[-1])
print("box", shapiro(box_df['dim']), anderson(box_df['dim']).statistic, anderson(box_df['dim']).critical_values[-1])
print("avg", shapiro(avgdist_df['dim']), anderson(avgdist_df['dim']).statistic, anderson(avgdist_df['dim']).critical_values[-1], ttest_1samp(avgdist_df['dim'], 1.75))


fig1, axes1 = plt.subplots(2,2,figsize=(12,12))

corr_df.hist(column="dim", ax=axes1[0,0], bins=10, grid=False,  color='#86bf91', zorder=2, rwidth=0.9)
ruler_df.hist(column="dim", ax=axes1[0,1], bins=10, grid=False, color='#86bf91', zorder=2, rwidth=0.9)
box_df.hist(column="dim", ax=axes1[1,0], bins=10, grid=False,  color='#86bf91', zorder=2, rwidth=0.9)
avgdist_df.hist(column="dim", ax=axes1[1,1], bins=10, grid=False,  color='#86bf91', zorder=2, rwidth=0.9)


axes1[0,0].axvline(corr_df["dim"].mean(), color='red', linestyle='dashed', linewidth=2, label=f'Mean = {corr_df["dim"].mean():.2f} ± {corr_df["dim"].sem():.3f}')
axes1[0,0].axvline(corr_df["dim"].mean()-corr_df["dim"].std(), color='orange', linestyle='dashed', linewidth=2, label=f'std = {corr_df["dim"].std():.2f}')
axes1[0,0].axvline(corr_df["dim"].mean()+corr_df["dim"].std(), color='orange', linestyle='dashed', linewidth=2)
axes1[0,0].set_title('Correlation Method')
axes1[0,0].set_xlabel('Dimension')
axes1[0,0].set_ylabel('Frequency')
axes1[0,0].legend()

axes1[0,1].axvline(ruler_df["dim"].mean(), color='red', linestyle='dashed', linewidth=2, label=f'Mean = {ruler_df["dim"].mean():.2f} ± {ruler_df["dim"].sem():.3f}')
axes1[0,1].axvline(ruler_df["dim"].mean()-ruler_df["dim"].std(), color='orange', linestyle='dashed', linewidth=2, label=f'std = {ruler_df["dim"].std():.2f}')
axes1[0,1].axvline(ruler_df["dim"].mean()+ruler_df["dim"].std(), color='orange', linestyle='dashed', linewidth=2)
axes1[0,1].set_title('Ruler Method')
axes1[0,1].set_xlabel('Dimension')
axes1[0,1].set_ylabel('Frequency')
axes1[0,1].legend()

axes1[1,0].axvline(box_df["dim"].mean(), color='red', linestyle='dashed', linewidth=2, label=f'Mean = {box_df["dim"].mean():.2f} ± {box_df["dim"].sem():.3f}')
axes1[1,0].axvline(box_df["dim"].mean()-box_df["dim"].std(), color='orange', linestyle='dashed', linewidth=2, label=f'std = {box_df["dim"].std():.2f}')
axes1[1,0].axvline(box_df["dim"].mean()+box_df["dim"].std(), color='orange', linestyle='dashed', linewidth=2)
axes1[1,0].set_title('Box Counting')
axes1[1,0].set_xlabel('Dimension')
axes1[1,0].set_ylabel('Frequency')
axes1[1,0].legend()

axes1[1,1].axvline(avgdist_df["dim"].mean(), color='red', linestyle='dashed', linewidth=2, label=f'Mean = {avgdist_df["dim"].mean():.2f} ± {avgdist_df["dim"].sem():.3f}')
axes1[1,1].axvline(avgdist_df["dim"].mean()-avgdist_df["dim"].std(), color='orange', linestyle='dashed', linewidth=2, label=f'std = {avgdist_df["dim"].std():.2f}')
axes1[1,1].axvline(avgdist_df["dim"].mean()+avgdist_df["dim"].std(), color='orange', linestyle='dashed', linewidth=2)
axes1[1,1].set_title('Equipaced Polygon Method')
axes1[1,1].set_xlabel('Dimension')
axes1[1,1].set_ylabel('Frequency')
axes1[1,1].legend()

fig1.suptitle(f'$n=2^{int(math.log2(n))}$, Sample size = {corr_df.shape[0]}')

#original data
orig_corr_df = corr_df.iloc[:, :-1].apply(lambda x: 2**x)
log_mean_corr = [math.log2(x) for x in list(orig_corr_df.mean())]
orig_ruler_df = ruler_df.iloc[:, :-1].apply(lambda x: 2**x)
log_mean_ruler = [math.log2(x) for x in list(orig_ruler_df.mean())]
orig_box_df = box_df.iloc[:, :-1].apply(lambda x: 2**x)
log_mean_box = [math.log2(x) for x in list(orig_box_df.mean())]
orig_avgdist_df = avgdist_df.iloc[:, :-1].apply(lambda x: 2**x)
log_mean_avgdist = [math.log2(x) for x in list(orig_avgdist_df.mean())]

#print(orig_corr_df.describe())
#print(orig_ruler_df.describe())
#print(orig_box_df.describe())
#print(orig_avgdist_df.describe())

#REGRESSION PLOTS
fig2, axes2 = plt.subplots(2,2, figsize=(12,10))

#Corr
log_radius_sizes= corr_df.iloc[:, :-1].columns.to_list()
log_correlation_sums = log_mean_corr
m, intercept = slope(log_radius_sizes[2:-3], log_correlation_sums[2:-3], intercept_needed=True)
regression_line = m * np.array(log_radius_sizes) + intercept
magic_line = (1.75) * np.array(log_radius_sizes) + intercept
axes2[0,0].plot(log_radius_sizes[2:-3], log_correlation_sums[2:-3], 'o', color = 'lime')
axes2[0,0].plot([log_radius_sizes[i] for i in [0,1,-3,-2,-1]], [log_correlation_sums[i] for i in [0,1,-3,-2,-1]], 'o', color = 'gray')
axes2[0,0].plot(log_radius_sizes, regression_line, linestyle='--', color='red', label=f'Regression Line (Slope = {m:.2f})')
axes2[0,0].plot(log_radius_sizes, magic_line, linestyle='--', color='orange', label=f'Predicted Line (Slope = {1.75:.2f})')
axes2[0,0].set_xlabel('Log(Radius Sizes)')
axes2[0,0].set_ylabel('Log(Correlation Sums)')
axes2[0,0].legend()
axes2[0,0].grid(True)

#Ruler
log_ruler_sizes = ruler_df.iloc[:, :-1].columns.to_list()
log_ruler_counts = log_mean_ruler
m, intercept = slope(log_ruler_sizes[4:-3], log_ruler_counts[4:-3], intercept_needed=True)
regression_line = m * np.array(log_ruler_sizes) + intercept
magic_line = (-1.75) * np.array(log_ruler_sizes) + intercept
axes2[0,1].plot(log_ruler_sizes[4:-3], log_ruler_counts[4:-3], 'o', color = 'lime')
axes2[0,1].plot([log_ruler_sizes[i] for i in [0,1,2,3,-3,-2,-1]], [log_ruler_counts[i] for i in [0,1,2,3,-3,-2,-1]], 'o', color = 'gray')
axes2[0,1].plot(log_ruler_sizes, regression_line, linestyle='--', color='red', label=f'Regression Line (Slope = {m:.2f})')
axes2[0,1].plot(log_ruler_sizes, magic_line, linestyle='--', color='orange', label=f'Predicted Line (Slope = {-1.75:.2f})')
axes2[0,1].set_xlabel('Log(Ruler Sizes)')
axes2[0,1].set_ylabel('Log(Ruler Counts)')
axes2[0,1].legend()
axes2[0,1].grid(True)

#Box
log_box_sizes = box_df.iloc[:, :-1].columns.to_list()
log_box_counts = log_mean_box
m, intercept = np.polyfit(log_box_sizes[2:-4], log_box_counts[2:-4], 1)
regression_line = m * np.array(log_box_sizes) + intercept
magic_line = (-1.75) * np.array(log_box_sizes) + intercept
axes2[1,0].plot(log_box_sizes[2:-4], log_box_counts[2:-4], 'o', color = 'lime')
axes2[1,0].plot([log_box_sizes[i] for i in [0,1,-4,-3,-2,-1]], [log_box_counts[i] for i in [0,1,-4,-3,-2,-1]], 'o', color = 'gray')
axes2[1,0].set_xlabel('Log(Box Sizes)')
axes2[1,0].set_ylabel('Log(Box Counts)')
axes2[1,0].plot(log_box_sizes, regression_line, linestyle='--', color='red', label=f'Regression Line (Slope = {m:.2f})')
axes2[1,0].plot(log_box_sizes, magic_line, linestyle='--', color='orange', label=f'Predicted Line (Slope = {-1.75:.2f})')
axes2[1,0].legend()
axes2[1,0].grid(True)

#Avgdist
log_k_lengths = avgdist_df.iloc[:, :-1].columns.to_list()
log_avg_dists = log_mean_avgdist
m, intercept = slope(log_k_lengths[4:-4], log_avg_dists[4:-4], intercept_needed=True)
regression_line = m * np.array(log_k_lengths) + intercept
magic_line = (1/1.75) * np.array(log_k_lengths) + intercept
axes2[1,1].plot(log_k_lengths, regression_line, linestyle='--', color='red', label=f'Regression Line (Slope = 1/{1/m:.2f})')
axes2[1,1].plot(log_k_lengths, magic_line, linestyle='--', color='orange', label=f'Predicted Line (Slope = 1/1.75)')
axes2[1,1].plot(log_k_lengths[4:-4], log_avg_dists[4:-4], 'o', color = 'lime')
axes2[1,1].plot([log_k_lengths[i] for i in [0,1,2,3,-4,-3,-2,-1]], [log_avg_dists[i] for i in [0,1,2,3,-4,-3,-2,-1]], 'o', color = 'gray')
axes2[1,1].set_xlabel('Log(k)')
axes2[1,1].set_ylabel('Log(Average Distances)')
axes2[1,1].legend()
axes2[1,1].grid(True)

fig2.suptitle(f'Regression on average data points ($n=2^{int(math.log2(n))}$, Sample size = {corr_df.shape[0]})')

fig3, axes3 = plt.subplots(2,2, figsize=(12,10))

x = [7,8,9,10]

corr = [1.71, 1.70, 1.72, 1.75]
corr_std = [0.06, 0.04, 0.04, 0.02]

rul = [1.63, 1.65, 1.71, 1.72]
rul_std = [0.06, 0.05, 0.05, 0.06]

box = [1.57, 1.58, 1.63, 1.64]
box_std = [0.07, 0.05, 0.04, 0.03]

avgd = [1.76, 1.74, 1.75, 1.75]
avgd_std = [0.09, 0.07, 0.05, 0.04]

axes3[0,0].errorbar(x, corr, corr_std, fmt='ok', color='red', lw=3)
axes3[0,1].errorbar(x, rul, rul_std, fmt='ok', color='green', lw=2)
axes3[1,0].errorbar(x, box, box_std, fmt='ok', color='blue', lw=3)
axes3[1,1].errorbar(x, avgd, avgd_std, fmt='ok', color='black', lw=3)

axes3[0,0].axhline(1.75, color='orange', linestyle='dashed')
axes3[0,1].axhline(1.75, color='orange', linestyle='dashed')
axes3[1,0].axhline(1.75, color='orange', linestyle='dashed')
axes3[1,1].axhline(1.75, color='orange', linestyle='dashed')

axes3[0,0].grid(True)
axes3[0,1].grid(True)
axes3[1,0].grid(True)
axes3[1,1].grid(True)

plt.show()
