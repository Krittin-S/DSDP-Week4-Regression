# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 16:20:58 2020

@author: nyzw
"""

#install packages if needed
#pip install #####


#import modules
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import math
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
from matplotlib import rcParams
from statsmodels.nonparametric.smoothers_lowess import lowess
from scipy.stats import shapiro
from statsmodels.graphics.gofplots import qqplot
from statsmodels.stats.outliers_influence import variance_inflation_factor
from patsy import dmatrices
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.anova import anova_lm

#Import in xlsx file
FacebookLikes = pd.read_excel(r'C:\Users\nyzw\OneDrive - Chevron\DSDP\DSDP 2020\Lectures\Week 4 Regression\Facebook_Post_Prediction_Truncated.xlsx')

#Look at data - visually look! 
#(may also open data frame in IDE to get a "cleaner" look)

#Look at the dataset dimensions
FacebookLikes.shape

#Check the variable types
FacebookLikes.info()

#Summary Statistics for dataset
SummaryResults = FacebookLikes.describe(include='all')


#Multivariate regression
PredFBLikes = smf.ols('NumberLikes24 ~ CommentCount24 + PostLength + PostShareCount +  PostSaturday', data = FacebookLikes).fit()
PredFBLikes.summary()

#Model Diagnostics
#Residuals vs Fitted Plot
residuals = PredFBLikes.resid
fitted = PredFBLikes.fittedvalues
smoothed = lowess(residuals,fitted)
top3 = abs(residuals).sort_values(ascending = False)[:3]

plt.rcParams.update({'font.size': 16})
plt.rcParams["figure.figsize"] = (8,7)
fig, ax = plt.subplots()
ax.scatter(fitted, residuals, edgecolors = 'k', facecolors = 'none')
ax.plot(smoothed[:,0],smoothed[:,1],color = 'r')
ax.set_ylabel('Residuals')
ax.set_xlabel('Fitted Values')
ax.set_title('Residuals vs. Fitted')
ax.plot([min(fitted),max(fitted)],[0,0],color = 'k',linestyle = ':', alpha = .3)

for i in top3.index:
    ax.annotate(i,xy=(fitted[i],residuals[i]))

plt.show()

#Normal Q-Q plot
sorted_student_residuals = pd.Series(PredFBLikes.get_influence().resid_studentized_internal)
sorted_student_residuals.index = PredFBLikes.resid.index
sorted_student_residuals = sorted_student_residuals.sort_values(ascending = True)
df = pd.DataFrame(sorted_student_residuals)
df.columns = ['sorted_student_residuals']
df['theoretical_quantiles'] = stats.probplot(df['sorted_student_residuals'], dist = 'norm', fit = False)[0]
rankings = abs(df['sorted_student_residuals']).sort_values(ascending = False)
top3 = rankings[:3]

fig, ax = plt.subplots()
x = df['theoretical_quantiles']
y = df['sorted_student_residuals']
ax.scatter(x,y, edgecolor = 'k',facecolor = 'none')
ax.set_title('Normal Q-Q')
ax.set_ylabel('Standardized Residuals')
ax.set_xlabel('Theoretical Quantiles')
ax.plot([np.min([x,y]),np.max([x,y])],[np.min([x,y]),np.max([x,y])], color = 'r', ls = '--')
for val in top3.index:
    ax.annotate(val,xy=(df['theoretical_quantiles'].loc[val],df['sorted_student_residuals'].loc[val]))
plt.show()


#Scale - Location Plot
student_residuals = PredFBLikes.get_influence().resid_studentized_internal
sqrt_student_residuals = pd.Series(np.sqrt(np.abs(student_residuals)))
sqrt_student_residuals.index = PredFBLikes.resid.index
smoothed = lowess(sqrt_student_residuals,fitted)
top3 = abs(sqrt_student_residuals).sort_values(ascending = False)[:3]

fig, ax = plt.subplots()
ax.scatter(fitted, sqrt_student_residuals, edgecolors = 'k', facecolors = 'none')
ax.plot(smoothed[:,0],smoothed[:,1],color = 'r')
ax.set_ylabel('$\sqrt{|Studentized \ Residuals|}$')
ax.set_xlabel('Fitted Values')
ax.set_title('Scale-Location')
ax.set_ylim(0,max(sqrt_student_residuals)+0.1)
for i in top3.index:
    ax.annotate(i,xy=(fitted[i],sqrt_student_residuals[i]))
plt.show()


#Residuals vs. Leverage
student_residuals = pd.Series(PredFBLikes.get_influence().resid_studentized_internal)
student_residuals.index = PredFBLikes.resid.index
df = pd.DataFrame(student_residuals)
df.columns = ['student_residuals']
df['leverage'] = PredFBLikes.get_influence().hat_matrix_diag
smoothed = lowess(df['student_residuals'],df['leverage'])
sorted_student_residuals = abs(df['student_residuals']).sort_values(ascending = False)
top3 = sorted_student_residuals[:3]

fig, ax = plt.subplots()
x = df['leverage']
y = df['student_residuals']
xpos = max(x)+max(x)*0.01  
ax.scatter(x, y, edgecolors = 'k', facecolors = 'none')
ax.plot(smoothed[:,0],smoothed[:,1],color = 'r')
ax.set_ylabel('Studentized Residuals')
ax.set_xlabel('Leverage')
ax.set_title('Residuals vs. Leverage')
ax.set_ylim(min(y)-min(y)*0.15,max(y)+max(y)*0.15)
ax.set_xlim(-0.01,max(x)+max(x)*0.05)
plt.tight_layout()
for val in top3.index:
    ax.annotate(val,xy=(x.loc[val],y.loc[val]))

cooksx = np.linspace(min(x), xpos, 50)
p = len(PredFBLikes.params)
poscooks1y = np.sqrt((p*(1-cooksx))/cooksx)
poscooks05y = np.sqrt(0.5*(p*(1-cooksx))/cooksx)
negcooks1y = -np.sqrt((p*(1-cooksx))/cooksx)
negcooks05y = -np.sqrt(0.5*(p*(1-cooksx))/cooksx)

ax.plot(cooksx,poscooks1y,label = "Cook's Distance", ls = ':', color = 'r')
ax.plot(cooksx,poscooks05y, ls = ':', color = 'r')
ax.plot(cooksx,negcooks1y, ls = ':', color = 'r')
ax.plot(cooksx,negcooks05y, ls = ':', color = 'r')
ax.plot([0,0],ax.get_ylim(), ls=":", alpha = .3, color = 'k')
ax.plot(ax.get_xlim(), [0,0], ls=":", alpha = .3, color = 'k')
ax.annotate('1.0', xy = (xpos, poscooks1y[-1]), color = 'r')
ax.annotate('0.5', xy = (xpos, poscooks05y[-1]), color = 'r')
ax.annotate('1.0', xy = (xpos, negcooks1y[-1]), color = 'r')
ax.annotate('0.5', xy = (xpos, negcooks05y[-1]), color = 'r')
ax.legend()
plt.show()

#Statistical Assumptions
#Linearity (DV and IV)
plt.scatter(x=FacebookLikes['CommentCount24'] , y=FacebookLikes['NumberLikes24'])
plt.scatter(x=FacebookLikes['PostLength'] , y=FacebookLikes['NumberLikes24'])
plt.scatter(x=FacebookLikes['PostShareCount'] , y=FacebookLikes['NumberLikes24'])

#Normality
#For the IV* (not really needed since it is technically not an assumption - should be done in cleaning)
#For the IV
#CommentCount
stat, p = shapiro(FacebookLikes['CommentCount24'])
print('Shapiro Statistic=%.3f, p=%.3f' % (stat, p))

FacebookLikes['CommentCount24'].plot.hist(alpha=0.5, bins=100, grid=True, 
            legend=None, density = True, color = 'gray', edgecolor = 'black')
mu = FacebookLikes['CommentCount24'].mean()
variance = FacebookLikes['CommentCount24'].var()
sigma = math.sqrt(variance)
x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
plt.plot(x, stats.norm.pdf(x, mu, sigma),color = 'r')
plt.show()

qqplot(FacebookLikes['CommentCount24'], line='q')
plt.show()

#PostLength
stat, p = shapiro(FacebookLikes['PostLength'])
print('Shapiro Statistic=%.3f, p=%.3f' % (stat, p))

FacebookLikes['PostLength'].plot.hist(alpha=0.5, bins=100, grid=True, 
            legend=None, density = True, color = 'gray', edgecolor = 'black')
mu = FacebookLikes['PostLength'].mean()
variance = FacebookLikes['PostLength'].var()
sigma = math.sqrt(variance)
x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
plt.plot(x, stats.norm.pdf(x, mu, sigma),color = 'r')
plt.show()

qqplot(FacebookLikes['PostLength'], line='q')
plt.show()

#PostShareCount
stat, p = shapiro(FacebookLikes['PostShareCount'])
print('Shapiro Statistic=%.3f, p=%.3f' % (stat, p))

FacebookLikes['PostShareCount'].plot.hist(alpha=0.5, bins=100, grid=True, 
            legend=None, density = True, color = 'gray', edgecolor = 'black')
mu = FacebookLikes['PostShareCount'].mean()
variance = FacebookLikes['PostShareCount'].var()
sigma = math.sqrt(variance)
x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
plt.plot(x, stats.norm.pdf(x, mu, sigma),color = 'r')
plt.show()

qqplot(FacebookLikes['PostShareCount'], line='q')
plt.show()

#For the Residuals
stat, p = shapiro(residuals)
print('Shapiro Statistic=%.3f, p=%.3f' % (stat, p))

residuals.plot.hist(alpha=0.5, bins=50, grid=True, 
                    legend=None, density = True, color = 'gray', edgecolor = 'black')
mu = residuals.mean()
variance = residuals.var()
sigma = math.sqrt(variance)
x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
plt.plot(x, stats.norm.pdf(x, mu, sigma),color = 'r')
plt.show()

qqplot(residuals, line='q')
plt.show()

#Multicollinearity
#Grab the numerical data for the correlation matrix
df_sub = pd.DataFrame(FacebookLikes, columns=['NumberLikes24', 'CommentCount24', 'PostLength', 'PostShareCount'])
corrmatrix = df_sub.corr()
print(corrmatrix)

# Calculate VIF for IVs
y, X = dmatrices(formula_like="NumberLikes24 ~ CommentCount24 + PostLength + PostShareCount +  PostSaturday ", 
                 data=FacebookLikes, return_type="dataframe")
vif = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
print('CommentCount24 VIF =', vif[1])
print('PostLength VIF =', vif[2])
print('PostShareCount VIF =', vif[3])
print('PostSaturday VIF =', vif[3])

#Autocorrelation
print('Durbin-Watson =', durbin_watson(residuals))

#Stepwise Regression
DataSubset = pd.DataFrame(FacebookLikes, columns=['NumberLikes24', 'CommentCount24', 
                                                 'PostLength', 'PostShareCount', 'PostSaturday'])

def forward_selected(data, response):
    """Linear model designed by forward selection.

    Parameters:
    -----------
    data : pandas DataFrame with all possible predictors and response

    response: string, name of response column in data

    Returns:
    --------
    model: an "optimal" fitted statsmodels linear model
           with an intercept
           selected by forward selection
           evaluated by adjusted R-squared
    """
    remaining = set(data.columns)
    remaining.remove(response)
    selected = []
    current_score, best_new_score = 0.0, 0.0
    while remaining and current_score == best_new_score:
        scores_with_candidates = []
        for candidate in remaining:
            formula = "{} ~ {} + 1".format(response,
                                           ' + '.join(selected + [candidate]))
            score = smf.ols(formula, data).fit().rsquared_adj
            scores_with_candidates.append((score, candidate))
        scores_with_candidates.sort()
        best_new_score, best_candidate = scores_with_candidates.pop()
        if current_score < best_new_score:
            remaining.remove(best_candidate)
            selected.append(best_candidate)
            current_score = best_new_score
    formula = "{} ~ {} + 1".format(response,
                                   ' + '.join(selected))
    model = smf.ols(formula, data).fit()
    return model

model = forward_selected(DataSubset, 'NumberLikes24')
print('Model Selected')
print(model.model.formula)
print('R2=', model.rsquared_adj)