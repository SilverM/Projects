#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 21 11:30:35 2018

@author: soonleqi
"""


#Predicting FIFA 2018 Winner

#Method:

#Use data from Kaggle to model the outcome of certain pairings between teams, given their rank, points, and the weighted point difference with the opponent.
Use this model to predict the outcome of the group rounds and then the single-elimination phase
#Compare results with the countries' talent scores
#The analysis can be split into 2 portions:

#Firstly, we will use logistic regression to predict which countries make it to the single elimination round, and subsequently win the world cup.
#(Reference to Dr. James Bond's predictive model)
#Secondly, we will use the talent scores of countries to better support the results of the finals matchup.
#Predicted FIFA 2018 winner : Germany

#Takeaways:

#Overlaying the output with the matchup predictions, we have 3 key observations:

#The logistic regression model coupled with the talent visualization shows that of the 16 top overall talent countries, 11/16 of countries will make it through to the single elimination rounds.
#While this is inconclusive, it does give an indication that the talent pool available is probably an important factor to a country's perfomance at FIFA. The format of FIFA ensures that not all 16 of the top talent pool countries will make it through to the elimination stages with ease, there are multiple points of upset even before FIFA begins. For example, Italy was previously ousted due to losing to Sweden in the 2nd round playoff in Nov 2017, despite having one of the best talents.
#There are multiple matches with almost 50/50 chance for either side to win, this implies that the model is unable to predict with high confidence of some matchups and will need to be tweaked as the results unfold. (this is what makes it interesting to watch isn't it),
#In the quarterfinals matchup, Argentina vs Portugal, only 1 team will make it to the finals with the top 2 players by overall ratings (Ronaldo & Messi) leading on each side. Interestingly, the winning chances of Portugal is only by 0.01.
#In the Finals matchup between Belgium and Germany, there is a substantially higher overall talent score for Germany (4129) as compared to Belgium(3960), even if Belgium were to maximize it's potential, it will only hit a score of 4085 (which is still below Germany's overall score). Hence, unless there is significant environmental influence, both the predictive model & talent assessment results indicate Germany as the champions of FIFA 2018.
#Data Used

#I used 4 datasets

#FIFA rankings from 1993 to 2018 (courtesy of Tadhg Fitzgerald
#This one I used to get the FIFA ranking and points for the teams, which is a monthly changing rank previously shown as a decent predictor of team performance
#International Soccer matches from 1872 to 2018 (courtesy of Mart Jürisoo)
#This I will use to find out how much the difference in point, ranks and the current rank of the team affects the outocme of a match
#FIFA World Cup 2018 data set (courtesy of Nuggs)
#This I will use to get the upcoming matches
#Complete FIFA 2017 Player dataset (Global) (https://www.kaggle.com/artimous/complete-fifa-2017-player-dataset-global) 
#Aggregating countries top players value & strength

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from matplotlib import pyplot as plt
​
rankings = pd.read_csv('../input/fifa-international-soccer-mens-ranking-1993now/fifa_ranking.csv')
rankings = rankings.loc[:,['rank', 'country_full', 'country_abrv', 'cur_year_avg_weighted', 'rank_date', 
                           'two_year_ago_weighted', 'three_year_ago_weighted']]
rankings = rankings.replace({"IR Iran": "Iran"})
rankings['weighted_points'] =  rankings['cur_year_avg_weighted'] + rankings['two_year_ago_weighted'] + rankings['three_year_ago_weighted']
rankings['rank_date'] = pd.to_datetime(rankings['rank_date'])
​
matches = pd.read_csv('../input/international-football-results-from-1872-to-2017/results.csv')
matches =  matches.replace({'Germany DR': 'Germany', 'China': 'China PR'})
matches['date'] = pd.to_datetime(matches['date'])
​
world_cup = pd.read_csv('../input/fifa-worldcup-2018-dataset/World Cup 2018 Dataset.csv')
world_cup = world_cup.loc[:, ['Team', 'Group', 'First match \nagainst', 'Second match\n against', 'Third match\n against']]
world_cup = world_cup.dropna(how='all')
world_cup = world_cup.replace({"IRAN": "Iran", 
                               "Costarica": "Costa Rica", 
                               "Porugal": "Portugal", 
                               "Columbia": "Colombia", 
                               "Korea" : "Korea Republic"})
world_cup = world_cup.set_index('Team')
#Feature Extraction

#Rank data are combined with the day matches are played:

#Then extract some features:

#Point and rank differences
#Friendly matches are removed, assumption being friendly is not reflective of final results in fifa, which is competitive in nature.

# I want to have the ranks for every day 
rankings = rankings.set_index(['rank_date'])\
            .groupby(['country_full'], group_keys=False)\
            .resample('D').first()\
            .fillna(method='ffill')\
            .reset_index()
​
# join the ranks
matches = matches.merge(rankings, 
                        left_on=['date', 'home_team'], 
                        right_on=['rank_date', 'country_full'])
matches = matches.merge(rankings, 
                        left_on=['date', 'away_team'], 
                        right_on=['rank_date', 'country_full'], 
                        suffixes=('_home', '_away'))
# feature generation
matches['rank_difference'] = matches['rank_home'] - matches['rank_away']
matches['point_difference'] = matches['weighted_points_home'] - matches['weighted_points_away']
matches['score_difference'] = matches['home_score'] - matches['away_score']
matches['is_won'] = matches['score_difference'] > 0 # take draw as lost
matches['is_stake'] = matches['tournament'] != 'Friendly'
​
#Modelling

#I used a Simple Logistic regression (Accuracy = 68.18%)

from sklearn import linear_model
from sklearn import ensemble
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
​
X, y = matches.loc[:,['rank_home', 'rank_difference', 'point_difference', 'is_stake']], matches['is_won']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
​
logreg = linear_model.LogisticRegression(C=1e-5)
features = PolynomialFeatures(degree=2)
model = Pipeline([
    ('polynomial_features', features),
    ('logistic_regression', logreg)
])
model = model.fit(X_train, y_train)
score = model.score(X_test,y_test)
cm = confusion_matrix(y_test, model.predict(X_test))
​
import seaborn as sns
plt.figure(figsize=(9,9))
sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r');
plt.ylabel('Actual label');
plt.xlabel('Predicted label');
all_sample_title = 'Accuracy Score: {0}'.format(score)
plt.title(all_sample_title, size = 15);
​
​
​
#With an accuracy level of 68.18%, the model is decent especially since soccer is a team sport and has a low score per game as compared to sports such as NBA -- which means it's much harder to have a good prediction of win loss outcome when game point differences are marginal.

#World Cup simulation
#Group rounds

# let's define a small margin when we safer to predict draw then win
margin = 0.05
​
# let's define the rankings at the time of the World Cup
world_cup_rankings = rankings.loc[(rankings['rank_date'] == rankings['rank_date'].max()) & 
                                    rankings['country_full'].isin(world_cup.index.unique())]
world_cup_rankings = world_cup_rankings.set_index(['country_full'])

from itertools import combinations
​
world_cup['points'] = 0
world_cup['total_prob'] = 0
​
for group in set(world_cup['Group']):
    print('___Starting group {}:___'.format(group))
    for home, away in combinations(world_cup.query('Group == "{}"'.format(group)).index, 2):
        print("{} vs. {}: ".format(home, away), end='')
        row = pd.DataFrame(np.array([[np.nan, np.nan, np.nan, True]]), columns=X_test.columns)
        # entering the rank_home value using loc to search ranking
        row['rank_home'] = world_cup_rankings.loc[home, 'rank']
        home_points = world_cup_rankings.loc[home, 'weighted_points']
        opp_rank = world_cup_rankings.loc[away, 'rank']
        opp_points = world_cup_rankings.loc[away, 'weighted_points']
        # Computing difference in ranking
        row['rank_difference'] = row['rank_home'] - opp_rank
        row['point_difference'] = home_points - opp_points
        
        home_win_prob = model.predict_proba(row)[:,1][0]
        world_cup.loc[home, 'total_prob'] += home_win_prob
        world_cup.loc[away, 'total_prob'] += 1-home_win_prob
        
        
        points = 0
        if home_win_prob <= 0.5 - margin:
            print("{} wins with {:.2f}".format(away, 1-home_win_prob))
            # Establishing the 'points' column here
            world_cup.loc[away, 'points'] += 3
        if home_win_prob > 0.5 - margin:
            points = 1
        if home_win_prob >= 0.5 + margin:
            points = 3
            world_cup.loc[home, 'points'] += 3
            print("{} wins with {:.2f}".format(home, home_win_prob))
        if points == 1:
            print("Draw")
            world_cup.loc[home, 'points'] += 1
            world_cup.loc[away, 'points'] += 1
#Single-elimination rounds


pairing = [0,3,4,7,8,11,12,15,1,2,5,6,9,10,13,14]
​
world_cup = world_cup.sort_values(by=['Group', 'points', 'total_prob'], ascending=False).reset_index()
next_round_wc = world_cup.groupby('Group').nth([0, 1]) # select the top 2
next_round_wc = next_round_wc.reset_index()
next_round_wc = next_round_wc.loc[pairing]
next_round_wc = next_round_wc.set_index('Team')
​
finals = ['round_of_16', 'quarterfinal', 'semifinal', 'final']
​
labels = list()
odds = list()
​
for f in finals:
    print("___Starting of the {}___".format(f))
    iterations = int(len(next_round_wc) / 2)
    winners = []
​
    for i in range(iterations):
        home = next_round_wc.index[i*2]
        away = next_round_wc.index[i*2+1]
        print("{} vs. {}: ".format(home,
                                   away), 
                                   end='')
        row = pd.DataFrame(np.array([[np.nan, np.nan, np.nan, True]]), columns=X_test.columns)
        row['rank_home'] = world_cup_rankings.loc[home, 'rank']
        home_points = world_cup_rankings.loc[home, 'weighted_points']
        opp_rank = world_cup_rankings.loc[away, 'rank']
        opp_points = world_cup_rankings.loc[away, 'weighted_points']
        row['rank_difference'] = row['rank_home'] - opp_rank
        row['point_difference'] = home_points - opp_points
​
        home_win_prob = model.predict_proba(row)[:,1][0]
        if model.predict_proba(row)[:,1] <= 0.5:
            print("{0} wins with probability {1:.2f}".format(away, 1-home_win_prob))
            winners.append(away)
        else:
            print("{0} wins with probability {1:.2f}".format(home, home_win_prob))
            winners.append(home)
​
        labels.append("{}({:.2f}) vs. {}({:.2f})".format(world_cup_rankings.loc[home, 'country_abrv'], 
                                                        1/home_win_prob, 
                                                        world_cup_rankings.loc[away, 'country_abrv'], 
                                                        1/(1-home_win_prob)))
        odds.append([home_win_prob, 1-home_win_prob])
                
    next_round_wc = next_round_wc.loc[winners]
    print("\n")


import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
​
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls

df = pd.read_csv('../input/complete-player-info/CompleteDataset.csv')
df.head()
#I will assume that the top 50 talents from each country rated by 'Overall' score is representative of the country's soccer capabilities


df2 = df.sort_values(['Nationality','Overall'],ascending=False).groupby('Nationality').head(50)


def str2number(amount):
    if amount[-1] == 'M':
        return float(amount[1:-1])*1000000
    elif amount[-1] == 'K':
        return float(amount[1:-1])*1000
    else:
        return float(amount[1:])
    
df2['MaxPotential'] = df2['Potential'] - df2['Overall']
df2['ValueNum'] = df2['Value'].apply(lambda x: str2number(x))
top_teams = df2.groupby("Nationality").sum().sort_values("ValueNum", ascending=False).head(16).reset_index()[["Nationality", "Overall", "ValueNum",'MaxPotential']]

#Visualizing top 16 countries with the highest talent score

trace1 = go.Bar(
    x = top_teams["Nationality"].tolist(),
    y = top_teams["Overall"].tolist(),
    name='Country Overall',
    marker=dict(
        color='rgba(55, 128, 191, 0.7)',
        line=dict(
            color='rgba(55, 128, 191, 1.0)',
            width=2,
        )
    )
)
​
trace2 = go.Bar(
    x = top_teams["Nationality"].tolist(),
    y = top_teams["MaxPotential"].tolist(),
    name='Country Potential',
    marker=dict(
        color='rgba(219, 64, 82, 0.7)',
        line=dict(
            color='rgba(219, 64, 82, 1.0)',
            width=2,
        )
    )
)
​
trace3 = go.Scatter(
    x = top_teams["Nationality"].tolist(),
    y = (top_teams["ValueNum"] / 1000000).tolist(),
    name='Country Value [M€]',
    mode = 'lines+markers',
    yaxis='y2'
)
​
data = [trace1, trace2,trace3]
​
layout = go.Layout(
    barmode='stack',
    title = 'Level of talent across countries',
    titlefont=dict(size=25),
    width=850,
    height=500,
    paper_bgcolor='rgb(244, 238, 225)',
    plot_bgcolor='rgb(244, 238, 225)',
    yaxis = dict(
        title= 'Country Overall/Potential',
        anchor = 'x',
        rangemode='tozero'
    ),
    xaxis = dict(title= 'Country Name'),
    yaxis2=dict(
        title='Country Value [M€]',
        titlefont=dict(
            color='rgb(148, 103, 189)'
        ),
        tickfont=dict(
            color='rgb(148, 103, 189)'
        ),
        overlaying='y',
        side='right',
        anchor = 'x',
        rangemode = 'tozero',
        dtick = 200
    ),
    #legend=dict(x=-.1, y=1.2)
    legend=dict(x=0.05, y=0.05)
)
​
fig = go.Figure(data=data, layout=layout)
py.iplot(fig)


# Countries identified by logreg to make it past group stages.
next_round_wc = world_cup.groupby('Group').nth([0, 1]) 
next_round_wc['Team']


# Overlaps
next_round_wc['Team'].isin(top_teams['Nationality']).agg('sum')
#Takeaways:

#Overlaying the output with the matchup predictions, we have 3 key observations:

#The logistic regression model coupled with the talent visualization shows that of the 16 top overall talent countries, 11/16 of countries will make it through to the single elimination rounds.
#While this is inconclusive, it does give an indication that the talent pool available is probably an important factor to a country's perfomance at FIFA. The format of FIFA ensures that not all 16 of the top talent pool countries will make it through to the elimination stages with ease, there are multiple points of upset even before FIFA begins. For example, Italy was previously ousted due to losing to Sweden in the 2nd round playoff in Nov 2017, despite having one of the best talents.
#There are multiple matches with almost 50/50 chance for either side to win, this implies that the model is unable to predict with high confidence of some matchups and will need to be tweaked as the results unfold. (this is what makes it interesting to watch isn't it),
#In the quarterfinals matchup, Argentina vs Portugal, only 1 team will make it to the finals with the top 2 players by overall ratings (Ronaldo & Messi) leading on each side. Interestingly, the winning chances of Portugal is only by 0.01.
#In the Finals matchup between Belgium and Germany, there is a substantially higher overall talent score for Germany (4129) as compared to Belgium(3960), even if Belgium were to maximize it's potential, it will only hit a score of 4085 (which is still below Germany's overall score). Hence, unless there is significant environmental influence, both the predictive model & talent assessment results indicate Germany as the champions of FIFA 2018.
​

