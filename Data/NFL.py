#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 16:34:46 2022

@author: deji
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import requests
from bs4 import BeautifulSoup 
import time
from sklearn.linear_model import LogisticRegression 
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import regex as re

sns.set()

pd.options.display.max_columns = None


# function to extract wins/losses of each game from past n years
# no functionality for user input or error handling yet
# start and end year give only last two digits

def get_win_loss(start_year, end_year):
    
    url = 'https://www.pro-football-reference.com/years/20'
    year = start_year
    data = []
    v_list = []
    
    while year < end_year:
        time.sleep(0.1)
        response = requests.get(url + str(year) + '/games.htm')
        soup = BeautifulSoup(response.text, 'lxml')
        table = soup.find('table')
        if table:
            table_body = table.find('tbody')
            for row in table_body.findAll('tr', attrs={'class': False}):
                links = row.findAll('a')
                week = row.find('th')
                cells = row.findAll('td')
                visitor = bool(re.search("@", cells[4].text))
                v_list.append(visitor)
                try:
                    boxscore_ = links[2]['href']
                    team_1 = cells[3].text
                    team_2 = cells[5].text
                    team_1_score = int(cells[7].text)
                    team_2_score = int(cells[8].text)
                    team_1_yards = int(cells[9].text)
                    team_2_yards = int(cells[11].text)
                    team_1_tovs = int(cells[10].text)
                    team_2_tovs = int(cells[12].text)
                    data_dict_W = {'boxscore':boxscore_,
                                    'name': team_1,
                                    'score': team_1_score,
                                    'yards': team_1_yards,
                                    'tovs':  team_1_tovs,
                                    'outcome': 1}
                    data.append(data_dict_W)
                                   
                    data_dict_L = {'boxscore':boxscore_,
                                   'name': team_2,
                                   'score': team_2_score,
                                   'yards': team_2_yards,
                                   'tovs':  team_2_tovs,
                                   'outcome': 0}
                    data.append(data_dict_L)
                except:
                    pass
    
        year += 1
    
    return pd.DataFrame(data), v_list


# function to extract passer rating of each QB in each game from past 5 years
def get_pass_stats(url, visitor):
    # knowledge of visiting team is needed because of pro football reference's setup
    # visiting QB stats come 1st 
    
    response = requests.get('https://www.pro-football-reference.com' + url)
    soup = BeautifulSoup(response.text, 'lxml')
    table = soup.find('table', id='player_offense')
    stat_list = []
    
    if table:
        table_body = table.find('tbody')
        for count, row in enumerate(table_body.findAll('tr', attrs={'class':False})):
            f_team = row.find('td')
            team = f_team.text
            if visitor:
                if count == 0:
                    try:
                        team_1 = team
                        player_url = row.find('a')          # for future stat lookup if needed
                        player_name = player_url.text
                        player_page = player_url['href']    # unused for now
                        cells = row.findAll('td')
                        
                        rate = float(cells[9].text)
                            
                        player_dict_W = {'qb':player_name,
                                       'rate': rate}
                        stat_list.append(player_dict_W)
                    except:
                        pass
                elif team != team_1:
                    try:
                        player_url = row.find('a')          
                        player_name = player_url.text
                        player_page = player_url['href']    
                        cells = row.findAll('td')
                        
                        rate = float(cells[9].text)
                            
                        player_dict_L = {'qb':player_name,
                                       'rate': rate}
                        stat_list.append(player_dict_L)
                        break
                    except:
                        pass
                else:
                    continue
            else:
                if count == 0:
                    try:
                        team_1 = team
                        player_url = row.find('a')          # for future stat lookup if needed
                        player_name = player_url.text
                        player_page = player_url['href']    # unused for now
                        cells = row.findAll('td')
                        
                        rate = float(cells[9].text)
                            
                        player_dict_L = {'qb':player_name,
                                       'rate': rate}
                    except:
                        pass
                elif team != team_1:
                    try:
                        player_url = row.find('a')          
                        player_name = player_url.text
                        player_page = player_url['href']    
                        cells = row.findAll('td')
                        
                        rate = float(cells[9].text)
                            
                        player_dict_W = {'qb':player_name,
                                       'rate': rate}
                        stat_list.append(player_dict_W)
                        stat_list.append(player_dict_L)
                        break
                    except:
                        pass
                else:
                    continue
            
    return stat_list



'''

MAIN

'''

win_loss, v_list = get_win_loss(19, 21)
print('Loaded.')

qbr = pd.DataFrame()

j = 0
for i, url in enumerate(win_loss.boxscore):
    if i % 2 == 0:
        pass_stats = get_pass_stats(url, v_list[j])
        qbr = qbr.append(pass_stats, ignore_index = True)
        j += 1
        time.sleep(0.1)
    else:
        continue

df = pd.concat([win_loss, qbr], axis=1)


# creates either logistic or linear regression object based on given input
def reg(inputs, dv, reg_type):
    if reg_type == 'log':
        x = sm.add_constant(inputs)
        reg_log = sm.Logit(dv, x)
        results_log = reg_log.fit()
        return results_log
    
    elif reg_type == 'linear':
        lin_reg = LinearRegression()
        lin_reg.fit(inputs, dv)
        return lin_reg
    else:
        print('Input the regression type as \'log\' forlogistic regression, or \'linear\' for linear regression.')
        return
    




# evaluation of influence that passer rating vs team turnovers have on winning
y = df['outcome']
x1 = df['rate']


x1_train, x1_test, y1_train, y1_test = train_test_split(x1, y, random_state = 20, test_size=0.2)

y_train_1 = y1_train.values.reshape(-1,1)
reg_log = reg(x1_train, y_train_1, 'log')
reg_log.summary()

coefs = [-1.40, 0.015]



# examine accuracy
cm_df = pd.DataFrame(reg_log.pred_table())
cm_df.columns = ['Predicted 0', 'Predicted 1']
cm_df = cm_df.rename(index={0: 'Actual 0', 1: 'Actual 1'})
cm = np.array(cm_df)

accuracy_train = (cm[0,0] + cm[1,1])/cm.sum()

reg_log_summary = pd.DataFrame(x1.columns.values, columns=['Features'])
reg_log_summary['Weights'] = coefs



'''
Conclusion: Passer rating is not a reliable indicator of winning vs losing. 
Proper evaluation of the impact of a QBs performance's on winning should involve: 
    - replace rating with other stats
        - (QBR, EPA/dropback, DVOA)
    - features related to qb-independent offensive impact
    - features related to defensive impact
    - features related to special teams impact
    - features related to coaching impact
    - features related to environment (ex. elevation, weather conditions)
    - features related to off-field impact (ex. travel distance)
    - separate analysis on different game situations (ex: by quarter/half, trailing/leading)

'''


