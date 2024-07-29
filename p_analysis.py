from data_load import load_mlb_data
import pandas as pd
import numpy as np

link_list = ['https://drive.google.com/file/d/11Wswb5fHy_BWkQnZ28SSdU6nsUDSomhu/view?usp=drive_link',
             'https://drive.google.com/file/d/1LSO1jY5Tbek6ZN24PuVdfe4gdiZW-vJu/view?usp=drive_link',
             'https://drive.google.com/file/d/1IezCfAMHw5UFroFonCa_ikJqij6QpLEi/view?usp=drive_link']

data = load_mlb_data(link_list)

player_mapping = data[0]
statcast_data = data[1]
traditional_data = data[2]

statcast_data = statcast_data.drop(['Unnamed: 0'], axis = 1)

#This data only includes starting pitchers, this is all I care about for this analysis so GS = 1 which means the player started the game

#To keep things simple for now I will only try to predict the number of outs (PO) that a pitcher will record in a given start

#Data leakage will be an issue in this model so I have to determine which statistics will be known before a pitcher makes a given start. Many of 
# the statistics in the dataset are captured throughout the game. This will introduce bias into the model so I will want to remove these features.

#I noticed that the opponent column had @ depending if the game was home/away. I will remove all @
statcast_data['Opp'] = statcast_data['Opp'].str.replace('@', '')

#I want to start looking at the data and see which values have the most missing values this will impact my model
null_values = pd.DataFrame()
null_values['Missing Values'] = statcast_data.isnull().sum().sort_values(ascending = False)
null_values['Non Null'] = statcast_data.count()
null_values['% Missing'] = null_values['Missing Values'] / statcast_data.shape[0]

#I am going to remove all columns that have more than 50% of the data being null 
under50 = null_values[null_values['% Missing'] > 0.5].index

#Create a list of all columns with more than 50% of its data being missing
#under50 = under50.to_list()

#Remove the columns with more than 50% of its data missing from the original dataframe. I will create a copy to retain the original information
#cleaned_statcast_data = statcast_data.drop(under50, axis = 1).copy()
cleaned_statcast_data = statcast_data.copy()


#This now leaves me with 211 features vs 302 that I started with
#Now I want to look at any statistics that would not be available to me before a game has started. I want to remove all of these except for PO
#PO is going to be our target for this particular model

#I don't want stats that would be accumulated in the start we are trying to predict but stats for all starts before the current one may be useful
#I am going to create features to calculate the season statistics for each pitcher
#Sort the data by date so I can correctly calculate cumulative stats for each season
cleaned_statcast_data = cleaned_statcast_data.sort_values(by = 'Date')

#The season column is an int currently I want to use it to group my data so I will change it to an object dtype
cleaned_statcast_data['season'] = cleaned_statcast_data['season'].astype('object')

#Get the cumulative stats for each player in each season and then shift them by one so we do not include the current row data 
cleaned_statcast_data['Season_W'] = cleaned_statcast_data.groupby(['playerid', 'season'])['W'].cumsum()

#Create Offset_W to make sure the wins are being offset so that the data is not counted until after the current game is completed
#Offset_W is correctly working so I will change this to Season_W to avoid confusion in my dataset
#cleaned_statcast_data['Offset_W'] = cleaned_statcast_data.groupby(['playerid', 'season'])['Season_W'].shift(1, fill_value = 0)
cleaned_statcast_data['Season_W'] = cleaned_statcast_data.groupby(['playerid', 'season'])['Season_W'].shift(1, fill_value = 0)

#I want to apply the same logic to the rest of the statistics in my dataset
cleaned_statcast_data['Season_L'] = cleaned_statcast_data.groupby(['playerid', 'season'])['L'].cumsum()
cleaned_statcast_data['Season_L'] = cleaned_statcast_data.groupby(['playerid', 'season'])['Season_L'].shift(1, fill_value = 0)

cleaned_statcast_data['Season_G'] = cleaned_statcast_data.groupby(['playerid', 'season'])['G'].cumsum()
cleaned_statcast_data['Season_G'] = cleaned_statcast_data.groupby(['playerid', 'season'])['Season_G'].shift(1, fill_value = 0)

cleaned_statcast_data['Season_QS'] = cleaned_statcast_data.groupby(['playerid', 'season'])['QS'].cumsum()
cleaned_statcast_data['Season_QS'] = cleaned_statcast_data.groupby(['playerid', 'season'])['Season_QS'].shift(1, fill_value = 0)

cleaned_statcast_data['Season_CG'] = cleaned_statcast_data.groupby(['playerid', 'season'])['CG'].cumsum()
cleaned_statcast_data['Season_CG'] = cleaned_statcast_data.groupby(['playerid', 'season'])['Season_CG'].shift(1, fill_value = 0)

cleaned_statcast_data['Season_ShO'] = cleaned_statcast_data.groupby(['playerid', 'season'])['ShO'].cumsum()
cleaned_statcast_data['Season_ShO'] = cleaned_statcast_data.groupby(['playerid', 'season'])['Season_ShO'].shift(1, fill_value = 0)

cleaned_statcast_data['Season_L'] = cleaned_statcast_data.groupby(['playerid', 'season'])['L'].cumsum()
cleaned_statcast_data['Season_L'] = cleaned_statcast_data.groupby(['playerid', 'season'])['Season_L'].shift(1, fill_value = 0)

cleaned_statcast_data['Season_IP'] = cleaned_statcast_data.groupby(['playerid', 'season'])['IP'].cumsum()
cleaned_statcast_data['Season_IP'] = cleaned_statcast_data.groupby(['playerid', 'season'])['Season_IP'].shift(1, fill_value = 0)

cleaned_statcast_data['Season_TBF'] = cleaned_statcast_data.groupby(['playerid', 'season'])['TBF'].cumsum()
cleaned_statcast_data['Season_TBF'] = cleaned_statcast_data.groupby(['playerid', 'season'])['Season_TBF'].shift(1, fill_value = 0)

cleaned_statcast_data['Season_H'] = cleaned_statcast_data.groupby(['playerid', 'season'])['H'].cumsum()
cleaned_statcast_data['Season_H'] = cleaned_statcast_data.groupby(['playerid', 'season'])['Season_H'].shift(1, fill_value = 0)

cleaned_statcast_data['Season_R'] = cleaned_statcast_data.groupby(['playerid', 'season'])['R'].cumsum()
cleaned_statcast_data['Season_R'] = cleaned_statcast_data.groupby(['playerid', 'season'])['Season_R'].shift(1, fill_value = 0)

cleaned_statcast_data['Season_ER'] = cleaned_statcast_data.groupby(['playerid', 'season'])['ER'].cumsum()
cleaned_statcast_data['Season_ER'] = cleaned_statcast_data.groupby(['playerid', 'season'])['Season_ER'].shift(1, fill_value = 0)

cleaned_statcast_data['Season_HR'] = cleaned_statcast_data.groupby(['playerid', 'season'])['HR'].cumsum()
cleaned_statcast_data['Season_HR'] = cleaned_statcast_data.groupby(['playerid', 'season'])['Season_HR'].shift(1, fill_value = 0)

cleaned_statcast_data['Season_BB'] = cleaned_statcast_data.groupby(['playerid', 'season'])['BB'].cumsum()
cleaned_statcast_data['Season_BB'] = cleaned_statcast_data.groupby(['playerid', 'season'])['Season_BB'].shift(1, fill_value = 0)

cleaned_statcast_data['Season_IBB'] = cleaned_statcast_data.groupby(['playerid', 'season'])['IBB'].cumsum()
cleaned_statcast_data['Season_IBB'] = cleaned_statcast_data.groupby(['playerid', 'season'])['Season_IBB'].shift(1, fill_value = 0)

cleaned_statcast_data['Season_HBP'] = cleaned_statcast_data.groupby(['playerid', 'season'])['HBP'].cumsum()
cleaned_statcast_data['Season_HBP'] = cleaned_statcast_data.groupby(['playerid', 'season'])['Season_HBP'].shift(1, fill_value = 0)

cleaned_statcast_data['Season_WP'] = cleaned_statcast_data.groupby(['playerid', 'season'])['WP'].cumsum()
cleaned_statcast_data['Season_WP'] = cleaned_statcast_data.groupby(['playerid', 'season'])['Season_WP'].shift(1, fill_value = 0)

cleaned_statcast_data['Season_BK'] = cleaned_statcast_data.groupby(['playerid', 'season'])['BK'].cumsum()
cleaned_statcast_data['Season_BK'] = cleaned_statcast_data.groupby(['playerid', 'season'])['Season_BK'].shift(1, fill_value = 0)

cleaned_statcast_data['Season_SO'] = cleaned_statcast_data.groupby(['playerid', 'season'])['SO'].cumsum()
cleaned_statcast_data['Season_SO'] = cleaned_statcast_data.groupby(['playerid', 'season'])['Season_SO'].shift(1, fill_value = 0)

cleaned_statcast_data['Season_GB'] = cleaned_statcast_data.groupby(['playerid', 'season'])['GB'].cumsum()
cleaned_statcast_data['Season_GB'] = cleaned_statcast_data.groupby(['playerid', 'season'])['Season_GB'].shift(1, fill_value = 0)

cleaned_statcast_data['Season_FB'] = cleaned_statcast_data.groupby(['playerid', 'season'])['FB'].cumsum()
cleaned_statcast_data['Season_FB'] = cleaned_statcast_data.groupby(['playerid', 'season'])['Season_FB'].shift(1, fill_value = 0)

cleaned_statcast_data['Season_LD'] = cleaned_statcast_data.groupby(['playerid', 'season'])['LD'].cumsum()
cleaned_statcast_data['Season_LD'] = cleaned_statcast_data.groupby(['playerid', 'season'])['Season_LD'].shift(1, fill_value = 0)

cleaned_statcast_data['Season_IFFB'] = cleaned_statcast_data.groupby(['playerid', 'season'])['IFFB'].cumsum()
cleaned_statcast_data['Season_IFFB'] = cleaned_statcast_data.groupby(['playerid', 'season'])['Season_IFFB'].shift(1, fill_value = 0)

cleaned_statcast_data['Season_IFH'] = cleaned_statcast_data.groupby(['playerid', 'season'])['IFH'].cumsum()
cleaned_statcast_data['Season_IFH'] = cleaned_statcast_data.groupby(['playerid', 'season'])['Season_IFH'].shift(1, fill_value = 0)

cleaned_statcast_data['Season_BU'] = cleaned_statcast_data.groupby(['playerid', 'season'])['BU'].cumsum()
cleaned_statcast_data['Season_BU'] = cleaned_statcast_data.groupby(['playerid', 'season'])['Season_BU'].shift(1, fill_value = 0)

cleaned_statcast_data['Season_BUH'] = cleaned_statcast_data.groupby(['playerid', 'season'])['BUH'].cumsum()
cleaned_statcast_data['Season_BUH'] = cleaned_statcast_data.groupby(['playerid', 'season'])['Season_BUH'].shift(1, fill_value = 0)

cleaned_statcast_data['Season_Balls'] = cleaned_statcast_data.groupby(['playerid', 'season'])['Balls'].cumsum()
cleaned_statcast_data['Season_Balls'] = cleaned_statcast_data.groupby(['playerid', 'season'])['Season_Balls'].shift(1, fill_value = 0)

cleaned_statcast_data['Season_Strikes'] = cleaned_statcast_data.groupby(['playerid', 'season'])['Strikes'].cumsum()
cleaned_statcast_data['Season_Strikes'] = cleaned_statcast_data.groupby(['playerid', 'season'])['Season_Strikes'].shift(1, fill_value = 0)

cleaned_statcast_data['Season_Pitches'] = cleaned_statcast_data.groupby(['playerid', 'season'])['Pitches'].cumsum()
cleaned_statcast_data['Season_Pitches'] = cleaned_statcast_data.groupby(['playerid', 'season'])['Season_Pitches'].shift(1, fill_value = 0)

cleaned_statcast_data['Season_Pull'] = cleaned_statcast_data.groupby(['playerid', 'season'])['Pull'].cumsum()
cleaned_statcast_data['Season_Pull'] = cleaned_statcast_data.groupby(['playerid', 'season'])['Season_Pull'].shift(1, fill_value = 0)

cleaned_statcast_data['Season_Cent'] = cleaned_statcast_data.groupby(['playerid', 'season'])['Cent'].cumsum()
cleaned_statcast_data['Season_Cent'] = cleaned_statcast_data.groupby(['playerid', 'season'])['Season_Cent'].shift(1, fill_value = 0)

cleaned_statcast_data['Season_Oppo'] = cleaned_statcast_data.groupby(['playerid', 'season'])['Oppo'].cumsum()
cleaned_statcast_data['Season_Oppo'] = cleaned_statcast_data.groupby(['playerid', 'season'])['Season_Oppo'].shift(1, fill_value = 0)

cleaned_statcast_data['Season_Soft'] = cleaned_statcast_data.groupby(['playerid', 'season'])['Soft'].cumsum()
cleaned_statcast_data['Season_Soft'] = cleaned_statcast_data.groupby(['playerid', 'season'])['Season_Soft'].shift(1, fill_value = 0)

cleaned_statcast_data['Season_Med'] = cleaned_statcast_data.groupby(['playerid', 'season'])['Med'].cumsum()
cleaned_statcast_data['Season_Med'] = cleaned_statcast_data.groupby(['playerid', 'season'])['Season_Med'].shift(1, fill_value = 0)

cleaned_statcast_data['Season_Hard'] = cleaned_statcast_data.groupby(['playerid', 'season'])['Hard'].cumsum()
cleaned_statcast_data['Season_Hard'] = cleaned_statcast_data.groupby(['playerid', 'season'])['Season_Hard'].shift(1, fill_value = 0)

cleaned_statcast_data['Season_bipCount'] = cleaned_statcast_data.groupby(['playerid', 'season'])['bipCount'].cumsum()
cleaned_statcast_data['Season_bipCount'] = cleaned_statcast_data.groupby(['playerid', 'season'])['Season_bipCount'].shift(1, fill_value = 0)

#RS(Run Support): Number of runs the pitcher’s team has scored during their appearances
cleaned_statcast_data['Season_RS'] = cleaned_statcast_data.groupby(['playerid', 'season'])['RS'].cumsum()
cleaned_statcast_data['Season_RS'] = cleaned_statcast_data.groupby(['playerid', 'season'])['Season_RS'].shift(1, fill_value = 0)

#pLI The average leverage index of each pitcher’s batters faced
cleaned_statcast_data['Season_pLI'] = cleaned_statcast_data.groupby(['playerid', 'season'])['pLI'].cumsum()
cleaned_statcast_data['Season_pLI'] = cleaned_statcast_data.groupby(['playerid', 'season'])['Season_pLI'].shift(1, fill_value = 0)
cleaned_statcast_data['Season_pLI'] = (cleaned_statcast_data['Season_pLI'] / cleaned_statcast_data['Season_G']).fillna(0)

#inLI The average leverage index at the beginning of the inning for the pitcher
cleaned_statcast_data['Season_inLI'] = cleaned_statcast_data.groupby(['playerid', 'season'])['inLI'].cumsum()
cleaned_statcast_data['Season_inLI'] = cleaned_statcast_data.groupby(['playerid', 'season'])['Season_inLI'].shift(1, fill_value = 0)
cleaned_statcast_data['Season_inLI'] = (cleaned_statcast_data['Season_inLI'] / cleaned_statcast_data['Season_G']).fillna(0)

#gmLI The average leverage index at the beginning of the game for the pitcher
cleaned_statcast_data['Season_gmLI'] = cleaned_statcast_data.groupby(['playerid', 'season'])['gmLI'].cumsum()
cleaned_statcast_data['Season_gmLI'] = cleaned_statcast_data.groupby(['playerid', 'season'])['Season_gmLI'].shift(1, fill_value = 0)
cleaned_statcast_data['Season_gmLI'] = (cleaned_statcast_data['Season_gmLI'] / cleaned_statcast_data['Season_G']).fillna(0)

#exLI The average leverage index at the end of the appearance for the pitcher
cleaned_statcast_data['Season_exLI'] = cleaned_statcast_data.groupby(['playerid', 'season'])['exLI'].cumsum()
cleaned_statcast_data['Season_exLI'] = cleaned_statcast_data.groupby(['playerid', 'season'])['Season_exLI'].shift(1, fill_value = 0)
cleaned_statcast_data['Season_exLI'] = (cleaned_statcast_data['Season_exLI'] / cleaned_statcast_data['Season_G']).fillna(0)

#Based on percentages calculate how many of each pitch type were thrown in the given game
cleaned_statcast_data['Fastballs'] = round((cleaned_statcast_data['FB%1'] * cleaned_statcast_data['Pitches']).fillna(0),0)
cleaned_statcast_data['Curveballs'] = round((cleaned_statcast_data['CB%'] * cleaned_statcast_data['Pitches']).fillna(0),0)
cleaned_statcast_data['ChangeUps'] = round((cleaned_statcast_data['CH%'] * cleaned_statcast_data['Pitches']).fillna(0),0)
cleaned_statcast_data['Sliders'] = round((cleaned_statcast_data['SL%'] * cleaned_statcast_data['Pitches']).fillna(0),0)
cleaned_statcast_data['Cutters'] = round((cleaned_statcast_data['CT%'] * cleaned_statcast_data['Pitches']).fillna(0),0)
cleaned_statcast_data['SplitF'] = round((cleaned_statcast_data['SF%'] * cleaned_statcast_data['Pitches']).fillna(0),0)
cleaned_statcast_data['KnuckleCurve'] = round((cleaned_statcast_data['KN%'] * cleaned_statcast_data['Pitches']).fillna(0),0)
cleaned_statcast_data['UnknownPitch'] = round((cleaned_statcast_data['XX%'] * cleaned_statcast_data['Pitches']).fillna(0),0)

#Calculate Season rates for each pitch type
cleaned_statcast_data['Season_Fastballs'] = cleaned_statcast_data.groupby(['playerid', 'season'])['Fastballs'].cumsum()
cleaned_statcast_data['Season_Fastballs'] = cleaned_statcast_data.groupby(['playerid', 'season'])['Season_Fastballs'].shift(1, fill_value = 0)
cleaned_statcast_data['Season_FB%'] = (cleaned_statcast_data['Season_Fastballs'] / cleaned_statcast_data['Season_Pitches']).fillna(0)

cleaned_statcast_data['Season_Curveballs'] = cleaned_statcast_data.groupby(['playerid', 'season'])['Curveballs'].cumsum()
cleaned_statcast_data['Season_Curveballs'] = cleaned_statcast_data.groupby(['playerid', 'season'])['Season_Curveballs'].shift(1, fill_value = 0)
cleaned_statcast_data['Season_CB%'] = (cleaned_statcast_data['Season_Curveballs'] / cleaned_statcast_data['Season_Pitches']).fillna(0)

cleaned_statcast_data['Season_Changeups'] = cleaned_statcast_data.groupby(['playerid', 'season'])['Changeups'].cumsum()
cleaned_statcast_data['Season_Changeups'] = cleaned_statcast_data.groupby(['playerid', 'season'])['Season_Changeups'].shift(1, fill_value = 0)
cleaned_statcast_data['Season_CH%'] = (cleaned_statcast_data['Season_Changeups'] / cleaned_statcast_data['Season_Pitches']).fillna(0)

cleaned_statcast_data['Season_Sliders'] = cleaned_statcast_data.groupby(['playerid', 'season'])['Sliders'].cumsum()
cleaned_statcast_data['Season_Sliders'] = cleaned_statcast_data.groupby(['playerid', 'season'])['Season_Sliders'].shift(1, fill_value = 0)
cleaned_statcast_data['Season_SL%'] = (cleaned_statcast_data['Season_Sliders'] / cleaned_statcast_data['Season_Pitches']).fillna(0)

cleaned_statcast_data['Season_Cutters'] = cleaned_statcast_data.groupby(['playerid', 'season'])['Cutters'].cumsum()
cleaned_statcast_data['Season_Cutters'] = cleaned_statcast_data.groupby(['playerid', 'season'])['Season_Cutters'].shift(1, fill_value = 0)
cleaned_statcast_data['Season_CT%'] = (cleaned_statcast_data['Season_Cutters'] / cleaned_statcast_data['Season_Pitches']).fillna(0)

cleaned_statcast_data['Season_SplitF'] = cleaned_statcast_data.groupby(['playerid', 'season'])['SplitF'].cumsum()
cleaned_statcast_data['Season_SplitF'] = cleaned_statcast_data.groupby(['playerid', 'season'])['Season_SplitF'].shift(1, fill_value = 0)
cleaned_statcast_data['Season_SF%'] = (cleaned_statcast_data['Season_SplitF'] / cleaned_statcast_data['Season_Pitches']).fillna(0)

cleaned_statcast_data['Season_KnuckleCurve'] = cleaned_statcast_data.groupby(['playerid', 'season'])['KnuckleCurve'].cumsum()
cleaned_statcast_data['Season_KnuckleCurve'] = cleaned_statcast_data.groupby(['playerid', 'season'])['Season_KnuckleCurve'].shift(1, fill_value = 0)
cleaned_statcast_data['Season_KN%'] = (cleaned_statcast_data['Season_KnuckleCurve'] / cleaned_statcast_data['Season_Pitches']).fillna(0)

cleaned_statcast_data['Season_UnknownPitch'] = cleaned_statcast_data.groupby(['playerid', 'season'])['UnknownPitch'].cumsum()
cleaned_statcast_data['Season_UnknownPitch'] = cleaned_statcast_data.groupby(['playerid', 'season'])['Season_UnknownPitch'].shift(1, fill_value = 0)
cleaned_statcast_data['Season_XX%'] = (cleaned_statcast_data['Season_UnknownPitch'] / cleaned_statcast_data['Season_Pitches']).fillna(0)


#Create Season stats for all calculated statistics

#Baseball stats use 6.1 to represent 6-1/3 innings pitched and 6.2 represents 6-2/3 innings pitched. I need to convert these values to 
# whole numbers to determine the number of outs a pitcher recorded in a game. I decided to change IP to a string and 
# split it on the '.' that would give me the number of whole innings pitched and the fractional portion
# once the column was split into two and in a new data frame I then converted both numbers back to integers. Then I multiplied
# my first column (0) by 3 because there are 3 outs in a complete inning and then added column (1) which is the fractional portion
# then I dropped columns 0 and 1 to leave me with the newly created PO column. I then added this back into my cleaned dataframe
temp_po_df = cleaned_statcast_data['IP'].astype('str').str.split('.', expand = True).copy()
temp_po_df[[0, 1]] = temp_po_df[[0, 1]].astype('int')
temp_po_df['PO'] = (temp_po_df[0] * 3) + temp_po_df[1]
temp_po_df = temp_po_df.drop([0, 1], axis = 1)
cleaned_statcast_data = pd.concat([cleaned_statcast_data, temp_po_df], axis =1)

cleaned_statcast_data['Season_PO'] = cleaned_statcast_data.groupby(['playerid', 'season'])['PO'].cumsum()
cleaned_statcast_data['Season_PO'] = cleaned_statcast_data.groupby(['playerid', 'season'])['Season_PO'].shift(1, fill_value = 0)

#ERA is calculated as the number of Earned Runs divided by the number of innings pitched times 9
cleaned_statcast_data['Season_ERA'] = 9 * (cleaned_statcast_data['Season_ER'] / (cleaned_statcast_data['Season_PO'] / 3))

#I will fill nan values in season_ERA to 0 becase if it is a pitcher's first start of the season they will not have an ERA for that season yet
cleaned_statcast_data['Season_ERA'] = cleaned_statcast_data['Season_ERA'].fillna(0)

#Calculate season per 9 stats and fill nan with 0 because there will be no per 9 stats in the first start of the season
cleaned_statcast_data['Season_K/9'] = ((cleaned_statcast_data['Season_SO'] * 9) / (cleaned_statcast_data['Season_PO'] / 3)).fillna(0)
cleaned_statcast_data['Season_BB/9'] = (((cleaned_statcast_data['Season_BB'] + cleaned_statcast_data['Season_IBB']) * 9) / (cleaned_statcast_data['Season_PO'] / 3)).fillna(0)
cleaned_statcast_data['Season_H/9'] = ((cleaned_statcast_data['Season_H'] * 9) / (cleaned_statcast_data['Season_PO'] / 3)).fillna(0)
cleaned_statcast_data['Season_HR/9'] = ((cleaned_statcast_data['Season_HR'] * 9) / (cleaned_statcast_data['Season_PO'] / 3)).fillna(0)

#Begin to calculate ratio stats
#Create strikeout to walk ratio. If the pitcher has not walked a batter set the value equal to the number of season strikeouts for the
#pitcher
cleaned_statcast_data['Season_K/BB'] = np.where((cleaned_statcast_data['Season_BB'] + cleaned_statcast_data['Season_IBB']) == 0, cleaned_statcast_data['Season_SO'],
                                                (cleaned_statcast_data['Season_SO'] / (cleaned_statcast_data['Season_BB'] + cleaned_statcast_data['Season_IBB'])))

#WHIP is calculated as the number of walks plus hits divided by innings pitched
cleaned_statcast_data['Season_WHIP'] = ((cleaned_statcast_data['Season_BB'] + cleaned_statcast_data['Season_IBB'] + cleaned_statcast_data['Season_H']) / (cleaned_statcast_data['Season_PO'] / 3)).fillna(0)

cleaned_statcast_data['Season_AVG'] = (cleaned_statcast_data['H'] / (cleaned_statcast_data['Season_TBF'] - (cleaned_statcast_data['Season_BB'] + cleaned_statcast_data['Season_IBB'])))
#cleaned_statcast_data['Season_AVG'] = cleaned_statcast_data['Season_AVG'].replace(np.inf, 0)

cleaned_statcast_data['Season_BABIP'] = (cleaned_statcast_data['Season_H'] / cleaned_statcast_data['Season_bipCount']).fillna(0)

cleaned_statcast_data['Season_K%'] = (cleaned_statcast_data['Season_SO'] / cleaned_statcast_data['Season_TBF']).fillna(0)

cleaned_statcast_data['Season_BB%'] = ((cleaned_statcast_data['Season_BB'] + cleaned_statcast_data['Season_IBB']) / cleaned_statcast_data['Season_TBF']).fillna(0)

cleaned_statcast_data['Season_K-BB%'] = cleaned_statcast_data['Season_K%'] - cleaned_statcast_data['Season_BB%']

#Using the formula from fangraphs https://library.fangraphs.com/pitching/lob/
cleaned_statcast_data['Season_LOB%'] = ((cleaned_statcast_data['Season_H'] + cleaned_statcast_data['Season_BB'] + cleaned_statcast_data['Season_IBB']+ cleaned_statcast_data['Season_HBP'] - cleaned_statcast_data['Season_R']) / ((cleaned_statcast_data['Season_H'] + cleaned_statcast_data['Season_BB'] + cleaned_statcast_data['Season_IBB']+ cleaned_statcast_data['Season_HBP']) - (1.4 * cleaned_statcast_data['Season_HR']))).fillna(0)

cleaned_statcast_data['Season_GB/FB'] = (cleaned_statcast_data['Season_GB'] / cleaned_statcast_data['Season_FB']).fillna(0)

#IFH% is the percentage of ground balls that are infield hits, calculated as IFG/GB
cleaned_statcast_data['Season_IFH%'] = (cleaned_statcast_data['Season_IFH'] / cleaned_statcast_data['Season_GB']).fillna(0)

#BUH% percentage of bunts that go for hits, calculated as bunt hits / bunts
cleaned_statcast_data['Season_BUH%'] = (cleaned_statcast_data['Season_BUH'] / cleaned_statcast_data['Season_BU']).fillna(0)

#LD% percentage of batter's balls in play that are line drives calculated as LD/BIP
cleaned_statcast_data['Season_LD%'] = (cleaned_statcast_data['Season_LD'] / cleaned_statcast_data['Season_bipCount']).fillna(0)

#GB% percentage of batter's balls in play that are ground balls calculated as GB/BIP
cleaned_statcast_data['Season_GB%'] = (cleaned_statcast_data['Season_GB'] / cleaned_statcast_data['Season_bipCount']).fillna(0)

#FB% percentage of batter's balls in play that are fly balls calculated as FB/BIP
cleaned_statcast_data['Season_FB%'] = (cleaned_statcast_data['Season_FB'] / cleaned_statcast_data['Season_bipCount']).fillna(0)

#IFFB% percentage of batter's balls in play that are infield fly balls calculated as IFFB/BIP
cleaned_statcast_data['Season_IFFB%'] = (cleaned_statcast_data['Season_IFFB'] / cleaned_statcast_data['Season_FB']).fillna(0)

#HR/FB (Home Run to Fly Ball Rate): Percentage of a batter’s fly balls that go for home runs, calculated as HB/FB (even though some HR are line drives).
cleaned_statcast_data['Season_HR/FB'] = (cleaned_statcast_data['Season_HR'] / cleaned_statcast_data['Season_FB']).fillna(0)

#RS/9 (Run Support per 9 innings): Number of runs the pitcher’s team has scored during their appearances per 9 innings
cleaned_statcast_data['Season_RS/9'] = (9 * (cleaned_statcast_data['Season_RS'] / (cleaned_statcast_data['Season_PO'] / 3))).fillna(0)









#Check to make sure the Season Wins are calculating correctly
cols_to_check = ['PlayerName', 'playerid', 'Date', 'Season_FBv', 'FBv', 'Season_G']
checking_W = cleaned_statcast_data[cols_to_check]
checking_W = checking_W[checking_W['playerid'] == 2036]



