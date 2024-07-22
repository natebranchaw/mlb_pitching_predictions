MLB Pitching Predictions - Machine Learning Approach

I am predicting MLB pitching stats utilizing Statcast and traditional statistics

A full definition of statistics included in my analysis is listed below <br />
PlayerName - <br />
playerid <br />
Date <br />
Opp <br />
teamid <br />
season <br />
Team <br />
HomeAway <br />
Age <br />
W - number of wins  <br />
L - number of losses <br />
ERA - average number of earned runs a pitcher allows per 9 innings <br />
G - number of games in which the pitcher appeared <br />
GS - number of games the pitcher started <br />
QS <br />
CG - number of starts in which the pitcher recorded every out of the game <br />
ShO - number of complete games in which the pitcher allowed 0 runs <br />
SV - number of saves <br />
HLD - number of holds <br />
BS - number of blown saves <br />
IP - total number of innings pitched (.1 represents 1/3 of an inning and .2 represents 2/3 of an inning) <br />
TBF - the number of batters a pitcher has faced, akin to plate appearances <br />
H - number of hits allowed by the pitcher <br />
R - number of runs allowed by the pitcher <br />
ER - number of earned runs allowed by the pitcher, determined by the official scorer <br />
HR - number of home runs allowed by the pitcher <br />
BB - number of walks allowed by the pitcher <br />
IBB - number of intentional walks issued by the pitcher <br />
HBP - number of hit batters <br />
WP - number of wild pitches <br />
BK - number of balks <br />
SO - number of strikeouts <br />
K/9 - average number of strikeouts per 9 innings <br />
BB/9 - average number of walks per 9 innings <br />
H/9 - average number of hits per 9 innings <br />
K/BB - strikeouts divided by walks <br />
IFH% - percentage of ground balls that are infield hits <br />
BUH% - percentage of bunts that go for hits <br />
GB - number of ground balls <br />
FB - number of fly balls <br />
LD - number of line drives <br />
IFFB - number of infield fly balls <br />
IFH - number of infield hits <br />
BU - <br />
BUH - number of bunt hits <br />
K% - frequency with which the pitcher has struck out a batter, strikeouts/hitters faced <br />
BB% - frequency with which the pitcher has walked a batter, walks/hitters faced <br />
K-BB% - 
SIERA - an ERA estimator that attempts to more accurately capture a pitcher's performance based on strikeouts, walks/HBP, home runs and batted ball data  <br />
HR/9 - average number of home runs per 9 innings  <br />
AVG - opponent's batting average against  <br />
WHIP - the average number of base runniers allowed via hit or walk per inning  <br />
BABIP - the rate at which the pitcher allows a hit when the ball in put in play (Batting Average on Balls in Play)  <br />
LOB% - percentage of pitcher's own base runners that they strand over the course of a season  <br />
 'FIP',  <br />
 'E-F',  <br />
 'xFIP',  <br />
 'ERA-',  <br />
 'FIP-',  <br />
 'xFIP-',  <br />
 'GB/FB',<br />
 'LD%',<br />
 'GB%',<br />
 'FB%',<br />
 'IFFB%',<br />
 'HR/FB',<br />
 'RS',<br />
 'RS/9',<br />
 'Balls',<br />
 'Strikes',<br />
 'Pitches',<br />
 'WPA',<br />
 '-WPA',<br />
 '+WPA',
 'RE24',
 'REW',
 'pLI',
 'inLI',
 'gmLI',
 'exLI',
 'Pulls',
 'Games',
 'WPA/LI',
 'Clutch',
 'SD',
 'MD',
 'FB%1',
 'FBv',
 'CT%',
 'CTv',
 'CB%',
 'CBv',
 'CH%',
 'CHv',
 'XX%',
 'wFB',
 'wCT',
 'wCB',
 'wCH',
 'wFB/C',
 'wCT/C',
 'wCB/C',
 'wCH/C',
 'O-Swing%',
 'Z-Swing%',
 'Swing%',
 'O-Contact%',
 'Z-Contact%',
 'Contact%',
 'Zone%',
 'F-Strike%',
 'SwStr%',
 'Pull',
 'Cent',
 'Oppo',
 'Soft',
 'Med',
 'Hard',
 'bipCount',
 'Pull%',
 'Cent%',
 'Oppo%',
 'Soft%',
 'Med%',
 'Hard%',
 'tERA',
 'pfxFA%',
 'pfxFC%',
 'pfxSI%',
 'pfxKC%',
 'pfxCH%',
 'pfxvFA',
 'pfxvFC',
 'pfxvSI',
 'pfxvKC',
 'pfxvCH',
 'pfxFA-X',
 'pfxFC-X',
 'pfxSI-X',
 'pfxKC-X',
 'pfxCH-X',
 'pfxFA-Z',
 'pfxFC-Z',
 'pfxSI-Z',
 'pfxKC-Z',
 'pfxCH-Z',
 'pfxwFA',
 'pfxwFC',
 'pfxwSI',
 'pfxwKC',
 'pfxwCH',
 'pfxwFA/C',
 'pfxwFC/C',
 'pfxwSI/C',
 'pfxwKC/C',
 'pfxwCH/C',
 'pfxO-Swing%',
 'pfxZ-Swing%',
 'pfxSwing%',
 'pfxO-Contact%',
 'pfxZ-Contact%',
 'pfxContact%',
 'pfxZone%',
 'pfxPace',
 'piCH%',
 'piCU%',
 'piFA%',
 'piFC%',
 'piSI%',
 'pivCH',
 'pivCU',
 'pivFA',
 'pivFC',
 'pivSI',
 'piCH-X',
 'piCU-X',
 'piFA-X',
 'piFC-X',
 'piSI-X',
 'piCH-Z',
 'piCU-Z',
 'piFA-Z',
 'piFC-Z',
 'piSI-Z',
 'piwCH',
 'piwCU',
 'piwFA',
 'piwFC',
 'piwSI',
 'piwCH/C',
 'piwCU/C',
 'piwFA/C',
 'piwFC/C',
 'piwSI/C',
 'piO-Swing%',
 'piZ-Swing%',
 'piSwing%',
 'piO-Contact%',
 'piZ-Contact%',
 'piContact%',
 'piZone%',
 'Events',
 'Barrels',
 'HardHit',
 'gamedate',
 'dh',
 'SF%',
 'SFv',
 'wSF',
 'wSF/C',
 'pfxFS%',
 'pfxSL%',
 'pfxCU%',
 'pfxvFS',
 'pfxvSL',
 'pfxvCU',
 'pfxFS-X',
 'pfxSL-X',
 'pfxCU-X',
 'pfxFS-Z',
 'pfxSL-Z',
 'pfxCU-Z',
 'pfxwFS',
 'pfxwSL',
 'pfxwCU',
 'pfxwFS/C',
 'pfxwSL/C',
 'pfxwCU/C',
 'piFS%',
 'pivFS',
 'piFS-X',
 'piFS-Z',
 'piwFS',
 'piwFS/C',
 'piSL%',
 'pivSL',
 'piSL-X',
 'piSL-Z',
 'piwSL',
 'piwSL/C',
 'SL%',
 'SLv',
 'wSL',
 'wSL/C',
 'GSv2',
 'pfxEP%',
 'pfxvEP',
 'pfxEP-X',
 'pfxEP-Z',
 'pfxwEP',
 'pfxwEP/C',
 'piCS%',
 'pivCS',
 'piCS-X',
 'piCS-Z',
 'piwCS',
 'piwCS/C',
 'piXX%',
 'pivXX',
 'piXX-X',
 'piXX-Z',
 'piwXX',
 'piwXX/C',
 'piSB%',
 'pivSB',
 'piSB-X',
 'piSB-Z',
 'piwSB',
 'piwSB/C',
 'KN%',
 'KNv',
 'wKN',
 'wKN/C',
 'pfxKN%',
 'pfxvKN',
 'pfxKN-X',
 'pfxKN-Z',
 'pfxwKN',
 'pfxwKN/C',
 'piKN%',
 'pivKN',
 'piKN-X',
 'piKN-Z',
 'piwKN',
 'piwKN/C',
 'pfxSC%',
 'pfxvSC',
 'pfxSC-X',
 'pfxSC-Z',
 'pfxwSC',
 'pfxwSC/C',
 'pfxFO%',
 'pfxvFO',
 'pfxFO-X',
 'pfxFO-Z',
 'pfxwFO',
 'pfxwFO/C',
 'EV',
 'LA',
 Barrel%',
 maxEV
 HardHit%
