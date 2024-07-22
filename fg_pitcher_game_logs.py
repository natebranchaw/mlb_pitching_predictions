import pandas as pd
import requests
import json

def fg_pitcher_game_logs(player_id, year):  
    url = 'https://www.fangraphs.com/api/players/game-log?playerid={}&position=P&type=0&gds=&gde=&z=1703085978&season={}'.format(player_id, year)
    
    response = requests.get(url)
    json_data = json.loads(response.text)
    
    if json_data is None:
        print('No data available for Player ID:', player_id)
    else:
        data = json_data['mlb']
        if (data is None) or (len(data) == 0):
            print('No data available for Player ID:', player_id)
        else:
            df = pd.DataFrame(data)
            df['Date'] = df['Date'].str.extract(r'(?<=\>)(.*?)(?=\<)')
            df = df[['PlayerName', 'playerid'] + [col for col in df.columns if col not in ['PlayerName', 'playerid']]]
            return df
