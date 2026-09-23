import requests
import pandas as pd

#Beisbol
#Principales estadios: Yankee Stadium, Citi Field


url = "https://statsapi.mlb.com/api/v1/schedule?sportId=1&season=2023&hydrate=gameInfo"
response = requests.get(url).json()

target_venues = ["Yankee Stadium", "Citi Field"]

games_data = []

for date in response['dates']:
    for game in date['games']:
        
        venue_name = game['venue']['name']
        
        if venue_name in target_venues:
            
            start_utc = pd.to_datetime(game['gameDate'], utc=True)
            start_ny = start_utc.tz_convert('America/New_York')
            
            duration = game.get('gameInfo', {}).get('gameDurationMinutes', None)
            
            if duration:
                end_ny = start_ny + pd.Timedelta(minutes=duration)
            else:
                end_ny = None

            games_data.append({
                "venue": venue_name,
                "start_time_ny": start_ny,
                "end_time_ny": end_ny
            }) 

mlb_ny_stadiums_2023 = pd.DataFrame(games_data).sort_values("start_time_ny")

mlb_ny_stadiums_2023.to_csv("mlb_games_ny_stadiums_2023.csv", index=False)

mlb_ny_stadiums_2023.head()

