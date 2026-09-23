import requests
import pandas as pd


#Sacamos eventos con permisos de la ciudad de Nueva York. Nos es imposible saber que eventos tienen qué afluencia, pero la localizacion y datos nos permitirán comparar con afluencia de taxis o, en caso de
#que se corte la circulación, la falta de estos

#HACE FALTA UNA APP_KEY

BASE_URL = "https://data.cityofnewyork.us/resource/tvpp-9vvx.json"
APP_TOKEN = "APP_KEY" 

headers = {"X-App-Token": APP_TOKEN}

all_data = []
limit = 50000
offset = 0


while True:
    params = {
        "$limit": limit,
        "$offset": offset
    }
    response = requests.get(BASE_URL, params=params, headers=headers)

    if response.status_code != 200:
        print("Error:", response.status_code, response.text)
        break

    data = response.json()
    if not data:
        break

    all_data.extend(data)
    print(f"Descargados {len(all_data)} eventos...")
    offset += limit

print("Total eventos descargados:", len(all_data))

if len(all_data) == 0:
    print("No se encontraron eventos.")
    exit()

df = pd.DataFrame(all_data)

df["start_date_time"] = pd.to_datetime(df["start_date_time"], errors="coerce")
df["end_date_time"] = pd.to_datetime(df["end_date_time"], errors="coerce")


output_file = "NYC_SAPO_Events_ALL.csv"
df.to_csv(output_file, index=False)
print(f"Archivo guardado como {output_file}")
