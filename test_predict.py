import requests

url = "http://127.0.0.1:5000/predict"
data = {
    "state_enc": 1,
    "district_enc": 2,
    "city_enc": 3,
    "crime_type_enc": 0,
    "year": 2022,
    "no_of_police_station": 50,
    "literacy_rate_in_%": 70,
    "area_in_kilometer_square": 1500,
    "population": 2000000
}

response = requests.post(url, json=data)
print(response.json())