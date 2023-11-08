import requests
body = {
    
  "open_price": 10000,
  "close_price": 10500,
  "volume": 1000000

    }
response = requests.post(url = 'http://127.0.0.1:8000/predict_price',
              json = body)
print (response.json())
# output: {'score': 0.866490130600765}
