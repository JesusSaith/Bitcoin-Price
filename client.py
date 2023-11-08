import requests
body = {
    
    "open_price": 1007700,
  "close_price": 1077500,
  "volume": 10884400
 }
response = requests.post(url = 'https://bitcoin-price-service-jesussaith.cloud.okteto.net/predict_price',
              json = body)
print (response.json())
# output: {'score': 0.866490130600765}
