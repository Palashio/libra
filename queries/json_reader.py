import json

data = {}

data["tree"] = "green"

with open('person.txt', 'w') as json_file:
  json.dump(data, json_file)