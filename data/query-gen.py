requests = [
    "I want to", "I need to", "Need to", "I'd like to", "Want to", "Please"
]
actions = [
    "model", "estimate", "predict", "forecast", "project",
]
stats = [
    "", "average ", "median ", "mean "
]
fields = [
    "number", "price", "rate"
]
subjects = {
    "number": [
        "pregnancies", "car crashes", "criminal incidents", "trees", "power failures", "wild dogs",
        "students", "professors", "lawyers", "doctors", "firefighters", "police officers", "voters",
        "teachers", "businesses", "restaurants", "people"

    ],
    "price": [
        "cars", "cats", "dogs", "candy", "steak", "beef", "pancakes", "sugar", "bread",
        "water", "houses", "couches", "sofas", "beds", "furniture", "TVs", "cheese", "game consoles",
        "PC parts", "pencils", "pens", "erasers", "food", "computers", "laptops", "gas", "electricity"
    ],
    "rate": [
        "poverty", "suicide", "college acceptance", "voter participation", "murder", "crime",
        "home ownership", "drug use", "business failure", "GDP growth", "productivity growth",
        "population growth", "infant mortality"
    ],

    "NA": [
        "weather", "temperature",
    ]
}
n = 0
final = True
if final:
    separator = "\t"
else:
    separator = "\t~ "

with open("text_data.txt", "w+") as file:
    for request in requests: # "i want to"
        for action in actions: # "predict"
            for stat in stats: # "average"
                for field in fields: # "number of"
                    for subjectType in subjects.keys(): # "items"
                        if field == subjectType:
                            for subject in subjects[subjectType]:
                                target = stat + field + " of " + subject

                                file.write(request + " " + action + " the " + target + separator +
                                           target + separator + "!\n")

                                n += 1

            for subjectType in subjects.keys():
                for subject in subjects[subjectType]:
                    file.write(request + " " + action + " " + subject + separator +
                               subject + separator + "!\n")
                    n += 1


print("done; created " + str(n) + " sentences")