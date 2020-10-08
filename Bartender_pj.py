questions = {
    "strong": "Do ye like yer drinks strong?",
    "salty": "Do ye like it with a salty tang?",
    "bitter": "Are ye a lubber who likes it bitter?",
    "sweet": "Would ye like a bit of sweetness with yer poison?",
    "fruity": "Are ye one for a fruity finish?",
}
ingredients = {
    "strong": ["glug of rum", "slug of whisky", "splash of gin"],
    "salty": ["olive on a stick", "salt-dusted rim", "rasher of bacon"],
    "bitter": ["shake of bitters", "splash of tonic", "twist of lemon peel"],
    "sweet": ["sugar cube", "spoonful of honey", "spash of cola"],
    "fruity": ["slice of orange", "dash of cassis", "cherry on top"],
}
import random
ans = ''
x = ''
while (ans != 'YES'):
    x = random.choice(list(questions.keys()))
    print(questions[x])
    ans = input()
ans = ''
while (ans != 'YES'):
    y = random.choice(list(ingredients[x]))
    print("Do you want a " + y +"?")
    ans = input()
print("OK!")