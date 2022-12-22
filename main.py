import prediction.prediction
import vegetation.vegetation
import inquirer

while True:
    questions = [
        inquirer.List('choice',
                      message="Choose Country?",
                      choices=['bolivia', 'china', 'exit'],
                      ),
    ]
    answers = inquirer.prompt(questions)
    country = answers['choice']
    if country == 'exit':
        break
    questions = [
        inquirer.List('choice',
                      message="Choose Action?",
                      choices=['find vegetation', 'show vegetation', 'predict', 'create model'],
                      ),
    ]
    answers = inquirer.prompt(questions)

    action = answers['choice']
    if action == "find vegetation":
        vegetation.vegetation.getvegetation(country)
    if action == "show vegetation":
        vegetation.vegetation.createOverlay(country)
    if action =="predict":
        imgs = vegetation.vegetation.convertImgsToArrays(country)
        img = imgs[len(imgs)-1]
        prediction.prediction.predictFutureVegetation(img, country)
    if action == "create model":
        prediction.prediction.teachModel(vegetation.vegetation.convertImgsToArrays(country),country)

