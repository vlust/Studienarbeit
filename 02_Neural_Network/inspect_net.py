import pickle
tuner = pickle.load(open("tuner_1650605663.pkl","rb"))
tuner.get_best_hyperparameters()[0].values
tuner.get_best_models()[0].summary()
tuner.results_summary()