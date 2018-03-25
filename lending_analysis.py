import numpy as np	# for basic algebra
import matplotlib.pyplot as plt   # for Plots
import price_prediction_utils as ppu	# Helper Functions to fetch data
from sklearn import linear_model, svm, ensemble	# Machine Learning Algorithms


data = pd.read_csv('prosper_data.csv')

