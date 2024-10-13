from __future__ import division, print_function
import numpy as np
import matplotlib.pyplot as plt
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# Step 1: Fuzzy C-Means Clustering for City Activities

# Example activity data for clustering (weather, preference, time available)
activity_data = np.array([
    [8, 7, 6],  # Good weather, outdoor preference, medium time
    [2, 3, 4],  # Poor weather, indoor preference, short time
    [5, 5, 6],  # Average weather, mixed preference, medium time
    [9, 9, 8],  # Great weather, outdoor preference, long time
    [3, 2, 5],  # Poor weather, indoor preference, medium time
])

# Run fuzzy c-means clustering
cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(activity_data.T, 3, 2, error=0.005, maxiter=1000)

# New user input to predict cluster
new_user_data = np.array([[8, 7, 6]])  # Good weather, outdoor preference, medium time
u_predicted, _, _, _, _, _ = fuzz.cluster.cmeans_predict(new_user_data.T, cntr, 2, error=0.005, maxiter=1000)

# Find the cluster with the highest membership
predicted_cluster = np.argmax(u_predicted, axis=0)
print(f"Predicted activity cluster: {predicted_cluster}")

# Step 2: Fuzzy Logic-Based Recommendation System

# Define fuzzy variables for recommendation system (weather, preference, time available)
weather = ctrl.Antecedent(np.arange(0, 11, 1), 'weather')
preference = ctrl.Antecedent(np.arange(0, 11, 1), 'preference')
time_available = ctrl.Antecedent(np.arange(0, 11, 1), 'time_available')
recommendation = ctrl.Consequent(np.arange(0, 26, 1), 'recommendation')

# Membership functions
weather.automf(3)  # poor, average, good
preference['indoor'] = fuzz.trimf(preference.universe, [0, 0, 5])
preference['outdoor'] = fuzz.trimf(preference.universe, [5, 10, 10])

# Fix for time_available: Replace the incorrect 'long' with the correct automf labels: 'poor', 'average', 'good'
time_available.automf(3)  # Generates 'poor', 'average', 'good'

# Membership functions for recommendation
recommendation['low'] = fuzz.trimf(recommendation.universe, [0, 0, 13])
recommendation['medium'] = fuzz.trimf(recommendation.universe, [0, 13, 25])
recommendation['high'] = fuzz.trimf(recommendation.universe, [13, 25, 25])

# Define rules based on fuzzy logic (adjust for proper time_available values)
rule1 = ctrl.Rule(weather['good'] & preference['outdoor'] & time_available['good'], recommendation['high'])
rule2 = ctrl.Rule(weather['poor'] & preference['indoor'] & time_available['poor'], recommendation['low'])
rule3 = ctrl.Rule(weather['average'] & preference['indoor'] & time_available['average'], recommendation['medium'])

# Create and simulate control system
recommendation_ctrl = ctrl.ControlSystem([rule1, rule2, rule3])
recommendation_sim = ctrl.ControlSystemSimulation(recommendation_ctrl)

# Provide input values (good weather, outdoor preference, medium time available)
recommendation_sim.input['weather'] = 8
recommendation_sim.input['preference'] = 7
recommendation_sim.input['time_available'] = 6  # 'Good' as per automf membership

# Compute the recommendation
recommendation_sim.compute()

# Output the recommendation
print(f"Recommended activity score: {recommendation_sim.output['recommendation']}")

# Visualize the recommendation result
recommendation.view(sim=recommendation_sim)
plt.show()
