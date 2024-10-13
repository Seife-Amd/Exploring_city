import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt

# Define fuzzy variables (Antecedents) for weather, preference, and time available
weather = ctrl.Antecedent(np.arange(0, 11, 1), 'weather')
preference = ctrl.Antecedent(np.arange(0, 11, 1), 'preference')
time_available = ctrl.Antecedent(np.arange(0, 11, 1), 'time_available')

# Consequent: Recommendation score for activities
recommendation = ctrl.Consequent(np.arange(0, 26, 1), 'recommendation')

# Membership functions for Weather
weather.automf(3)  # Generates 'poor', 'average', 'good'

# Membership functions for Preference (like indoor/outdoor)
preference['indoor'] = fuzz.trimf(preference.universe, [0, 0, 5])
preference['outdoor'] = fuzz.trimf(preference.universe, [5, 10, 10])

# Membership functions for Time Available (short, medium, long)
time_available.automf(3)  # Generates 'poor', 'average', 'good'

# Membership functions for Recommendation (low, medium, high)
recommendation['low'] = fuzz.trimf(recommendation.universe, [0, 0, 13])
recommendation['medium'] = fuzz.trimf(recommendation.universe, [0, 13, 25])
recommendation['high'] = fuzz.trimf(recommendation.universe, [13, 25, 25])

# Fuzzy Rules for recommendation based on conditions
rule1 = ctrl.Rule(weather['good'] & preference['outdoor'] & time_available['good'], recommendation['high'])
rule2 = ctrl.Rule(weather['poor'] & preference['indoor'] & time_available['poor'], recommendation['low'])
rule3 = ctrl.Rule(weather['average'] & preference['indoor'] & time_available['average'], recommendation['medium'])

# Create control system and simulation
recommendation_ctrl = ctrl.ControlSystem([rule1, rule2, rule3])
recommendation_sim = ctrl.ControlSystemSimulation(recommendation_ctrl)

# Input values for a sample case
recommendation_sim.input['weather'] = 8  # Good weather
recommendation_sim.input['preference'] = 7  # Outdoor preference
recommendation_sim.input['time_available'] = 6  # Medium time available

# Compute recommendation
recommendation_sim.compute()

# Print the recommended activity score
print(f"Recommended activity score: {recommendation_sim.output['recommendation']}")

# Visualize the membership functions and the output
recommendation.view(sim=recommendation_sim)
plt.show()

# Defuzzification Example: Apply different defuzzification techniques
x = np.arange(0, 5.05, 0.1)
mfx = fuzz.trapmf(x, [2, 2.5, 3, 4.5])

# Different defuzzification methods
defuzz_centroid = fuzz.defuzz(x, mfx, 'centroid')
defuzz_bisector = fuzz.defuzz(x, mfx, 'bisector')
defuzz_mom = fuzz.defuzz(x, mfx, 'mom')
defuzz_som = fuzz.defuzz(x, mfx, 'som')
defuzz_lom = fuzz.defuzz(x, mfx, 'lom')

# Plot the defuzzification results
labels = ['centroid', 'bisector', 'mean of maximum', 'min of maximum', 'max of maximum']
xvals = [defuzz_centroid, defuzz_bisector, defuzz_mom, defuzz_som, defuzz_lom]
colors = ['r', 'b', 'g', 'c', 'm']
ymax = [fuzz.interp_membership(x, mfx, i) for i in xvals]

plt.figure(figsize=(8, 5))
plt.plot(x, mfx, 'k')
for xv, y, label, color in zip(xvals, ymax, labels, colors):
    plt.vlines(xv, 0, y, label=label, color=color)
plt.ylabel('Fuzzy membership')
plt.xlabel('Universe variable')
plt.ylim(-0.1, 1.1)
plt.legend(loc=2)
plt.show()
