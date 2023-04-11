import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import chisquare

def lin(x, a, b):
    return a*x + b

# nm
WAVELENGTHS = np.array([390, 455, 505, 535, 590, 615, 640, 935])
WAVELENGTHS_UNCERT = np.array([40, 40, 30, 30, 10, 10, 10, 10])

FREQUENCIES = 3*(10**8) / (WAVELENGTHS * 10**-9)
FREQ_UNCERT = (WAVELENGTHS_UNCERT/WAVELENGTHS) * (FREQUENCIES)
e = 1.602176633 * 10**-19

################### Experiment 1 ###################
# Uncertainty for all oscilloscope measurements: 0.0001 V

stopping_v = np.array([1.5078, 1.2434, 0.9190, 0.7972, 0.5891, 0.5117, 0.4936, 0.4633])
stopping_v_uncert = 0.0001

# Plot stopping voltage v. frequency
popt, pcov = curve_fit(lin, FREQUENCIES, stopping_v)
x_pts = np.linspace(FREQUENCIES[0], FREQUENCIES[-1], 100)

plt.figure()
plt.plot(x_pts, lin(x_pts, popt[0], popt[1]), label="Curve of Best Fit") # fit
plt.errorbar(FREQUENCIES, stopping_v, xerr=FREQ_UNCERT, yerr=stopping_v_uncert, 
             color="orange", fmt="None") # uncert
plt.scatter(FREQUENCIES, stopping_v, label="Data Points", color="orange") # data
plt.legend()
plt.xlabel("Frequency (Hz)")
plt.ylabel("Stopping Voltage (V)")
plt.title("Stopping Voltage v. Frequency")

#plt.show()

slope, slope_uncert = popt[0], np.sqrt(pcov[0][0])
inter, inter_uncert = popt[1], np.sqrt(pcov[1][1])
thresh_freq = np.average(FREQUENCIES - (stopping_v/(slope/e)))
thresh_freq_uncert = thresh_freq * np.average(
    np.sqrt( (FREQ_UNCERT/FREQUENCIES)**2 + (stopping_v_uncert/stopping_v)**2 ))

print(f"The experimental value of Planck's constant is: {slope}+-{slope_uncert}")
print(f"The experimental value of the work function is: {inter}+-{inter_uncert}")
print(f"The threshold frequency is {thresh_freq}+-{thresh_freq_uncert}")


# Plot fit residuals
plt.figure()
plt.scatter(FREQUENCIES, stopping_v - lin(FREQUENCIES, popt[0], popt[1]))
plt.axhline(0, color='black', linewidth=.5)
plt.xlabel("Frequency (Hz)")
plt.ylabel("Stopping Voltage (V)")
plt.title("Stopping Voltage v. Frequency Residuals")
#plt.show()


# Calculate Reduced Chi Square
chi_square = np.sum((stopping_v - lin(FREQUENCIES, popt[0], popt[1]))**2) / np.var(stopping_v)
reduced_chi_square = chi_square/len(stopping_v)

print(f"The Reduced Chi-Square is {reduced_chi_square}")


################### Experiment 2 ###################
stopping_v_2 = [0.9789, 0.9773, 0.9759, 0.9775]
photocurrent = [1.4391, 1.7482, 2.3066, 3.8875]

# Plot stopping voltage v. intensity
plt.figure()
plt.errorbar([1, 2, 3, 4], stopping_v_2, yerr=stopping_v_uncert, fmt="None") # uncert
plt.scatter([1, 2, 3, 4], stopping_v_2)
plt.xlabel("Intensity")
plt.ylabel("Stopping Voltage (V)")
plt.title("Stopping Voltage v. Intensity")


# Plot photocurrent v. intensity
plt.figure()
plt.errorbar([1, 2, 3, 4], photocurrent, yerr=stopping_v_uncert, fmt="None") # uncert
plt.scatter([1, 2, 3, 4], photocurrent)
plt.xlabel("Intensity")
plt.ylabel("Photocurrent (V)")
plt.title("Photocurrent v. Intensity")

plt.show()

################### Experiment 3 ###################
Pe = 60 * 10**(-3) * ((0.3 * 10**(-9))**2 / (3.23 * (0.01)**2))
time_delay = (inter*e) / Pe 
time_delay_uncert = time_delay * (inter_uncert/inter)

print(Pe, inter)
print(f"The time delay is {time_delay * -1}+-{time_delay_uncert}")
