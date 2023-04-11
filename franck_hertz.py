import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks


def read_positions(data):
    x_pos, y_pos = [], []
    for i in range(len(data)-2):
        temp = data[2+i][0:-2].split("\t")
        x_pos.append(float(temp[0]))
        y_pos.append(float(temp[1]))

    #plt.plot(x_pos, y_pos)
    return x_pos, y_pos


def graph_residuals(x_coords, data, err_data, fit, x_label):
    plt.figure()
    plt.scatter(x_coords, np.array(data)-np.array(fit))
    plt.errorbar(x_coords, np.array(data)-np.array(fit), yerr=err_data, fmt='None', \
                    ecolor="red", zorder=0)
    plt.axhline(0, color='black', linewidth=.5)
    plt.title(f"Residuals of Fit")
    plt.xlabel(x_label)
    plt.ylabel("Residual Values")


def calc_chi_squared(data, fit):
    chi_squared = np.sum((((np.array(data)-np.array(fit))**2) / np.array(fit)))
    reduced_chi_squared = chi_squared/len(data)

    return chi_squared, reduced_chi_squared


if __name__ == "__main__":
    files = os.listdir("./Data")

    # Create graphs of all data
    """
    for i in files:
        with open(f"./Data/{i}") as f:
            lines = f.readlines()
            
            # First 2 lines are filler
            # Data is stored: "xval\tyval\n"
            
            # Produce list of x and y positions
            ch1, ch2 = read_positions(lines)

            
            plt.figure()
            plt.plot(ch1, ch2)
            plt.title(i)
            plt.xlabel("Voltage (V)")
            plt.ylabel("Current (mA)")

            plt.savefig(f"./Graphs/{i[0:-4]}.png")
    """

    
    # Find local max and min of chosen trial
    chosen = "./Data/E2(1.5).2.txt"
    with open(chosen) as f:
        lines = f.readlines()

        ch1, ch2 = read_positions(lines)


    ch1, ch2 = np.array(ch1), np.array(ch2)

    """ Tried to be cool, didn't work
    max_idx = find_peaks(ch2, threshold=0.01)
    min_idx = find_peaks(ch2*-1, threshold=0.01)
    """

    max_pts = np.array([[7.19862, 11.75827, 16.59899, 21.67395, 26.88943], 
               [0.137695, 0.230469, 0.347650, 0.446289, 0.531789]])
    min_pts = np.array([[8.69768, 13.77264, 18.98632, 24.21923], 
               [0.131582, 0.186035, 0.261719, 0.330078]])

    plt.figure()
    plt.plot(ch1, ch2, label="Data", zorder=1)
    plt.scatter(max_pts[0], max_pts[1], 
                label="Local Maximums", zorder=2)
    plt.scatter(min_pts[0], min_pts[1], 
                label="Local Minimums", zorder=2)

    plt.title("Accelerating Voltage (E3) v. Current Voltage ")
    plt.xlabel("Accelerating Voltage (V)")
    plt.ylabel("Current Voltage (V)")
    plt.legend()
    plt.show()

    # Plot fit of differences + fit (# last point omitted bc no uncertainty)
    plt.figure()
    plt.plot(fit_x, fit_y, label="Curve of best fit")
    plt.scatter([1, 2, 3, 4], max_pts[0][:-1], label="Accelerating Voltage")
    plt.errorbar([1, 2, 3, 4], max_pts[0][:-1], np.abs(max_pts[0][:-1] - min_pts[0]), fmt='None')

    plt.title("Accelerating Voltage v. Peak Number")
    plt.xlabel("Peak Number")
    plt.ylabel("Accelerating Voltage (V)")
    plt.legend(loc="upper left")
    plt.show()

    # Plot residuals
    graph_residuals([1, 2, 3, 4], max_pts[0][:-1], np.abs(max_pts[0][:-1] - min_pts[0]),
                    fit_y, "Peak Number")
    plt.show()

    # Energy Transfer
    energy = np.average(np.diff(max_pts[0]))
    energy_unc = np.average(np.abs(max_pts[0][:-1] - min_pts[0]))
    print(f"The energy transfer is {energy} +- {energy_unc} eV.")

    # Wavelength
    wavelength = ((4.135667516*10**(-15)) * (3*(10**8))) / energy
    wavelength_unc = (energy_unc / energy) * wavelength
    print(f"This corresponds to a wavelength of {wavelength} +- {wavelength_unc} m.")

    print(np.diff(max_pts[0]))
    print(np.abs(max_pts[0][:-1] - min_pts[0]))

    
