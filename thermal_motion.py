# Import libraries
import glob
import scipy
import numpy as np
import matplotlib.pyplot as plt

# Constants
d_bead = 1.9 * 10**(-6)      # meter
unc_bead = 0.1 * 10**(-6)    # meter

temperature = 296.5                    # kelvin
diff_temperature = temperature - (293.15)    # kelvin
unc_temperature = 0.5                  # kelvin

visc = (1 * (1 - (0.02 * diff_temperature))) * 10**(-3) # Pa*s
unc_visc = ((0.05/1)**2 + (0.5/296.5)**2)**(1/2) * visc # Pa*s

pixel_meters = 0.1155 * 10**(-6)   # m/pixel
unc_pos = 10**(-1) * 10**(-6)      # m

unc_time = 0.03 # seconds

actual_k = 1.38 * 10**(-23) # J/K


# Functions
def conv_dist(x_pos, y_pos):
    conv_x = (np.array(x_pos)*pixel_meters).tolist()
    conv_y = (np.array(y_pos)*pixel_meters).tolist()

    return conv_x, conv_y


def read_positions(data):
    x_pos, y_pos = [], []
    for i in range(len(data)-2):
        temp = data[2+i][0:-2].split("\t")
        x_pos.append(float(temp[0]))
        y_pos.append(float(temp[1]))

    return x_pos, y_pos


def calc_step_sizes(x_pos, y_pos):
    sizes = []
    unc_sizes = []
    for i in range(len(x_pos)-1):
        size = ((x_pos[1+i]-x_pos[i])**(2) + (y_pos[1+i]-y_pos[i])**(2))**(1/2)

        sizes.append(size)
        unc_sizes.append((2*((x_pos[1+i]-x_pos[i])*unc_pos*pixel_meters/size)**2 \
                          + 2*((y_pos[1+i]-y_pos[i])*unc_pos*pixel_meters/size)**2))

    return sizes, unc_sizes


def diff_2d_pdf(r, D):
    t = 0.5 # 2 frames per second
    return (r/(20000000*D*t))*np.exp(-(r**2)/(4*D*t))


def calc_msd(x_pos, y_pos):
    msd = []
    unc_msd = []
    for i in range(len(x_pos)):
        msd.append((x_pos[i]-x_pos[0])**2 + (y_pos[i] - y_pos[0])**2)
        unc_msd.append(2*unc_pos*(2*((x_pos[i]-x_pos[0])**2 + (y_pos[i] - y_pos[0])**2))**(1/2))

    return msd, unc_msd


def msd_lin(t, D):
    return 4*D*t


def calc_k(D, unc_D):
    gamma = 6 * np.pi * visc * d_bead/2
    k = (D*gamma) / temperature
    unc_k = k * ((unc_visc/visc)**2 + (unc_bead/d_bead)**2 \
                 + (unc_temperature/temperature)**2 + (unc_D/D)**2)**(1/2)

    return k, unc_k

def calc_p_diff(experimental_k):
    return (abs(experimental_k - actual_k) / ((experimental_k + actual_k)/2)) * 100


def graph_residuals(x_coords, data, err_data, fit, approach, x_label):
    plt.figure()
    plt.scatter(x_coords, np.array(data)-np.array(fit))
    if "time" in x_label.lower():
        plt.errorbar(x_coords, np.array(data)-np.array(fit), xerr=unc_time, yerr=err_data, \
                     fmt='None', ecolor="red", zorder=0)
    else:
        print(err_data)
        plt.errorbar(x_coords, np.array(data)-np.array(fit), yerr=err_data, fmt='None', \
                     ecolor="red", zorder=0)
    plt.axhline(0, color='black', linewidth=.5)
    plt.title(f"{approach} Approach Residuals")
    plt.xlabel(x_label)
    plt.ylabel("Residual Values")


def calc_chi_squared(data, fit):
    chi_squared = np.sum((((np.array(data)-np.array(fit))**2) / fit)[1:]) # first value bad
    reduced_chi_squared = chi_squared/len(data)

    return chi_squared, reduced_chi_squared


if __name__ == "__main__":
    data_paths = glob.glob("./raw_data/good_data/**/*.txt", recursive=True)
    data = []
    unc_data = []
    mean_squared_dist = []
    unc_msd = []

    # Process step sizes into 1D array
    for file in data_paths:
        with open(file) as f:
            lines = f.readlines()
            
            # First 2 lines are filler
            # Data is stored: "xval\tyval\n"
            
            # Produce list of x and y positions
            x_pos, y_pos = read_positions(lines)
            x_pos, y_pos = conv_dist(x_pos, y_pos)

            # Calculate step sizes
            temp, unc_temp = calc_step_sizes(x_pos, y_pos)

            # Append to data
            data += temp
            unc_data += unc_temp

            # Calculate mean squared distance
            msd_temp, unc_msd_temp = calc_msd(x_pos, y_pos)
            mean_squared_dist.append(msd_temp)
            unc_msd.append(unc_msd_temp)

    data = np.array(data)
    unc_data = np.array(unc_data)
    raw_data = data.copy()

    # Fit Rayleigh curve
    counts, bins = np.histogram(data, bins=50)
    popt, pcov = scipy.optimize.curve_fit(diff_2d_pdf, \
                                          np.array(bins[:-1]), \
                                            np.array(counts)/sum(counts),
                                            p0 = [0.0000000000000003])
    D_curve, unc_D_curve = popt[0], *((np.diag(pcov))**(1/2))
    k_curve, unc_k_curve = calc_k(D_curve, unc_D_curve)
    cs_curve, rcs_curve = calc_chi_squared((np.array(counts)/(sum(counts))),\
                                           np.vectorize(diff_2d_pdf)(bins[:-1], D_curve))
    std_dev_1_2 = np.std(np.array(counts)/(sum(counts)))

    print(f"The value of D from fitting the curve is: {D_curve}")
    print(f"The uncertainty of D from fitting the curve is: {unc_D_curve}")
    print(f"The curve fit calculation for k is: {k_curve} +- {unc_k_curve}")
    print(f"The % difference in the k value is: {calc_p_diff(k_curve)} +- \
{(unc_k_curve/k_curve)*(2**(1/2))*calc_p_diff(k_curve)}")
    print(f"The chi squared and reduced chi squared is: {cs_curve} and {rcs_curve}\n")

    # Plot histogram
    plt.figure()
    plt.stairs(np.array(counts)/(sum(counts)), bins, label="Measured step size")
    plt.errorbar((np.array(bins[:-1]+((bins[1]-bins[0])/2))), np.array(counts)/(sum(counts)), \
                 yerr=std_dev_1_2, fmt="None", ecolor="red", zorder=0)
    plt.plot(bins, np.vectorize(diff_2d_pdf)(bins, D_curve), label="Curve Fit")
    plt.title("Rayleigh Distribution Fitting Approach")
    plt.xlabel("Step Size (m)")
    plt.ylabel("Probability (P(x, 0.5s))")
    plt.legend()

    # Plot residuals
    graph_residuals(bins[:-1], np.array(counts)/(sum(counts)), std_dev_1_2,\
               np.vectorize(diff_2d_pdf)(bins[:-1], D_curve), "Rayleigh Distribution Fitting", "Step Size (m)")


    # Maximum Liklihood
    t_max = 60
    D_max = np.sum((np.array(raw_data))**2) /(4*len(raw_data)*t_max)
    unc_D_max = (((np.sum(np.array(unc_data)**2))**(1/2) \
                  /np.sum(np.array(raw_data)**2))**(2) + \
                    (unc_time/t_max)**(2))**(1/2) * D_max
    k_max, unc_k_max = calc_k(D_max, unc_D_max)
    cs_max, rcs_max = calc_chi_squared((np.array(counts)/(sum(counts))),\
                                           np.vectorize(diff_2d_pdf)(bins[:-1], D_max))
    
    print(f"The value of D from maximum likelihood is: {D_max}")
    print(f"The uncertainty of D from maximum likelihood is: {unc_D_max}")
    print(f"The maximum likelihood calculation for k is: {k_max} +- {unc_k_max}")
    print(f"The % difference in the k value is: {calc_p_diff(k_max)} +- \
{(unc_k_max/k_max)*(2**(1/2))*calc_p_diff(k_max)}")
    print(f"The chi squared and reduced chi squared is: {cs_max} and {rcs_max}\n")
    
    plt.figure()
    plt.stairs(np.array(counts)/(sum(counts)), bins, label="Measured step size")
    plt.errorbar((np.array(bins[:-1]+((bins[1]-bins[0])/2))), np.array(counts)/(sum(counts)), \
                 yerr=std_dev_1_2, fmt="None", ecolor="red", zorder=0)
    plt.plot(bins, np.vectorize(diff_2d_pdf)(bins, D_max), label="Calculated Curve")
    plt.title("Maximum Likelihood Approach")
    plt.xlabel("Step Size (m)")
    plt.ylabel("Probability (P(x, 0.5s))")
    plt.legend()
    
    # Plot residuals
    graph_residuals(bins[:-1], np.array(counts)/(sum(counts)), std_dev_1_2,\
                    np.vectorize(diff_2d_pdf)(bins[:-1], D_max), "Maximum Likelihood", "Step Size (m)")
          
    
    # Fit mean square distance
    avg_msd = np.sum(mean_squared_dist, axis=0)/len(mean_squared_dist)
    avg_unc_msd = np.sum(unc_msd, axis=0)/len(unc_msd)
    popt, pcov = scipy.optimize.curve_fit(msd_lin, \
                                          np.arange(0, 60, 0.5), \
                                            np.array(avg_msd))
    D_msd, unc_D_msd = popt[0], *(np.diag(pcov))**(1/2)
    k_msd, unc_k_msd = calc_k(D_msd, unc_D_msd)
    cs_msd, rcs_msd = calc_chi_squared(avg_msd, \
                                       np.vectorize(msd_lin)(np.arange(0, 60, 0.5), D_msd))
    
    print(f"The value of D from fitting the mean square distance is: {D_msd}")
    print(f"The uncertainty of D from fitting the mean square distance is: {unc_D_msd}")
    print(f"The mean square distance calculation for k is: {k_msd} +- {unc_k_msd}")
    print(f"The % difference in the k value is: {calc_p_diff(k_msd)} +- \
{(unc_k_msd/k_msd)*(2**(1/2))*calc_p_diff(k_msd)}")
    print(f"The chi squared and reduced chi squared is: {cs_msd} and {rcs_msd}\n")

    # Plot linear fit
    plt.figure()
    plt.scatter(np.arange(0, 60, 0.5), avg_msd, label="Measured Mean Square Distance")
    plt.plot(np.arange(0, 60, 0.5), np.vectorize(msd_lin)(np.arange(0, 60, 0.5), D_msd), \
             label="Linear Fit", color="orange")
    plt.errorbar(np.arange(0, 60, 0.5), avg_msd, yerr=avg_unc_msd, fmt="None", ecolor="red", zorder=0)
    plt.title("Mean Square Distance Approach")
    plt.ylabel("Mean Square Distance (m^2)")
    plt.xlabel("Time (s)")
    plt.legend()

    # Plot residuals
    graph_residuals(np.arange(0, 60, 0.5), avg_msd, avg_unc_msd, \
                    np.vectorize(msd_lin)(np.arange(0, 60, 0.5), D_msd), "Mean Square Distance", "Time (s)")


    plt.show()
