import numpy as np
import matplotlib.pyplot as plt
import pywt

# Set random seed for reproducibility
np.random.seed(42)

def wavelet_denoising(data, wavelet='db4', level=1):
    coeff = pywt.wavedec(data, wavelet, mode="per")
    sigma = (1/0.6745) * np.median(np.abs(coeff[-level] - np.median(coeff[-level])))
    uthresh = sigma * np.sqrt(2 * np.log(len(data)))
    coeff[1:] = [pywt.threshold(i, value=uthresh, mode='soft') for i in coeff[1:]]
    return pywt.waverec(coeff, wavelet, mode='per')

# Create time axis
t = np.linspace(0, 10, 500)
noise = np.random.normal(0, 0.2, 500)

# 1. Point Anomaly (Spike)
spike_data = np.zeros(500)
spike_data[250] = 5.0
spike_noisy = spike_data + noise
spike_denoised = wavelet_denoising(spike_noisy, wavelet='db4', level=3)

# 2. Level Shift
shift_data = np.where(t < 5, 0, 3)
shift_noisy = shift_data + noise
shift_denoised = wavelet_denoising(shift_noisy, wavelet='db4', level=3)

# 3. Trend
trend_data = 0.5 * t
trend_noisy = trend_data + noise
trend_denoised = wavelet_denoising(trend_noisy, wavelet='db4', level=3)

# 4. Seasonal
seasonal_data = 2 * np.sin(2 * np.pi * 0.5 * t)
seasonal_noisy = seasonal_data + noise
seasonal_denoised = wavelet_denoising(seasonal_noisy, wavelet='db4', level=3)

# Plotting
fig, axs = plt.subplots(2, 2, figsize=(14, 10))
plt.subplots_adjust(hspace=0.4, wspace=0.3)

# Plot 1: Spike
axs[0, 0].plot(t, spike_noisy, color='lightblue', alpha=0.7, label='Noisy (Spike)')
axs[0, 0].plot(t, spike_denoised, color='red', linewidth=2, label='Denoised')
axs[0, 0].set_title('Point Anomaly (Spike) - Significantly Flattened')
axs[0, 0].legend()

# Plot 2: Level Shift
axs[0, 1].plot(t, shift_noisy, color='lightblue', alpha=0.7, label='Noisy (Level Shift)')
axs[0, 1].plot(t, shift_denoised, color='red', linewidth=2, label='Denoised')
axs[0, 1].set_title('Level Shift - Rounded Edges, Level Maintained')
axs[0, 1].legend()

# Plot 3: Trend
axs[1, 0].plot(t, trend_noisy, color='lightblue', alpha=0.7, label='Noisy (Trend)')
axs[1, 0].plot(t, trend_denoised, color='red', linewidth=2, label='Denoised')
axs[1, 0].set_title('Trend Change - Perfectly Preserved')
axs[1, 0].legend()

# Plot 4: Seasonal
axs[1, 1].plot(t, seasonal_noisy, color='lightblue', alpha=0.7, label='Noisy (Seasonal)')
axs[1, 1].plot(t, seasonal_denoised, color='red', linewidth=2, label='Denoised')
axs[1, 1].set_title('Seasonal Pattern - Pattern Retained, Noise Removed')
axs[1, 1].legend()

plt.tight_layout()
plt.savefig('wavelet_denoising_effects.png')
plt.show()