import adi
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.signal import decimate, chirp

""" Pluto Configurations """
sdr = adi.ad9361(uri='ip:192.168.2.1')
samp_rate = 30.72e6  # must be <=30.72 MHz
rx_lo = 2.4e9  # 2.4 GHz
rx_mode = "fast_attack"
rx_gain0 = 70
rx_gain1 = 70
tx_lo = rx_lo
tx_gain0 = -10

""" Radar Parameters """
T_chirp = 1e-3  # chirp süresi (1ms)
num_samps = int(T_chirp * samp_rate) # Tek chirp için örnek sayısı
num_chirps = 32
B = 12e6
k = B / T_chirp
fs = samp_rate
N = num_samps
ts = 1 / fs
t = np.arange(0, T_chirp, ts)
f0 = 0
f1 = 12e6
decim = 50
c = 3e8

""" RX Configuration """
sdr.rx_enabled_channels = [0, 1]
sdr.sample_rate = int(samp_rate)
sdr.rx_lo = int(rx_lo)
sdr.gain_control_mode = rx_mode
sdr.rx_hardwaregain_chan0 = int(rx_gain0)
sdr.rx_hardwaregain_chan1 = int(rx_gain1)
sdr.rx_buffer_size = num_samps

""" TX Configuration """
sdr.tx_enabled_channels = [0]
sdr.tx_rf_bandwidth = int(samp_rate)
sdr.tx_lo = int(tx_lo)
sdr.tx_cyclic_buffer = True
sdr.tx_hardwaregain_chan0 = int(tx_gain0)
sdr.tx_buffer_size = num_samps

""" Chirp Configuration """
phase = 2 * np.pi * chirp(t, f0=f0, t1=(N*ts), f1=f1, method='linear', phi=0)
i = np.cos(phase) * 2**14
q = np.sin(phase) * 2**14
chirp_iq = i + 1j * q
iq0 = np.tile(chirp_iq, num_chirps)
sdr.tx(iq0)

cmn = ''
PRF = 1 / T_chirp
wavelength = c/rx_lo
max_doppler_freq = PRF / 2
max_doppler_vel = max_doppler_freq * wavelength / 2

""" Real-time Görselleştirme """
plt.ion()
fig, ax = plt.subplots(1, figsize=(8,8))
cmaps = ['inferno', 'plasma']
cmn = cmaps[0]
ax.set_ylabel('Range [m]')
ax.set_title('Range-Doppler Map')
ax.set_xlabel('Velocity [m/s]')
img = ax.imshow(np.zeros((num_samps, num_chirps // (2 * decim))), origin='lower',
                aspect='auto', extent=[-max_doppler_vel, max_doppler_vel, 0, 100],
                cmap=matplotlib.colormaps.get_cmap(cmn))
cbar = plt.colorbar(img, ax=ax)
cbar.set_label("Amplitude (dB)")
text_handle = ax.text(0.97, 0.97, "", transform=ax.transAxes, 
                      fontsize=12, color='white', 
                      verticalalignment='top', horizontalalignment='right',
                      bbox=dict(facecolor='black', alpha=0.5, pad=5))
  
def range_doppler_fft (data, num_chirps, num_samps, beta):
    global fs_dec, fs, decim, k, c, rx_lo, T_chirp
    
    win_range = np.kaiser(num_samps, beta=beta)
    range_fft = np.fft.fft(data * win_range, axis=1)
    range_fft = range_fft[:, :num_samps // 2]

    win_doppler = np.kaiser(num_chirps, beta=beta)[:, None]
    doppler_fft = np.fft.fftshift(np.fft.fft(range_fft * win_doppler, axis=0), axes=0)
    
    rd_mag = np.abs(doppler_fft) / np.max(np.abs(doppler_fft))
    rd_dbfs = 20*np.log10(rd_mag/(2**12))

    fs_dec = fs / decim
    freqs = np.fft.fftfreq(num_samps_decimated, 1/fs_dec)[:num_samps_decimated // 2]
    real_range = freqs * c / (2*k)
    doppler_freqs = np.fft.fftshift(np.fft.fftfreq(num_chirps, 2 * T_chirp))
    real_velocity = doppler_freqs * c / (2 * rx_lo)
    
    return rd_dbfs, real_range, real_velocity

print("Radar ON")

""" Radar Döngüsü """
try:
    while True:
        rx_data = np.zeros((num_chirps, num_samps), dtype=complex)
        
        for i in range(num_chirps):
            data = sdr.rx()
            rx_data[i,:] = data[0] * np.conj(data[1])
                
        rx_data_decimate = np.array([
            decimate(rx_data[i,:], q=decim, ftype='fir', zero_phase=True)
            for i in range(num_chirps)])
        num_samps_decimated = rx_data_decimate.shape[1]
        
        # 2D FFT
        rd_dbfs, real_range, real_velocity = range_doppler_fft(rx_data_decimate,
                                                               num_chirps=num_chirps,
                                                               num_samps=num_samps_decimated,
                                                               beta=5)
        
        # Max Range
        min_range = 0
        max_range = 150
        range_indices = np.where((real_range >= min_range) & (real_range <= max_range))[0]
        range_cropped = real_range[range_indices]
        rd_dbfs_cropped = rd_dbfs[:, range_indices].T
        
        img.set_data(rd_dbfs_cropped)
        img.set_extent([real_velocity[0], real_velocity[-1],
                        range_cropped[0], range_cropped[-1]])
        vmax = np.max(rd_dbfs_cropped)
        vmin = vmax - 50
        img.set_clim(vmin=vmin, vmax=vmax)
        plt.pause(0.01)
        
        max_idx = np.unravel_index(np.argmax(rd_dbfs_cropped, axis=None), rd_dbfs_cropped.shape)
        rng_idx, vel_idx = max_idx
        target_range = range_cropped[rng_idx]
        target_velocity = real_velocity[vel_idx]

        print(f"Hedef Mesafesi: {target_range:.2f} m | Hedef Hızı: {target_velocity:.2f} m/s")
        text_handle.set_text(f"Hedef Mesafesi: {target_range:.2f} m\nHedef Hızı: {target_velocity:.2f} m/s")
        
except KeyboardInterrupt:
    sdr.tx_destroy_buffer()
    print("Radar OFF")
    plt.ioff()
    
    
    
    