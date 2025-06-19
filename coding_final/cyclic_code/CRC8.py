import numpy as np
import matplotlib.pyplot as plt

# ===================== CRC-8 Encode/Decode =======================

GEN = 0x107  # CRC-8 generator polynomial: x^8 + x^2 + x + 1

def crc8_encode(data):
    """
    data: shape (8,) - input data bits
    return: shape (16,) - encoded data (8 data bits + 8 CRC bits)
    """
    m = int(''.join(str(b) for b in data), 2)
    m_shift = m << 8  # append 8 zeros (shift left)
    val = m_shift
    for i in reversed(range(8, 16)):
        if (val >> i) & 1:
            val ^= (GEN << (i - 8))
    crc = val & 0xFF  # last 8 bits are CRC
    codeword = (m << 8) | crc
    return np.array([int(b) for b in f"{codeword:016b}"])

def crc8_check(codeword):
    """
    codeword: shape (16,)
    return: decoded 8-bit data
    If no error detected, return the data bits.
    Otherwise, return the data bits as is (no correction).
    """
    c = int(''.join(str(b) for b in codeword), 2)
    val = c
    for i in reversed(range(8, 16)):
        if (val >> i) & 1:
            val ^= (GEN << (i - 8))
    if val & 0xFF == 0:  # CRC check passed
        return np.array([int(b) for b in f"{c >> 8:08b}"])
    else:
        return np.array([int(b) for b in f"{c >> 8:08b}"])  # no correction

# ===================== BPSK Modulation/Detection =====================

def bpsk_mod(bits):
    return 1 - 2 * bits  # 0 -> +1, 1 -> -1

def bpsk_demod(symbols):
    return (symbols < 0).astype(int)  # hard decision

# ===================== Channel Models =====================

def awgn(signal, snr_db):
    snr = 10 ** (snr_db / 10)
    sigma = np.sqrt(1 / (2 * snr))
    noise = sigma * np.random.randn(*signal.shape)
    return signal + noise

def bsc(signal, p):
    flips = np.random.rand(*signal.shape) < p
    return (signal + flips) % 2

# ===================== Simulation Functions =====================

def simulate_awgn_crc8(snr_db, nblocks=100000):
    bit_errors, frame_errors, total_bits = 0, 0, 0
    for _ in range(nblocks):
        data = np.random.randint(0, 2, 8)
        code = crc8_encode(data)
        modulated = bpsk_mod(code)
        received = awgn(modulated, snr_db)
        hard_bits = bpsk_demod(received)
        decoded = crc8_check(hard_bits)
        errors = np.sum(decoded != data)
        bit_errors += errors
        frame_errors += int(errors > 0)
        total_bits += 8
    return bit_errors / total_bits, frame_errors / nblocks

def simulate_bsc_crc8(p, nblocks=100000):
    bit_errors, frame_errors, total_bits = 0, 0, 0
    for _ in range(nblocks):
        data = np.random.randint(0, 2, 8)
        code = crc8_encode(data)
        received = bsc(code, p)
        decoded = crc8_check(received)
        errors = np.sum(decoded != data)
        bit_errors += errors
        frame_errors += int(errors > 0)
        total_bits += 8
    return bit_errors / total_bits, frame_errors / nblocks

# ===================== Plotting =====================

def plot_awgn_crc8():
    snr_list = np.arange(0, 9, 1)
    ber_list, fer_list = [], []
    for snr in snr_list:
        ber, fer = simulate_awgn_crc8(snr)
        ber_list.append(ber)
        fer_list.append(fer)
        print(f"SNR = {snr} dB | BER = {ber:.3e} | FER = {fer:.3e}")
    plt.figure()
    plt.semilogy(snr_list, ber_list, 'o-', label='BER')
    plt.semilogy(snr_list, fer_list, 's--', label='FER')
    plt.xlabel('SNR (dB)')
    plt.ylabel('Error Rate')
    plt.title('CRC-8 Code on AWGN Channel')
    plt.grid(True, which='both')
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_bsc_crc8():
    p_list = np.logspace(-4, -1, 8)
    ber_list, fer_list = [], []
    for p in p_list:
        ber, fer = simulate_bsc_crc8(p)
        ber_list.append(ber)
        fer_list.append(fer)
        print(f"Flip Probability = {p:.1e} | BER = {ber:.3e} | FER = {fer:.3e}")
    plt.figure()
    plt.semilogy(p_list, ber_list, 'o-', label='BER')
    plt.semilogy(p_list, fer_list, 's--', label='FER')
    plt.xlabel('BSC Flip Probability')
    plt.ylabel('Error Rate')
    plt.title('CRC-8 Code on BSC Channel')
    plt.grid(True, which='both')
    plt.legend()
    plt.tight_layout()
    plt.show()

# ===================== Main Entry =====================

if __name__ == "__main__":
    print("Simulating CRC-8 on AWGN Channel:")
    plot_awgn_crc8()
    print("\nSimulating CRC-8 on BSC Channel:")
    plot_bsc_crc8()
