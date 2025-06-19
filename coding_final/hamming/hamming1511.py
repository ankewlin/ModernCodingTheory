import numpy as np
import matplotlib.pyplot as plt

# ===================== Hamming(15,11) Encoder and Decoder =======================
# Corrected generator matrix G (11x15)
G = np.array([
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1],
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1],
    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1],
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1],
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1]
], dtype=int)

# Corrected parity-check matrix H (4x15)
H = np.array([
    [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    [1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0],
    [0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0],
    [1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1]
], dtype=int)

# Verify G*H^T = 0 (mod 2)
if not np.all((G @ H.T) % 2 == 0):
    print("Warning: G and H are not orthogonal!")

# Syndrome lookup table for single-bit error correction
syndrome_table = {}
for i in range(15):
    error_vector = np.zeros(15, dtype=int)
    error_vector[i] = 1
    syndrome = tuple((H @ error_vector) % 2)
    syndrome_table[syndrome] = i
syndrome_table[(0,0,0,0)] = None  # No error

def hamming1511_encode(data):
    return (data @ G) % 2

def hamming1511_decode(codeword):
    syndrome = tuple((codeword @ H.T) % 2)
    corrected = codeword.copy()
    if syndrome in syndrome_table:
        error_idx = syndrome_table[syndrome]
        if error_idx is not None:
            corrected[error_idx] ^= 1  # Flip the erroneous bit
    return corrected[:11]  # Return only data bits

# ======================= BPSK Modulation / Demodulation ===========================
def bpsk_mod(bits):
    return 1 - 2 * bits  # 0 -> +1, 1 -> -1

def bpsk_demod(symbols):
    return (symbols < 0).astype(int)  # Hard decision

# ========================= Channel Models ========================================
def awgn(signal, ebno_db, rate=11/15):
    # Convert Eb/N0 (dB) to linear SNR
    ebno_linear = 10 ** (ebno_db / 10)
    # Calculate noise variance (accounting for coding rate)
    noise_var = 1 / (2 * rate * ebno_linear)
    noise = np.sqrt(noise_var) * np.random.randn(*signal.shape)
    return signal + noise

def bsc(bits, flip_prob):
    flips = np.random.rand(*bits.shape) < flip_prob
    return (bits + flips) % 2

# ========================= Simulation Core =======================================
def simulate_bsc(p, nblocks=10000):
    total_bit_errors = 0
    total_frame_errors = 0
    for _ in range(nblocks):
        data = np.random.randint(0, 2, 11)
        encoded = hamming1511_encode(data)
        received = bsc(encoded, p)
        decoded = hamming1511_decode(received)
        bit_errors = np.sum(decoded != data)
        total_bit_errors += bit_errors
        total_frame_errors += (bit_errors > 0)
    ber = total_bit_errors / (nblocks * 11)
    fer = total_frame_errors / nblocks
    return ber, fer

def simulate_awgn(ebno_db, nblocks=10000):
    total_bit_errors = 0
    total_frame_errors = 0
    for _ in range(nblocks):
        data = np.random.randint(0, 2, 11)
        encoded = hamming1511_encode(data)
        modulated = bpsk_mod(encoded)
        received = awgn(modulated, ebno_db)
        demodulated = bpsk_demod(received)
        decoded = hamming1511_decode(demodulated)
        bit_errors = np.sum(decoded != data)
        total_bit_errors += bit_errors
        total_frame_errors += (bit_errors > 0)
    ber = total_bit_errors / (nblocks * 11)
    fer = total_frame_errors / nblocks
    return ber, fer

# ========================= Plotting Functions ====================================
def plot_bsc():
    p_list = np.logspace(-4, -1, 8)
    ber_list, fer_list = [], []
    for p in p_list:
        ber, fer = simulate_bsc(p)
        ber_list.append(ber)
        fer_list.append(fer)
        print(f"Flip Prob = {p:.1e}, BER = {ber:.3e}, FER = {fer:.3e}")
    
    plt.figure(figsize=(10, 6))
    plt.semilogy(p_list, ber_list, 'o-', label='BER')
    plt.semilogy(p_list, fer_list, 's--', label='FER')
    plt.xlabel("BSC Flip Probability")
    plt.ylabel("Error Rate")
    plt.title("Hamming(15,11) Performance on BSC")
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig('hamming_bsc.png')
    plt.show()

def plot_awgn():
    ebno_db_range = np.linspace(0, 8, 9)
    ber_list, fer_list = [], []
    
    for ebno_db in ebno_db_range:
        ber, fer = simulate_awgn(ebno_db)
        ber_list.append(ber)
        fer_list.append(fer)
        print(f"Eb/N0 = {ebno_db:.1f} dB, BER = {ber:.3e}, FER = {fer:.3e}")
    
    plt.figure(figsize=(10, 6))
    plt.semilogy(ebno_db_range, ber_list, 'o-', label='BER')
    plt.semilogy(ebno_db_range, fer_list, 's--', label='FER')
    plt.xlabel("Eb/N0 (dB)")
    plt.ylabel("Error Rate")
    plt.title("Hamming(15,11) Performance on AWGN Channel")
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig('hamming_awgn.png')
    plt.show()

# ========================= Main Entry ============================================
if __name__ == "__main__":
    print("Simulating Hamming(15,11) over AWGN Channel:")
    plot_awgn()
    
    print("\nSimulating Hamming(15,11) over BSC Channel:")
    plot_bsc()