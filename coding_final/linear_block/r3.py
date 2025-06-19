import numpy as np
import matplotlib.pyplot as plt

# ========== R3重复码编码与译码 =========

def r3_encode(data):
    return np.repeat(data, 3)

def r3_decode(received):
    bits = []
    for i in range(0, len(received), 3):
        chunk = received[i:i+3]
        bit = int(np.sum(chunk) > 1)
        bits.append(bit)
    return np.array(bits)

# ========== BPSK/信道同原代码 ==========

def bpsk_mod(bits):
    return 1 - 2 * bits

def bpsk_demod(symbols):
    return (symbols < 0).astype(int)

def awgn(signal, snr_db):
    snr = 10 ** (snr_db / 10)
    sigma = np.sqrt(1 / (2 * snr))
    noise = sigma * np.random.randn(*signal.shape)
    return signal + noise

def bsc(signal, p):
    flips = np.random.rand(*signal.shape) < p
    return (signal + flips) % 2

# ========== 仿真流程 ==========

def simulate_awgn(snr_db, nblocks=400000):
    nerr, nfer, ntotal = 0, 0, 0
    fixed_seq = np.array([1, 1, 0, 1])
    for _ in range(nblocks):
        data = fixed_seq
        code = r3_encode(data)
        mod = bpsk_mod(code)
        rx = awgn(mod, snr_db)
        hard = bpsk_demod(rx)
        dec = r3_decode(hard)
        nbit_err = np.sum(dec != data)
        nerr += nbit_err
        nfer += int(nbit_err > 0)
        ntotal += 4
    return nerr / ntotal, nfer / nblocks

def simulate_bsc(p, nblocks=400000):
    nerr, nfer, ntotal = 0, 0, 0
    fixed_seq = np.array([1, 1, 0, 1])
    for _ in range(nblocks):
        data = fixed_seq
        code = r3_encode(data)
        code_bsc = bsc(code, p)
        dec = r3_decode(code_bsc)
        nbit_err = np.sum(dec != data)
        nerr += nbit_err
        nfer += int(nbit_err > 0)
        ntotal += 4
    return nerr / ntotal, nfer / nblocks

# ========== 绘图同原代码 ==========

def plot_awgn():
    snr_db_list = np.arange(0, 9, 1)
    ber, fer = [], []
    for snr in snr_db_list:
        b, f = simulate_awgn(snr, nblocks=400000)
        ber.append(b)
        fer.append(f)
        print(f"SNR={snr}dB, BER={b:.3e}, FER={f:.3e}")
    ber = np.array(ber)
    fer = np.array(fer)
    # fer[fer == 0] = 1e-8   # 避免0值
    # ber[ber == 0] = 1e-8
    plt.figure()
    plt.semilogy(snr_db_list, ber, 'o-', label='BER')
    plt.semilogy(snr_db_list, fer, 's--', label='FER')
    plt.xlabel('SNR (dB)')
    plt.ylabel('Error Rate')
    plt.title('R3 Code on AWGN')
    plt.ylim(1e-9, 1)  # 或 plt.ylim(1e-8, 1)
    plt.grid(True, which='both')
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_bsc():
    p_list = np.logspace(-4, -1, 8)
    ber, fer = [], []
    for p in p_list:
        b, f = simulate_bsc(p, nblocks=400000)
        ber.append(b)
        fer.append(f)
        print(f"p={p:.2e}, BER={b:.3e}, FER={f:.3e}")
    ber = np.array(ber)
    fer = np.array(fer)
    # fer[fer == 0] = 1e-8   # 避免0值
    # ber[ber == 0] = 1e-8
    plt.figure()
    plt.semilogy(p_list, ber, 'o-', label='BER')
    plt.semilogy(p_list, fer, 's--', label='FER')
    plt.xlabel('BSC Flip Probability')
    plt.ylabel('Error Rate')
    plt.ylim(1e-9, 1)  # 或 plt.ylim(1e-8, 1)
    plt.title('R3 Code on BSC')
    plt.grid(True, which='both')
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    print("AWGN信道仿真：")
    plot_awgn()
    print("\nBSC信道仿真：")
    plot_bsc()
