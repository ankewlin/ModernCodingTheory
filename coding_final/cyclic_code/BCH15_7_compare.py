import numpy as np
import matplotlib.pyplot as plt
from BCH15_7 import bch_decode, bch_encode
###实际上BCH(7,4)和Hamming(7,4)本质等价

# ===================== BCH(7,4) 编码与译码 =======================

G = np.array([
    [1, 0, 0, 0, 1, 1, 0],
    [0, 1, 0, 0, 1, 0, 1],
    [0, 0, 1, 0, 0, 1, 1],
    [0, 0, 0, 1, 1, 1, 1]
])  # (4,7)

H = np.array([
    [1, 1, 0, 1, 1, 0, 0],
    [1, 0, 1, 1, 0, 1, 0],
    [0, 1, 1, 1, 0, 0, 1]
])  # (3,7)

# Syndrome表: 错误位映射
syndrome_table = {
    (0, 0, 0): None,
    (0, 0, 1): 6,
    (0, 1, 0): 5,
    (0, 1, 1): 2,
    (1, 0, 0): 4,
    (1, 0, 1): 0,
    (1, 1, 0): 1,
    (1, 1, 1): 3
}


# ======================= BPSK 调制/解调 ============================

def bpsk_mod(bits):
    return 1 - 2 * bits  # 0 -> +1, 1 -> -1

def bpsk_demod(symbols):
    return (symbols < 0).astype(int)

# ======================= 信道模型 ================================

def awgn(signal, snr_db):
    snr = 10 ** (snr_db / 10)
    sigma = np.sqrt(1 / (2 * snr))
    noise = sigma * np.random.randn(*signal.shape)
    return signal + noise

def bsc(signal, p):
    flips = np.random.rand(*signal.shape) < p
    return (signal + flips) % 2

# ======================== 性能仿真主流程 =========================

def simulate_awgn(snr_db, nblocks=10000):
    nerr, nfer, ntotal = 0, 0, 0
    fixed_seq = [1, 1, 0, 1, 0, 1, 0]  # 固定信源比特序列
    for _ in range(nblocks):
        data = fixed_seq
        code = bch_encode(data)
        code = np.array(code)
        mod = bpsk_mod(code)
        rx = awgn(mod, snr_db)
        hard = bpsk_demod(rx)
        dec = bch_decode(hard)
        nbit_err = np.sum(dec != data)
        nerr += nbit_err
        nfer += int(nbit_err > 0)
        ntotal += 4
    return nerr / ntotal, nfer / nblocks

def simulate_bsc(p, nblocks=10000):
    nerr, nfer, ntotal = 0, 0, 0
    fixed_seq = [1, 1, 0, 1, 0, 1, 0]  # 固定信源比特序列
    for _ in range(nblocks):
        data = fixed_seq
        code = bch_encode(data)
        code = np.array(code)
        code_bsc = bsc(code, p)
        dec = bch_decode(code_bsc)
        nbit_err = np.sum(dec != data)
        nerr += nbit_err
        nfer += int(nbit_err > 0)
        ntotal += 4
    return nerr / ntotal, nfer / nblocks

# ======================= 绘制性能曲线 =============================

def plot_awgn():
    snr_db_list = np.arange(0, 5, 0.5)
    ber, fer = [], []
    for snr in snr_db_list:
        b, f = simulate_awgn(snr, nblocks=10000)
        ber.append(b)
        fer.append(f)
        print(f"SNR={snr}dB, BER={b:.3e}, FER={f:.3e}")
    plt.figure()
    plt.semilogy(snr_db_list, ber, 'o-', label='BER')
    plt.semilogy(snr_db_list, fer, 's--', label='FER')
    plt.xlabel('SNR (dB)')
    plt.ylabel('Error Rate')
    plt.title('BCH [7,4] on AWGN')
    plt.grid(True, which='both')
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_bsc():
    p_list = np.logspace(-4, -1, 8)
    ber, fer = [], []
    for p in p_list:
        b, f = simulate_bsc(p, nblocks=10000)
        ber.append(b)
        fer.append(f)
        print(f"p={p:.2e}, BER={b:.3e}, FER={f:.3e}")
    plt.figure()
    plt.semilogy(p_list, ber, 'o-', label='BER')
    plt.semilogy(p_list, fer, 's--', label='FER')
    plt.xlabel('BSC Flip Probability')
    plt.ylabel('Error Rate')
    plt.title('BCH [7,4] on BSC')
    plt.grid(True, which='both')
    plt.legend()
    plt.tight_layout()
    plt.show()

# ======================== 主程序入口 =============================

if __name__ == "__main__":
    print("AWGN信道仿真：")
    plot_awgn()
    print("\nBSC信道仿真：")
    plot_bsc()
