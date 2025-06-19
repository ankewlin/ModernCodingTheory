import numpy as np
import matplotlib.pyplot as plt
"""
卷积编码: 
约束长度（Constraint Length）:K=3 
码率（Code Rate）: R=1/2
g1=111 (7), g2=101 (5):(g1=111, g2=101)

AWGN信道：SNR（信噪比，单位dB），常用范围0~8 dB

BSC信道：p（二进制对称信道的比特翻转概率），常用范围1e-4~1e-1

Viterbi译码参数
状态数：2^(K-1) = 4（因为K=3）
硬判决Viterbi：只使用解调后的0/1输入
"""
# ============= 卷积编码 K=3, R=1/2, (g1=111, g2=101) =============

def conv_encode(bits):
    """
    bits: (n,) array of 0/1
    return: (2*n,) array (每个输入出2个输出)
    g1 = 111, g2 = 101
    """
    n = len(bits)
    state = [0,0]  # K-1=2 个初始状态
    out = []
    for b in bits:
        s = [b] + state
        # g1: 1 1 1
        o1 = s[0] ^ s[1] ^ s[2]
        # g2: 1 0 1
        o2 = s[0] ^ s[2]
        out += [o1, o2]
        state = [b] + state[:1]
    return np.array(out)

# ============== Viterbi 硬判决译码算法 =================

def viterbi_decode(y):
    """
    y: (2*N,) 硬判决（0/1）
    return: (N,) 译码比特流
    """
    N = len(y)//2
    trellis = np.ones((N+1,4)) * 1e9  # 4状态
    path = np.zeros((N+1,4), dtype=int)
    trellis[0,0] = 0  # 初始状态00

    for i in range(N):
        for s in range(4):
            if trellis[i,s]<1e8:
                for b in [0,1]:
                    # b为输入，s为前状态（低位新）
                    # 计算当前输入b时的输出和下一状态
                    s_in = [b, (s>>1)&1, s&1]
                    o1 = s_in[0]^s_in[1]^s_in[2]
                    o2 = s_in[0]^s_in[2]
                    metric = (y[2*i]!=o1) + (y[2*i+1]!=o2)
                    ss = ((s>>1)|(b<<1))&0b11
                    if trellis[i+1,ss]>trellis[i,s]+metric:
                        trellis[i+1,ss]=trellis[i,s]+metric
                        path[i+1,ss]=s
    # 回溯
    s = np.argmin(trellis[N])
    x_hat = []
    for i in range(N,0,-1):
        prev = path[i,s]
        b = (s >> 1) & 1
        x_hat.append(b)
        s = prev
    x_hat.reverse()
    return np.array(x_hat)

# ================== BPSK调制/解调、信道 ==================

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

# =================== 仿真流程 =========================

def simulate_conv_awgn(snr_db, nblocks=10000):
    nerr, nfer, ntotal = 0, 0, 0
    fixed_seq = np.array([1,1,0,1])
    for _ in range(nblocks):
        data = fixed_seq
        code = conv_encode(data)
        mod = bpsk_mod(code)
        rx = awgn(mod, snr_db)
        hard = bpsk_demod(rx)
        dec = viterbi_decode(hard)
        nbit_err = np.sum(dec != data)
        nerr += nbit_err
        nfer += int(nbit_err>0)
        ntotal += len(data)
    return nerr/ntotal, nfer/nblocks

def simulate_conv_bsc(p, nblocks=10000):
    nerr, nfer, ntotal = 0, 0, 0
    fixed_seq = np.array([1,1,0,1])
    for _ in range(nblocks):
        data = fixed_seq
        code = conv_encode(data)
        code_bsc = bsc(code, p)
        dec = viterbi_decode(code_bsc)
        nbit_err = np.sum(dec != data)
        nerr += nbit_err
        nfer += int(nbit_err>0)
        ntotal += len(data)
    return nerr/ntotal, nfer/nblocks

# =================== 绘图函数 =========================

def plot_conv_awgn():
    snr_db_list = np.arange(0, 9, 1)
    ber, fer = [], []
    for snr in snr_db_list:
        b, f = simulate_conv_awgn(snr, nblocks=10000)
        ber.append(b)
        fer.append(f)
        print(f"SNR={snr}dB, BER={b:.3e}, FER={f:.3e}")
    plt.figure()
    plt.semilogy(snr_db_list, ber, 'o-', label='BER')
    plt.semilogy(snr_db_list, fer, 's--', label='FER')
    plt.xlabel('SNR (dB)')
    plt.ylabel('Error Rate')
    plt.title('Convolutional Code (K=3, R=1/2) on AWGN')
    plt.grid(True, which='both')
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_conv_bsc():
    p_list = np.logspace(-4, -1, 8)
    ber, fer = [], []
    for p in p_list:
        b, f = simulate_conv_bsc(p, nblocks=10000)
        ber.append(b)
        fer.append(f)
        print(f"p={p:.2e}, BER={b:.3e}, FER={f:.3e}")
    plt.figure()
    plt.semilogy(p_list, ber, 'o-', label='BER')
    plt.semilogy(p_list, fer, 's--', label='FER')
    plt.xlabel('BSC Flip Probability')
    plt.ylabel('Error Rate')
    plt.title('Convolutional Code (K=3, R=1/2) on BSC')
    plt.grid(True, which='both')
    plt.legend()
    plt.xscale('log')
    plt.tight_layout()
    plt.show()

# =================== 主程序入口 =========================

if __name__ == "__main__":
    print("AWGN信道仿真：")
    plot_conv_awgn()
    print("\nBSC信道仿真：")
    plot_conv_bsc()
