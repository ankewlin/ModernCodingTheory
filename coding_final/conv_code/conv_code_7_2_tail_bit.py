import numpy as np
import matplotlib.pyplot as plt
from conv_code_7_2 import ConvolutionalCode


"""
卷积编码: 
约束长度（Constraint Length）: K=7 
码率（Code Rate）: R=1/2
生成多项式: g1=111 (7), g2=101 (5)

主要修改：
1. 编码函数添加尾比特处理
2. 译码函数添加尾比特处理
3. 更新仿真函数以处理尾比特
"""




# ================== BPSK调制/解调、信道 ==================

def bpsk_mod(bits):
    """BPSK调制: 0->+1, 1->-1"""
    return 1 - 2 * bits


def bpsk_demod(symbols):
    """BPSK解调: +1->0, -1->1"""
    return (symbols < 0).astype(int)


def awgn(signal, snr_db):
    """添加AWGN噪声"""
    snr = 10 ** (snr_db / 10)
    # 计算噪声标准差（假设信号功率为1）
    sigma = np.sqrt(1 / (2 * snr))
    noise = sigma * np.random.randn(*signal.shape)
    return signal + noise


def bsc(signal, p):
    """二进制对称信道"""
    flips = np.random.rand(*signal.shape) < p
    return (signal + flips) % 2


# =================== 仿真流程 =========================

def simulate_conv_awgn(snr_db, nblocks=10000):
    """AWGN信道仿真"""
    nerr, nfer, ntotal = 0, 0, 0
    fixed_seq = np.array([1, 1, 0, 1])  # 原始信息比特

    for _ in range(nblocks):
        # 编码（自动添加尾比特）
        code = cc.encode(fixed_seq)
        code = np.array(code)
        # BPSK调制
        mod = bpsk_mod(code)

        # AWGN信道
        rx = awgn(mod, snr_db)

        # 硬判决解调
        hard = bpsk_demod(rx)

        # 维特比译码（自动去除尾比特）
        dec = cc.decode(hard)

        # 错误统计
        nbit_err = np.sum(dec != fixed_seq)
        nerr += nbit_err
        nfer += int(nbit_err > 0)
        ntotal += len(fixed_seq)

    return nerr / ntotal, nfer / nblocks


def simulate_conv_bsc(p, nblocks=10000):
    """BSC信道仿真"""
    nerr, nfer, ntotal = 0, 0, 0
    fixed_seq = np.array([1, 1, 0, 1])  # 原始信息比特

    for _ in range(nblocks):
        # 编码（自动添加尾比特）
        code = cc.encode(fixed_seq)
        code = np.array(code)
        # BSC信道
        code_bsc = bsc(code, p)

        # 维特比译码（自动去除尾比特）
        dec = cc.decode(code_bsc)

        # 错误统计
        nbit_err = np.sum(dec != fixed_seq)
        nerr += nbit_err
        nfer += int(nbit_err > 0)
        ntotal += len(fixed_seq)

    return nerr / ntotal, nfer / nblocks


# =================== 绘图函数 =========================

def plot_conv_awgn():
    """绘制AWGN信道下的性能曲线"""
    snr_db_list = np.arange(0, 9, 1)
    ber, fer = [], []

    print("AWGN信道仿真:")
    print("SNR(dB) | BER       | FER")
    print("--------|-----------|-----------")

    for snr in snr_db_list:
        b, f = simulate_conv_awgn(snr, nblocks=10000)
        ber.append(b)
        fer.append(f)
        print(f"{snr:6} | {b:.2e} | {f:.2e}")

    plt.figure(figsize=(10, 6))
    plt.semilogy(snr_db_list, ber, 'o-', linewidth=2, label='BER')
    plt.semilogy(snr_db_list, fer, 's--', linewidth=2, label='FER')
    plt.xlabel('SNR (dB)', fontsize=12)
    plt.ylabel('Error Rate', fontsize=12)
    plt.title('Convolutional Code (K=7, R=1/2) Performance on AWGN', fontsize=14)
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    plt.xticks(snr_db_list)
    plt.tight_layout()
    plt.savefig('conv_awgn_performance.png', dpi=300)
    plt.show()
    print("\n图表已保存为 'conv_awgn_performance.png'")


def plot_conv_bsc():
    """绘制BSC信道下的性能曲线"""
    p_list = np.logspace(-4, -1, 8)  # 10^{-4}到10^{-1}
    ber, fer = [], []

    print("\nBSC信道仿真:")
    print("p       | BER       | FER")
    print("--------|-----------|-----------")

    for p in p_list:
        b, f = simulate_conv_bsc(p, nblocks=10000)
        ber.append(b)
        fer.append(f)
        print(f"{p:.1e} | {b:.2e} | {f:.2e}")

    plt.figure(figsize=(10, 6))
    plt.semilogy(p_list, ber, 'o-', linewidth=2, label='BER')
    plt.semilogy(p_list, fer, 's--', linewidth=2, label='FER')
    plt.xlabel('BSC Flip Probability (p)', fontsize=12)
    plt.ylabel('Error Rate', fontsize=12)
    plt.title('Convolutional Code (K=7, R=1/2) Performance on BSC', fontsize=14)
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    plt.xscale('log')
    plt.tight_layout()
    plt.savefig('conv_bsc_performance.png', dpi=300)
    plt.show()
    print("\n图表已保存为 'conv_bsc_performance.png'")


# =================== 主程序入口 =========================

if __name__ == "__main__":
    # 测试编码和解码

    cc = ConvolutionalCode(constraint_length=7, rate=1 / 2)
    
    test_data = np.array([1, 0, 1, 1])
    print("测试数据:", test_data)

    # 编码（自动添加尾比特）
    encoded = cc.encode(test_data)
    print("编码结果:", encoded)
    print("编码长度:", len(encoded), "bits")

    # 无噪声解码
    decoded = cc.decode(encoded)
    print("解码结果:", decoded)
    print("解码是否正确:", np.array_equal(decoded, test_data))

    # 添加噪声测试
    noisy_encoded = encoded.copy()
    noisy_encoded[3] = 1 - noisy_encoded[3]  # 翻转一个比特
    noisy_decoded = cc.decode(noisy_encoded)
    print("\n带噪声解码结果:", noisy_decoded)
    print("解码是否正确:", np.array_equal(noisy_decoded, test_data))

    # 运行完整仿真
    plot_conv_awgn()
    plot_conv_bsc()