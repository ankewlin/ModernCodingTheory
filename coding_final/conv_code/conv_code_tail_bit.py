import numpy as np
import matplotlib.pyplot as plt

"""
卷积编码: 
约束长度（Constraint Length）: K=3 
码率（Code Rate）: R=1/2
生成多项式: g1=111 (7), g2=101 (5)

主要修改：
1. 编码函数添加尾比特处理
2. 译码函数添加尾比特处理
3. 更新仿真函数以处理尾比特
"""


# ============= 卷积编码 K=3, R=1/2, (g1=111, g2=101) =============

def conv_encode(bits):
    """
    bits: (n,) array of 0/1
    return: (2*(n+K-1),) array (添加尾比特使状态归零)
    g1 = 111, g2 = 101
    """
    # 添加尾比特（K-1=2个0）
    padded_bits = np.append(bits, [0, 0])
    n = len(padded_bits)

    state = [0, 0]  # K-1=2 个初始状态
    out = []

    for b in padded_bits:
        # 当前状态：新输入 + 前两个状态
        s = [b] + state

        # g1: 1 1 1
        o1 = s[0] ^ s[1] ^ s[2]
        # g2: 1 0 1
        o2 = s[0] ^ s[2]

        out += [o1, o2]

        # 更新状态：新输入成为下一个状态的高位
        state = [b] + state[:1]

    return np.array(out)


# ============== Viterbi 硬判决译码算法 =================

def viterbi_decode(y):
    """
    y: (2*N,) 硬判决（0/1）
    return: (N-K+1,) 译码比特流（去除尾比特）
    """
    N = len(y) // 2  # 总步数（包括尾比特）
    num_states = 4  # 2^(K-1) = 4

    # 初始化网格图和路径
    trellis = np.full((N + 1, num_states), np.inf)
    path = np.zeros((N + 1, num_states), dtype=int)  # 存储前一状态
    survivor_bits = np.zeros((N + 1, num_states), dtype=int)  # 存储输入比特

    # 初始状态：00
    trellis[0, 0] = 0

    # 状态转移表：state -> (input 0: next_state, output; input 1: next_state, output)
    # 格式: {当前状态: [(下一状态0, 输出0), (下一状态1, 输出1)]}
    transitions = {
        0: [(0, (0, 0)), (2, (1, 1))],  # 00 -> 00(00) or 10(11)
        1: [(0, (1, 1)), (2, (0, 0))],  # 01 -> 00(11) or 10(00)
        2: [(1, (1, 0)), (3, (0, 1))],  # 10 -> 01(10) or 11(01)
        3: [(1, (0, 1)), (3, (1, 0))]  # 11 -> 01(01) or 11(10)
    }

    # 维特比算法主循环
    for t in range(N):
        received_pair = (y[2 * t], y[2 * t + 1])

        for state in range(num_states):
            if trellis[t, state] == np.inf:
                continue  # 跳过不可达状态

            # 处理两个可能的输入 (0 和 1)
            for input_bit in [0, 1]:
                next_state, output_pair = transitions[state][input_bit]

                # 计算分支度量（汉明距离）
                branch_metric = (received_pair[0] != output_pair[0]) + \
                                (received_pair[1] != output_pair[1])

                # 更新路径度量
                new_metric = trellis[t, state] + branch_metric

                if new_metric < trellis[t + 1, next_state]:
                    trellis[t + 1, next_state] = new_metric
                    path[t + 1, next_state] = state
                    survivor_bits[t + 1, next_state] = input_bit

    # 回溯 - 从终态00开始
    current_state = 0  # 尾比特确保终态为00
    decoded_bits = []

    # 从最后一步回溯到第一步
    for t in range(N, 0, -1):
        decoded_bits.append(survivor_bits[t, current_state])
        current_state = path[t, current_state]

    # 反转并去除尾比特 (最后K-1=2个比特)
    decoded_bits = np.array(decoded_bits[::-1])
    return decoded_bits[:-2]  # 去除尾比特


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
        code = conv_encode(fixed_seq)

        # BPSK调制
        mod = bpsk_mod(code)

        # AWGN信道
        rx = awgn(mod, snr_db)

        # 硬判决解调
        hard = bpsk_demod(rx)

        # 维特比译码（自动去除尾比特）
        dec = viterbi_decode(hard)

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
        code = conv_encode(fixed_seq)

        # BSC信道
        code_bsc = bsc(code, p)

        # 维特比译码（自动去除尾比特）
        dec = viterbi_decode(code_bsc)

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
    plt.title('Convolutional Code (K=3, R=1/2) Performance on AWGN', fontsize=14)
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
    plt.title('Convolutional Code (K=3, R=1/2) Performance on BSC', fontsize=14)
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
    test_data = np.array([1, 0, 1, 1])
    print("测试数据:", test_data)

    # 编码（自动添加尾比特）
    encoded = conv_encode(test_data)
    print("编码结果:", encoded)
    print("编码长度:", len(encoded), "bits")

    # 无噪声解码
    decoded = viterbi_decode(encoded)
    print("解码结果:", decoded)
    print("解码是否正确:", np.array_equal(decoded, test_data))

    # 添加噪声测试
    noisy_encoded = encoded.copy()
    noisy_encoded[3] = 1 - noisy_encoded[3]  # 翻转一个比特
    noisy_decoded = viterbi_decode(noisy_encoded)
    print("\n带噪声解码结果:", noisy_decoded)
    print("解码是否正确:", np.array_equal(noisy_decoded, test_data))

    # 运行完整仿真
    plot_conv_awgn()
    plot_conv_bsc()