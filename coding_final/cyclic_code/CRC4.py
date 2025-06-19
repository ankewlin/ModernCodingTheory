import numpy as np
import matplotlib.pyplot as plt
"""
生成多项式（Generator Polynomial）：g(x)=x^4+x+1

二进制表示：0𝑏10011 或 十进制 19

冗余位数（r）：4（即CRC-4）

信息位数（k）：4（每帧用4位信息）

码字长度（n）：8（4位信息 + 4位CRC校验）
"""
# ===================== CRC-4 编码与检测 =======================

GEN = 0b10011  # x^4 + x + 1

def crc4_encode(data):
    """
    data: shape (4,) 一组信息比特
    return: 长度8的码字（信息位+CRC校验4位）
    """
    m = int(''.join(str(b) for b in data), 2)
    m_shift = m << 4  # 左移4位，为生成4位校验
    val = m_shift
    for i in reversed(range(4, 8)):
        if (val >> i) & 1:
            val ^= (GEN << (i - 4))
    crc = val & 0xF  # 取后4位余数
    codeword = (m << 4) | crc
    # 转成8位二进制数组
    return np.array([int(b) for b in f"{codeword:08b}"])

def crc4_decode(codeword):
    """
    codeword: shape (8,)
    return: 长度4的解码信息比特
    检错型：只做错误检测，不纠错
    """
    c = int(''.join(str(b) for b in codeword), 2)
    val = c
    for i in reversed(range(4, 8)):
        if (val >> i) & 1:
            val ^= (GEN << (i - 4))
    if val & 0xF == 0:  # 校验通过
        # 返回前4位（信息位）
        return np.array([int(b) for b in f"{c>>4:04b}"])
    else:
        # 检测到错误，实际应用可返回全0或原始信息位或其他标记
        # 这里为统计误码，直接输出前4位（你也可改成全0/全1/报错等）
        return np.array([int(b) for b in f"{c>>4:04b}"])

# ======================= BPSK 调制/解调 ============================

def bpsk_mod(bits):
    return 1 - 2 * bits  # 0 -> +1, 1 -> -1

def bpsk_demod(symbols):
    return (symbols < 0).astype(int) # 硬判决:小于0判为1，大于零判为0

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

def simulate_awgn(snr_db, nblocks=100000):
    nerr, nfer, ntotal = 0, 0, 0
    fixed_seq = np.array([1, 1, 0, 1])  # 固定信源比特序列
    for _ in range(nblocks):
        data = fixed_seq
        code = crc4_encode(data)
        mod = bpsk_mod(code)
        rx = awgn(mod, snr_db)
        hard = bpsk_demod(rx)
        dec = crc4_decode(hard)
        nbit_err = np.sum(dec != data)
        nerr += nbit_err
        nfer += int(nbit_err > 0)
        ntotal += 4
    return nerr / ntotal, nfer / nblocks

def simulate_bsc(p, nblocks=100000):
    nerr, nfer, ntotal = 0, 0, 0
    fixed_seq = np.array([1, 1, 0, 1])  # 固定信源比特序列
    for _ in range(nblocks):
        data = fixed_seq
        code = crc4_encode(data)
        code_bsc = bsc(code, p)
        dec = crc4_decode(code_bsc)
        nbit_err = np.sum(dec != data)
        nfer += int(nbit_err > 0)
        nerr += nbit_err
        ntotal += 4
    return nerr / ntotal, nfer / nblocks

# ======================= 绘制性能曲线 =============================

def plot_awgn():
    snr_db_list = np.arange(0, 9, 1)
    ber, fer = [], []
    for snr in snr_db_list:
        b, f = simulate_awgn(snr, nblocks=100000)
        ber.append(b)
        fer.append(f)
        print(f"SNR={snr}dB, BER={b:.3e}, FER={f:.3e}")
    plt.figure()
    plt.semilogy(snr_db_list, ber, 'o-', label='BER')
    plt.semilogy(snr_db_list, fer, 's--', label='FER')
    plt.xlabel('SNR (dB)')
    plt.ylabel('Error Rate')
    plt.title('CRC-4 Code on AWGN')
    plt.grid(True, which='both')
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_bsc():
    p_list = np.logspace(-4, -1, 8)
    ber, fer = [], []
    for p in p_list:
        b, f = simulate_bsc(p, nblocks=100000)
        ber.append(b)
        fer.append(f)
        print(f"p={p:.2e}, BER={b:.3e}, FER={f:.3e}")
    plt.figure()
    plt.semilogy(p_list, ber, 'o-', label='BER')
    plt.semilogy(p_list, fer, 's--', label='FER')
    plt.xlabel('BSC Flip Probability')
    plt.ylabel('Error Rate')
    plt.title('CRC-4 Code on BSC')
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
