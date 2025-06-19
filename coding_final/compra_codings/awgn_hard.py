import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

"""
==================== Convolutional Code 完整版 ====================
"""
class ConvolutionalCode:
    def __init__(self, constraint_length=3, rate=1 / 2, polynomials=[0o7, 0o5]):
        self.K = constraint_length
        self.rate = rate
        self.n = int(1 / rate)
        self.num_states = 2 ** (self.K - 1)

        self.generators = []
        for poly in polynomials:
            bin_str = bin(poly)[2:].zfill(self.K)
            self.generators.append([int(bit) for bit in bin_str])

        if any(len(g) != self.K for g in self.generators):
            raise ValueError(f"所有生成多项式长度必须等于约束长度 {self.K}")

    def encode(self, input_bits):
        input_bits = list(input_bits) + [0] * (self.K - 1)
        register = [0] * (self.K - 1)
        output_bits = []

        for bit in input_bits:
            for generator in self.generators:
                out_bit = bit * generator[0]
                for i in range(1, self.K):
                    out_bit ^= register[i - 1] * generator[i]
                output_bits.append(out_bit)
            register = [bit] + register[:-1]

        return np.array(output_bits)

    def _branch_metrics(self, state, input_bit, received):
        register = [int(b) for b in bin(state)[2:].zfill(self.K - 1)]
        outputs = []
        for gen in self.generators:
            out_bit = input_bit * gen[0]
            for i in range(1, self.K):
                if i - 1 < len(register):
                    out_bit ^= register[i - 1] * gen[i]
            outputs.append(out_bit)
        distance = sum(1 for a, b in zip(outputs, received) if a != b)
        return distance, outputs

    def decode(self, received_bits):
        num_steps = len(received_bits) // self.n
        path_metrics = np.full(self.num_states, np.inf)
        path_metrics[0] = 0
        decoded_outputs = [[] for _ in range(self.num_states)]

        for step in range(num_steps):
            rx_sym = received_bits[step * self.n: (step + 1) * self.n]
            new_path_metrics = np.full(self.num_states, np.inf)
            new_decoded = [[] for _ in range(self.num_states)]

            for state in range(self.num_states):
                if path_metrics[state] == np.inf:
                    continue

                for input_bit in [0, 1]:
                    next_state = (input_bit << (self.K - 2)) | (state >> 1)
                    distance, _ = self._branch_metrics(state, input_bit, rx_sym)
                    total_metric = path_metrics[state] + distance

                    if total_metric < new_path_metrics[next_state]:
                        new_path_metrics[next_state] = total_metric
                        new_decoded[next_state] = decoded_outputs[state] + [input_bit]

            path_metrics = new_path_metrics
            decoded_outputs = new_decoded

        final_state = np.argmin(path_metrics)
        decoded_bits = decoded_outputs[final_state]

        if len(decoded_bits) > (self.K - 1):
            return decoded_bits[:-(self.K - 1)]
        else:
            return decoded_bits

"""
===================== 其余编码保持不变 ==========================
"""
G_hamming = np.array([
    [1, 0, 0, 0, 1, 1, 0],
    [0, 1, 0, 0, 1, 0, 1],
    [0, 0, 1, 0, 0, 1, 1],
    [0, 0, 0, 1, 1, 1, 1]
])
H_hamming = np.array([
    [1, 1, 0, 1, 1, 0, 0],
    [1, 0, 1, 1, 0, 1, 0],
    [0, 1, 1, 1, 0, 0, 1]
])
syndrome_table = {
    (0, 0, 0): None, (0, 0, 1): 6, (0, 1, 0): 5, (0, 1, 1): 2,
    (1, 0, 0): 4, (1, 0, 1): 0, (1, 1, 0): 1, (1, 1, 1): 3
}
def hamming74_encode(data):
    return (np.dot(data, G_hamming) % 2).astype(int)
def hamming74_decode(codeword):
    s = np.dot(codeword, H_hamming.T) % 2
    synd = tuple(s)
    codeword_corr = np.copy(codeword)
    if synd in syndrome_table and syndrome_table[synd] is not None:
        idx = syndrome_table[synd]
        codeword_corr[idx] ^= 1
    return codeword_corr[:4]
def bch74_encode(data):
    return (np.dot(data, G_hamming) % 2).astype(int)
def bch74_decode(received):
    s = np.dot(received, H_hamming.T) % 2
    return received[:4]
def r3_encode(data):
    return np.repeat(data, 3)
def r3_decode_soft(received):
    bits = []
    for i in range(0, len(received), 3):
        chunk = received[i:i+3]
        bits.append(int(np.sum(chunk) < 0))  # soft decision
    return np.array(bits)

"""
======================= 调制与信道 =========================
"""
def bpsk_mod(bits):
    return 1 - 2 * bits
def bpsk_demod(symbols):
    return (symbols < 0).astype(int)
def awgn(signal, snr_db):
    snr = 10 ** (snr_db / 10)
    sigma = np.sqrt(1 / (2 * snr))
    noise = sigma * np.random.randn(*signal.shape)
    return signal + noise

"""
==================== 统一仿真主循环 ===========================
"""
def simulate_all_awgn(snr_db, nblocks=200, block_len=200):
    ber = {}
    fer = {}
    stats = { 'Hamming74': [0,0,0], 'R3': [0,0,0], 'BCH74': [0,0,0], 'Conv': [0,0,0] }

    conv = ConvolutionalCode(constraint_length=3, polynomials=[0o7, 0o5])

    for _ in tqdm(range(nblocks), desc=f"SNR={snr_db}dB"):
        data = np.random.randint(0, 2, block_len)

        # Hamming(7,4)
        data_padded = data[:len(data)//4*4].reshape(-1,4)
        code = np.vstack([hamming74_encode(block) for block in data_padded]).reshape(-1)
        mod = bpsk_mod(code)
        rx = awgn(mod, snr_db)
        hard = bpsk_demod(rx)
        decoded = np.vstack([hamming74_decode(block) for block in hard.reshape(-1,7)]).reshape(-1)
        data_cut = data_padded.reshape(-1)
        nbit_err = np.sum(decoded != data_cut)
        stats['Hamming74'][0] += nbit_err
        stats['Hamming74'][1] += int(nbit_err>0)
        stats['Hamming74'][2] += len(data_cut)

        # R3
        code = r3_encode(data)
        mod = bpsk_mod(code)
        rx = awgn(mod, snr_db)
        decoded = r3_decode_soft(rx)
        nbit_err = np.sum(decoded != data)
        stats['R3'][0] += nbit_err
        stats['R3'][1] += int(nbit_err>0)
        stats['R3'][2] += len(data)

        # BCH74 (用Hamming同样的方式)
        code = np.vstack([bch74_encode(block) for block in data_padded]).reshape(-1)
        mod = bpsk_mod(code)
        rx = awgn(mod, snr_db)
        hard = bpsk_demod(rx)
        decoded = np.vstack([bch74_decode(block) for block in hard.reshape(-1,7)]).reshape(-1)
        nbit_err = np.sum(decoded != data_cut)
        stats['BCH74'][0] += nbit_err
        stats['BCH74'][1] += int(nbit_err>0)
        stats['BCH74'][2] += len(data_cut)

        # Conv
        code = conv.encode(data)
        mod = bpsk_mod(code)
        rx = awgn(mod, snr_db)
        hard = bpsk_demod(rx)
        decoded = conv.decode(hard)
        data_cut = data[:len(decoded)]
        nbit_err = np.sum(decoded != data_cut)
        stats['Conv'][0] += nbit_err
        stats['Conv'][1] += int(nbit_err>0)
        stats['Conv'][2] += len(data_cut)

    for key in stats:
        nerr, nfer, ntotal = stats[key]
        ber[key] = nerr/ntotal if ntotal else 0
        fer[key] = nfer/nblocks if nblocks else 0
    return ber, fer
"""
========================= 绘图部分 ===========================
"""
def plot_all_awgn():
    snr_db_list = np.arange(-5, 5, 1)
    ber_dict = {k:[] for k in ['Hamming74', 'R3', 'BCH74', 'Conv']}
    fer_dict = {k:[] for k in ['Hamming74', 'R3', 'BCH74', 'Conv']}
    for snr in snr_db_list:
        ber, fer = simulate_all_awgn(snr, nblocks=200)
        for k in ber_dict:
            ber_dict[k].append(ber[k])
            fer_dict[k].append(fer[k])
        print(f"SNR={snr}dB:", {k: f"{ber[k]:.2e}/{fer[k]:.2e}" for k in ber})

    plt.figure()
    for k,style in zip(ber_dict, ['-o','-s','-^','-d']):
        plt.semilogy(snr_db_list, ber_dict[k], style, label=f'{k} BER')
    plt.xlabel('SNR (dB)')
    plt.ylabel('BER')
    plt.title('BER of Different Codes on AWGN ')
    plt.grid(True, which='both')
    plt.legend()
    plt.tight_layout()
    plt.ylim(1e-5, 1)
    plt.show()

    plt.figure()
    for k,style in zip(fer_dict, ['-o','-s','-^','-d']):
        plt.semilogy(snr_db_list, fer_dict[k], style, label=f'{k} FER')
    plt.xlabel('SNR (dB)')
    plt.ylabel('FER')
    plt.title('FER of Different Codes on AWGN ')
    plt.grid(True, which='both')
    plt.legend()
    plt.ylim(1e-5, 1)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_all_awgn()
