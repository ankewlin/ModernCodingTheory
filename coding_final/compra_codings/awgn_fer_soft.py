import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import itertools


# ====================== BCH(15,7)定义 =======================

def setup_tables():
    exp_table = [1] * 15
    exp_table[0] = 1
    for i in range(1, 15):
        exp_table[i] = exp_table[i - 1] << 1
        if exp_table[i] & 0x10:
            exp_table[i] ^= 0x13
        exp_table[i] &= 0xF
    log_table = [0] * 16
    for i in range(15):
        log_table[exp_table[i]] = i
    return exp_table, log_table

exp_table, log_table = setup_tables()

def gf16_add(a, b):
    return a ^ b

def gf16_mul(a, b):
    if a == 0 or b == 0:
        return 0
    return exp_table[(log_table[a] + log_table[b]) % 15]

def gf16_div(a, b):
    if b == 0:
        raise ValueError("Division by zero")
    if a == 0:
        return 0
    return exp_table[(log_table[a] - log_table[b] + 15) % 15]

def bch_encode(message):
    g = [1, 1, 1, 0, 1, 0, 0, 0, 1]
    msg_ext = message + [0] * 8
    for i in range(7):
        if msg_ext[i] == 1:
            for j in range(len(g)):
                msg_ext[i + j] ^= g[j]
    parity = msg_ext[7:15]
    return message + parity

def calculate_syndromes(received):
    S1, S3 = 0, 0
    for j in range(15):
        if received[j] == 1:
            idx1 = (14 - j) % 15
            S1 = gf16_add(S1, exp_table[idx1])
            idx3 = (3 * (14 - j)) % 15
            S3 = gf16_add(S3, exp_table[idx3])
    return S1, S3

def bch_decode(received):
    S1, S3 = calculate_syndromes(received)
    if S1 == 0 and S3 == 0:
        return received[:7]
    if S1 != 0:
        S1_cubed = gf16_mul(S1, gf16_mul(S1, S1))
        if S3 == S1_cubed:
            error_pos = (14 - log_table[S1]) % 15
            corrected = received.copy()
            corrected[error_pos] ^= 1
            return corrected[:7]
    if S1 != 0:
        S1_squared = gf16_mul(S1, S1)
        sigma1 = S1
        S3_over_S1 = gf16_div(S3, S1)
        sigma2 = gf16_add(S1_squared, S3_over_S1)
        error_positions = []
        for k in range(15):
            alpha_k = exp_table[k]
            term1 = gf16_mul(sigma1, alpha_k)
            alpha_2k = gf16_mul(alpha_k, alpha_k)
            term2 = gf16_mul(sigma2, alpha_2k)
            result = gf16_add(1, gf16_add(term1, term2))
            if result == 0:
                j = (k-1) % 15
                error_positions.append(j)
        if len(error_positions) == 2:
            corrected = received.copy()
            for pos in error_positions:
                corrected[pos] ^= 1
            S1_new, S3_new = calculate_syndromes(corrected)
            if S1_new == 0 and S3_new == 0:
                return corrected[:7]
    return received[:7]

# Chase-II软判决BCH译码
def bch157_chase_decode(received_symbols):
    hard_bits = (received_symbols < 0).astype(int)
    reliabilities = np.abs(received_symbols)
    flip_candidates = np.argsort(reliabilities)[:3]
    best_metric = np.inf
    best_decoded = hard_bits[:7]
    for flips in itertools.chain.from_iterable(itertools.combinations(flip_candidates, r) for r in range(3)):
        candidate = hard_bits.copy()
        for pos in flips:
            candidate[pos] ^= 1
        decoded = bch_decode(candidate)
        re_encoded = bch_encode(list(decoded))
        re_encoded_symbols = 1 - 2 * np.array(re_encoded)
        metric = np.sum((received_symbols - re_encoded_symbols)**2)
        if metric < best_metric:
            best_metric = metric
            best_decoded = decoded
    return best_decoded

# ================ Hamming(7,4) 软判决版 =====================

G_hamming = np.array([
    [1, 0, 0, 0, 1, 1, 0],
    [0, 1, 0, 0, 1, 0, 1],
    [0, 0, 1, 0, 0, 1, 1],
    [0, 0, 0, 1, 1, 1, 1]
])

def hamming74_encode(data):
    return (np.dot(data, G_hamming) % 2).astype(int)

def hamming74_decode_soft(received_symbols):
    codebook = []
    for i in range(16):
        data_bits = np.array(list(np.binary_repr(i, 4))).astype(int)
        codeword = hamming74_encode(data_bits)
        codebook.append((data_bits, codeword))
    min_metric = np.inf
    best_data = None
    for data_bits, codeword in codebook:
        expected_symbols = 1 - 2 * codeword
        metric = np.sum((received_symbols - expected_symbols) ** 2)
        if metric < min_metric:
            min_metric = metric
            best_data = data_bits
    return best_data

# =============== R3软判决似然版 ================
def r3_encode(data):
    return np.repeat(data, 3)

def r3_decode_soft_ml(received):
    bits = []
    for i in range(0, len(received), 3):
        chunk = received[i:i+3]
        dist0 = np.sum((chunk - 1)**2)
        dist1 = np.sum((chunk + 1)**2)
        bits.append(int(dist1 < dist0))
    return np.array(bits)

# ============== 卷积码软判决版 ===============
class ConvolutionalCode:
    def __init__(self, constraint_length=3, polynomials=[0o7, 0o5]):
        self.K = constraint_length
        self.n = len(polynomials)
        self.num_states = 2 ** (self.K - 1)
        self.generators = []
        for poly in polynomials:
            bin_str = bin(poly)[2:].zfill(self.K)
            self.generators.append([int(b) for b in bin_str])

    def encode(self, input_bits):
        input_bits = list(input_bits) + [0] * (self.K - 1)
        register = [0] * (self.K - 1)
        output_bits = []
        for bit in input_bits:
            for gen in self.generators:
                out = bit * gen[0]
                for i in range(1, self.K):
                    out ^= register[i-1] * gen[i]
                output_bits.append(out)
            register = [bit] + register[:-1]
        return np.array(output_bits)

    def _branch_metrics_soft(self, state, input_bit, received):
        reg = [int(b) for b in bin(state)[2:].zfill(self.K-1)]
        outputs = []
        for gen in self.generators:
            out = input_bit * gen[0]
            for i in range(1, self.K):
                out ^= reg[i-1] * gen[i]
            outputs.append(out)
        expected = 1 - 2 * np.array(outputs)
        metric = np.sum((received - expected) ** 2)
        return metric

    def decode(self, received):
        num_steps = len(received) // self.n
        path_metrics = np.full(self.num_states, np.inf)
        path_metrics[0] = 0
        decoded_paths = [[] for _ in range(self.num_states)]

        for step in range(num_steps):
            rx = received[step*self.n:(step+1)*self.n]
            new_path_metrics = np.full(self.num_states, np.inf)
            new_paths = [[] for _ in range(self.num_states)]
            for state in range(self.num_states):
                if path_metrics[state] == np.inf:
                    continue
                for bit in [0, 1]:
                    next_state = (bit << (self.K-2)) | (state >> 1)
                    metric = self._branch_metrics_soft(state, bit, rx)
                    total_metric = path_metrics[state] + metric
                    if total_metric < new_path_metrics[next_state]:
                        new_path_metrics[next_state] = total_metric
                        new_paths[next_state] = decoded_paths[state] + [bit]
            path_metrics = new_path_metrics
            decoded_paths = new_paths

        final_state = np.argmin(path_metrics)
        decoded = decoded_paths[final_state]
        return np.array(decoded[:-(self.K-1)])

# =========== AWGN通道和BPSK调制 ==========
def bpsk_mod(bits):
    return 1 - 2 * bits

def awgn(signal, snr_db):
    snr = 10**(snr_db/10)
    sigma = np.sqrt(1/(2*snr))
    noise = sigma * np.random.randn(*signal.shape)
    return signal + noise

# ===============保存同级不同文件夹的图片函数=====================
def save_figure(filename, folder="../figures"):
    import os
    os.makedirs(folder, exist_ok=True)
    plt.savefig(os.path.join(folder, filename), dpi=300, bbox_inches='tight')

# ============ 统一仿真框架 ============
def simulate_all_awgn(snr_db, nblocks=200, block_len=4000):
    fer = {}
    stats = {'Hamming74': 0, 'R3': 0, 'Conv': 0, 'BCH157': 0}
    conv = ConvolutionalCode(constraint_length=3, polynomials=[0o7, 0o5])

    for _ in tqdm(range(nblocks), desc=f"SNR={snr_db}dB"):
        data = np.random.randint(0, 2, block_len)

        # Hamming74
        data_padded = data[:len(data)//4*4].reshape(-1, 4)
        code = np.vstack([hamming74_encode(block) for block in data_padded]).reshape(-1)
        mod = bpsk_mod(code)
        rx = awgn(mod, snr_db)
        rx_blocks = rx.reshape(-1, 7)
        decoded = np.vstack([hamming74_decode_soft(block) for block in rx_blocks]).reshape(-1)
        data_cut = data_padded.reshape(-1)
        frame_err = int(np.any(decoded != data_cut))
        stats['Hamming74'] += frame_err

        # R3
        code = r3_encode(data)
        mod = bpsk_mod(code)
        rx = awgn(mod, snr_db)
        decoded = r3_decode_soft_ml(rx)
        frame_err = int(np.any(decoded != data))
        stats['R3'] += frame_err

        # BCH(15,7)
        data_padded_bch = data[:len(data)//7*7].reshape(-1, 7)
        code_bch = np.vstack([bch_encode(list(block)) for block in data_padded_bch]).reshape(-1)
        mod = bpsk_mod(code_bch)
        rx = awgn(mod, snr_db)
        rx_blocks = rx.reshape(-1, 15)
        decoded = np.vstack([bch157_chase_decode(block) for block in rx_blocks]).reshape(-1)
        data_cut_bch = data_padded_bch.reshape(-1)
        frame_err = int(np.any(decoded != data_cut_bch))
        stats['BCH157'] += frame_err

        # 卷积码
        code = conv.encode(data)
        mod = bpsk_mod(code)
        rx = awgn(mod, snr_db)
        decoded = conv.decode(rx)
        data_cut = data[:len(decoded)]
        frame_err = int(np.any(decoded != data_cut))
        stats['Conv'] += frame_err

    for k in stats:
        fer[k] = stats[k] / nblocks
    return fer


# ============ 绘图模块 ============
def plot_all_awgn():
    snr_db_list = np.arange(-5, 4, 0.5)
    fer_dict = {k: [] for k in ['Hamming74', 'R3', 'Conv', 'BCH157']}

    for snr in snr_db_list:
        fer = simulate_all_awgn(snr, nblocks=200, block_len=4000)
        for k in fer_dict:
            fer_dict[k].append(fer[k])
        print(f"SNR={snr}dB:", {k: f"FER={fer[k]:.3e}" for k in fer})

    plt.figure(figsize=(6,6))
    for k, style in zip(fer_dict, ['-o','-s','-d','-^']):
        plt.semilogy(snr_db_list, fer_dict[k], style, label=f'{k} FER')
    plt.xlabel('SNR (dB)')
    plt.ylabel('FER')
    plt.title('FER Comparison of Different Codes on AWGN')
    plt.grid(True, which='both')
    plt.legend()
    plt.tight_layout()
    plt.ylim(1e-3, 1)
    save_figure("fer_awgn_comparison_minus5To35.png")
    plt.show()

# ============ 入口 ============
if __name__ == "__main__":
    plot_all_awgn()
