import numpy as np


class ConvolutionalCode:
    def __init__(self, constraint_length=7, rate=1 / 2, polynomials=[0o171, 0o133]):
        """
        初始化卷积码参数
        :param constraint_length: 约束长度 (K)
        :param rate: 码率 (r)
        :param polynomials: 生成多项式列表 (八进制表示)
        """
        self.K = constraint_length
        self.rate = rate
        self.n = int(1 / rate)  # 输出比特数
        self.num_states = 2 ** (self.K - 1)  # 状态数

        # 将八进制多项式转换为二进制连接向量
        self.generators = []
        for poly in polynomials:
            bin_str = bin(poly)[2:].zfill(self.K)
            self.generators.append([int(bit) for bit in bin_str])

        # 验证多项式长度
        if any(len(g) != self.K for g in self.generators):
            raise ValueError(f"所有生成多项式长度必须等于约束长度 {self.K}")

    def encode(self, input_bits):
        """
        卷积码编码器
        :param input_bits: 输入比特流 (0/1列表)
        :return: 编码后的比特流
        """
        # 添加尾比特 (K-1个0) 使状态归零
        input_bits = list(input_bits) + [0] * (self.K - 1)
        register = [0] * (self.K - 1)  # 移位寄存器初始化
        output_bits = []

        for bit in input_bits:
            # 计算当前时刻的n个输出比特
            for generator in self.generators:
                out_bit = bit * generator[0]  # 当前输入
                # 累加寄存器状态
                for i in range(1, self.K):
                    out_bit ^= register[i - 1] * generator[i]
                output_bits.append(out_bit)

            # 更新寄存器: 右移并插入新比特
            register = [bit] + register[:-1]

        return output_bits

    def _branch_metrics(self, state, input_bit, received):
        """
        计算分支度量 (汉明距离)
        :param state: 当前状态 (整数)
        :param input_bit: 输入比特 (0/1)
        :param received: 接收到的n个比特
        :return: 分支度量和输出比特
        """
        # 状态解码为寄存器值 (K-1个比特)
        register = [int(b) for b in bin(state)[2:].zfill(self.K - 1)]

        # 计算理论输出
        outputs = []
        for gen in self.generators:
            out_bit = input_bit * gen[0]
            for i in range(1, self.K):
                if i - 1 < len(register):
                    out_bit ^= register[i - 1] * gen[i]
            outputs.append(out_bit)

        # 计算汉明距离
        distance = sum(1 for a, b in zip(outputs, received) if a != b)
        return distance, outputs

    def decode(self, received_bits):
        """
        维特比解码器
        :param received_bits: 接收到的比特流 (含噪声)
        :return: 解码后的比特流 (移除尾比特)
        """
        num_steps = len(received_bits) // self.n
        # 初始化路径度量和历史记录
        path_metrics = np.full(self.num_states, np.inf)
        path_metrics[0] = 0  # 初始状态为全0
        survivor_paths = [[] for _ in range(self.num_states)]
        decoded_outputs = [[] for _ in range(self.num_states)]

        # 前向传递: 处理每个时间步
        for step in range(num_steps):
            # 当前时间步接收的n个比特
            rx_sym = received_bits[step * self.n: (step + 1) * self.n]
            new_path_metrics = np.full(self.num_states, np.inf)
            new_survivor_paths = [[] for _ in range(self.num_states)]
            new_decoded = [[] for _ in range(self.num_states)]

            # 处理每个当前状态
            for state in range(self.num_states):
                if path_metrics[state] == np.inf:
                    continue

                # 尝试两个可能的输入 (0和1)
                for input_bit in [0, 1]:
                    # 计算下一状态 (状态寄存器的移位操作)
                    next_state = (input_bit << (self.K - 2)) | (state >> 1)

                    # 计算分支度量和输出
                    distance, _ = self._branch_metrics(state, input_bit, rx_sym)
                    total_metric = path_metrics[state] + distance

                    # 更新下一状态的路径
                    if total_metric < new_path_metrics[next_state]:
                        new_path_metrics[next_state] = total_metric
                        new_survivor_paths[next_state] = survivor_paths[state] + [state]
                        new_decoded[next_state] = decoded_outputs[state] + [input_bit]

            # 更新为下一时间步
            path_metrics = new_path_metrics
            survivor_paths = new_survivor_paths
            decoded_outputs = new_decoded

        # 回溯: 选择最小度量的路径 (应为状态0)
        final_state = np.argmin(path_metrics)
        decoded_bits = decoded_outputs[final_state]

        # 移除尾比特 (最后K-1个比特)
        return decoded_bits[:-(self.K - 1)]


# 测试代码
if __name__ == "__main__":
    # 初始化卷积码 (K=7, r=1/2)
    cc = ConvolutionalCode(constraint_length=7, rate=1 / 2)

    # 生成随机输入数据
    np.random.seed(42)
    input_data = np.random.randint(0, 2, 2000)  # 20个随机比特

    # 编码
    encoded = cc.encode(input_data)
    print(f"原始数据: {input_data}")
    print(f"编码后长度: {len(encoded)}")

    # 模拟带噪声信道 (添加错误)
    noise = np.random.randint(0, 2, len(encoded))  # 随机噪声
    received = [(e + n) % 2 for e, n in zip(encoded, noise)]

    # 解码
    decoded = cc.decode(received)
    print(f"解码数据: {decoded}")

    # 验证解码正确性
    is_correct = np.array_equal(input_data, decoded)
    print(f"解码是否正确: {is_correct}")
    print(f"错误比特数: {np.sum(np.array(input_data) != np.array(decoded))}")