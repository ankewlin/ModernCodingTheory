# 定义GF(16)域的本原多项式：x^4 + x + 1
def setup_tables():
    # 指数表: exp_table[i] = α^i (i从0到14)
    exp_table = [1] * 15
    exp_table[0] = 1  # α^0
    for i in range(1, 15):
        exp_table[i] = exp_table[i - 1] << 1  # 乘以α（左移1位）
        if exp_table[i] & 0x10:  # 如果超过4位，模本原多项式
            exp_table[i] ^= 0x13  # 异或x^4 + x + 1 (二进制10011)
        exp_table[i] &= 0xF  # 保留低4位

    # 对数表: log_table[value] = i, 其中value = α^i
    log_table = [0] * 16
    for i in range(15):
        log_table[exp_table[i]] = i

    return exp_table, log_table


# 初始化全局表
exp_table, log_table = setup_tables()


# GF(16)乘法
def gf16_mul(a, b):
    if a == 0 or b == 0:
        return 0
    return exp_table[(log_table[a] + log_table[b]) % 15]


# GF(16)除法
def gf16_div(a, b):
    if b == 0:
        raise ValueError("Division by zero")
    if a == 0:
        return 0
    return exp_table[(log_table[a] - log_table[b] + 15) % 15]


# GF(16)加法 (异或)
def gf16_add(a, b):
    return a ^ b


# BCH编码函数
def bch_encode(message):
    # 生成多项式: g(x) = x^8 + x^7 + x^6 + x^4 + 1 (二进制:111010001)
    g = [1, 1, 1, 0, 1, 0, 0, 0, 1]  # 系数列表，从最高次项开始

    # 信息位后添加8个零
    msg_ext = message + [0] * 8

    # 多项式除法
    for i in range(7):  # 处理7位信息位
        if msg_ext[i] == 1:  # 如果首位为1
            for j in range(len(g)):
                msg_ext[i + j] ^= g[j]  # 异或生成多项式

    # 取后8位作为校验位
    parity = msg_ext[7:15]

    # 返回码字: 信息位 + 校验位
    return message + parity


# 计算伴随式
def calculate_syndromes(received):
    # 只计算S1和S3（S2和S4可以从它们推导）
    S1 = 0
    S3 = 0

    # 遍历接收码字的每一位
    for j in range(15):
        if received[j] == 1:  # 如果该位有值
            # S1 = ∑ r_j * α^(14-j)
            idx1 = (14 - j) % 15
            S1 = gf16_add(S1, exp_table[idx1])

            # S3 = ∑ r_j * α^(3*(14-j))
            idx3 = (3 * (14 - j)) % 15
            S3 = gf16_add(S3, exp_table[idx3])

    return S1, S3


# BCH译码函数
def bch_decode(received):
    # 步骤1: 计算伴随式
    S1, S3 = calculate_syndromes(received)

    # 步骤2: 检查伴随式（无错误）
    if S1 == 0 and S3 == 0:
        return received[:7]  # 直接返回信息位

    # 步骤3: 处理单错误情况
    if S1 != 0:
        # 计算S1^3
        S1_cubed = gf16_mul(S1, gf16_mul(S1, S1))

        # 检查单错误条件
        if S3 == S1_cubed:
            # 单错误: 错误位置 = 14 - log(S1)
            error_pos = (14 - log_table[S1]) % 15
            corrected = received.copy()
            corrected[error_pos] ^= 1  # 翻转错误位
            return corrected[:7]

    # 步骤4: 处理双错误情况
    if S1 != 0:
        # 计算S1^2
        S1_squared = gf16_mul(S1, S1)

        # 计算σ1 = S1
        sigma1 = S1

        # 计算σ2 = S1^2 + S3/S1
        S3_over_S1 = gf16_div(S3, S1)
        sigma2 = gf16_add(S1_squared, S3_over_S1)

        # 步骤5: 寻找错误位置
        error_positions = []
        for k in range(15):
            # 计算位置k对应的α的幂: α^k
            alpha_k = exp_table[k]

            # 计算σ(α^k) = 1 + σ1*α^k + σ2*α^(2k)
            term1 = gf16_mul(sigma1, alpha_k)
            alpha_2k = gf16_mul(alpha_k, alpha_k)
            term2 = gf16_mul(sigma2, alpha_2k)
            result = gf16_add(1, gf16_add(term1, term2))

            if result == 0:  # 找到根
                j =  (k-1) % 15
                error_positions.append(j)

        # 步骤6: 纠正错误
        if len(error_positions) == 2:
            corrected = received.copy()
            for pos in error_positions:
                corrected[pos] ^= 1  # 翻转错误位

            # 验证纠正是否成功
            S1_new, S3_new = calculate_syndromes(corrected)
            if S1_new == 0 and S3_new == 0:
                return corrected[:7]  # 返回信息位

    # 无法纠正的情况，返回原始信息位
    return received[:7]


# 测试代码
if __name__ == "__main__":
    # 示例1: 编码
    message = [1, 0, 1,1, 0, 0, 0]  # 信息位
    codeword = bch_encode(message)
    print("信息位:", message)
    print("编码后:", codeword)

    # 示例2: 无错误译码
    decoded_no_error = bch_decode(codeword)
    print("无错误译码结果:", decoded_no_error)

    # 示例3: 双错误译码 (在位置0和1引入错误)
    received = codeword.copy()
    received[2] ^= 1  # 添加错误
    received[0] ^= 1  # 添加错误
    print("接收码字 (含错误):", received)
    decoded_with_errors = bch_decode(received)
    print("双错误译码结果:", decoded_with_errors)

    # 示例4: 单个错误译码
    received_single = codeword.copy()
    received_single[3] ^= 1  # 添加单个错误
    print("接收码字 (单个错误):", received_single)
    decoded_single_error = bch_decode(received_single)
    print("单个错误译码结果:", decoded_single_error)

    # 示例5: 三错误译码 (应无法纠正)
    received_triple = codeword.copy()
    received_triple[0] ^= 1
    received_triple[1] ^= 1
    received_triple[2] ^= 1
    print("接收码字 (三错误):", received_triple)
    decoded_triple_error = bch_decode(received_triple)
    print("三错误译码结果 (应无法纠正):", decoded_triple_error)