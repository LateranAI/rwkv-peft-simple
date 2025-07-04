"""
文件说明:
    项目内部通用工具 - 文本 Tokenizer 与采样工具集合。

功能角色:
    - 支持两种模式: charMode(字符级自定义 tokenizer) 与 wordPiece/BPE 模式 (由 HuggingFace Tokenizer 提供)。
    - 提供 `Tokenizer` 类封装常用文本预处理、采样逻辑 (top-p、温度、多设备兼容)。
    - 附带若干素数检验函数 (Fermat / Miller-Rabin) 供随机性或加密场景使用。

核心依赖:
    - torch / numpy: 数值计算及随机采样
    - transformers(GPT2TokenizerFast/PreTrainedTokenizerFast): 非字符模式 tokenizer 加载

使用示例:
    >>> tok = Tokenizer(['tokenizer.json', 'vocab.json'])
    >>> ids = tok.tokenizer.encode("Hello")
    >>> text = tok.tokenizer.decode(ids)

副作用说明:
    - `record_time` 会在全局字典 `time_slot` 中记录最短执行时间，用于性能统计。
"""
import json, time, random, os
import numpy as np
import torch
from torch.nn import functional as F

time_slot = {}
time_ref = time.time_ns()

def record_time(name):
    """记录当前时间段到全局字典

    参数:
        name (str): 需要统计的时间段名称
    副作用:
        更新/写入 `time_slot` 全局字典中对应键的最小耗时值 (秒)
    """
    if name not in time_slot:
        time_slot[name] = 1e20
    tt = (time.time_ns() - time_ref) / 1e9
    if tt < time_slot[name]:
        time_slot[name] = tt

class Tokenizer:
    """通用 Tokenizer 封装类

    功能:
        - 根据 `WORD_NAME` 的类型决定字符级或 HuggingFace Tokenizer 模式
        - 提供 `refine_context` 用于清洗多行上下文
        - 提供 `sample_logits` 根据 logits 执行 temperature + top-p 采样

    参数:
        WORD_NAME (Union[str, List[str]]): 字符模式时为词表基名; 否则为 [tokenizer.json, vocab.json] 等文件列表
        UNKNOWN_CHAR (str): 字符模式下未知字符占位符, 默认为 U+E083

    属性:
        charMode (bool): 是否为字符模式
        tokenizer: 非字符模式下的 HuggingFace Tokenizer 实例
        stoi / itos (dict): 字符↔索引映射 (仅字符模式)
        vocab_size (int): 词表大小
    """
    def __init__(self, WORD_NAME, UNKNOWN_CHAR='\ue083'):
        if 'list' in str(type(WORD_NAME)):
            self.charMode = False
            if WORD_NAME[0] == WORD_NAME[1]:
                from transformers import PreTrainedTokenizerFast
                self.tokenizer = PreTrainedTokenizerFast(tokenizer_file=WORD_NAME[0])
            else:
                from transformers import GPT2TokenizerFast
                self.tokenizer = GPT2TokenizerFast(WORD_NAME[0], WORD_NAME[1])
            self.vocab_size = len(self.tokenizer)
        else:
            self.charMode = True
            with open(WORD_NAME + '.json', "r", encoding="utf-16") as result_file:
                self.word_table = json.load(result_file)

            self.vocab_size = len(self.word_table)

            self.stoi = {v: int(k) for k, v in self.word_table.items()}
            self.itos = {int(k): v for k, v in self.word_table.items()}

            self.UNKNOWN_CHAR = self.stoi[UNKNOWN_CHAR]

    def refine_context(self, context):
        """清洗多行上下文字符串

        去除首尾空格/全角空格/回车, 去掉空行, 并确保返回值以换行符开头。

        参数:
            context (str): 原始上下文文本 (可含多行)
        返回:
            str: 清洗后的上下文
        """
        context = context.strip().split('\n')
        for c in range(len(context)):
            context[c] = context[c].strip().strip('\u3000').strip('\r')
        context = list(filter(lambda c: c != '', context))
        context = '\n' + ('\n'.join(context)).strip()
        if context == '':
            context = '\n'
        return context

    def sample_logits(self, out, x, ctx_len, temperature=1.0, top_p_usual=None, top_p_newline=None):
        """基于 logits 执行 top-p + 温度采样

        根据最近一次采样的 token 是否为换行符决定使用 `top_p_newline` 还是 `top_p_usual`。
        当 `train_config.accelerator` 为 cpu 时, 使用 numpy 采样; 否则使用 torch CUDA 采样。

        参数:
            out (Tensor): 当前 step 的 logits, shape = (vocab_size,)
            x (Tensor): 输入 token 序列
            ctx_len (int): 上下文长度 (目前未直接使用)
            temperature (float): 温度系数, 1.0 为不变
            top_p_usual (float): 非换行情况下的 top-p
            top_p_newline (float): 最近 token 为换行时的 top-p
        返回:
            int: 采样得到的下一个 token id
        """
        # out[self.UNKNOWN_CHAR] = -float('Inf')
        lastChar = int(x[-1])

        probs = F.softmax(out, dim=-1)

        if self.charMode:
            if self.itos[lastChar] == '\n':
                top_p = top_p_newline
            else:
                top_p = top_p_usual
        else:
            top_p = top_p_usual

        from src.configs.train import train_config
        if train_config.accelerator == "cpu":
            probs = probs.numpy()
            sorted_probs = np.sort(probs)[::-1]
            cumulative_probs = np.cumsum(sorted_probs)
            cutoff = float(sorted_probs[np.argmax(cumulative_probs > top_p)])
            probs[probs < cutoff] = 0
            if temperature != 1.0:
                probs = probs.pow(1.0 / temperature)
            probs = probs / np.sum(probs)
            out = np.random.choice(a=len(probs), p=probs)
            return out
        else:
            sorted_probs = torch.sort(probs, descending=True)[0]
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1).cpu().numpy()
            cutoff = float(sorted_probs[np.argmax(cumulative_probs > top_p)])
            probs[probs < cutoff] = 0
            if temperature != 1.0:
                probs = probs.pow(1.0 / temperature)
            out = torch.multinomial(probs, num_samples=1)[0]
            return out

def MaybeIsPrime(number):
    """判断整数是否可能为素数 (Fermat + Miller-Rabin 简易联合测试)

    参数:
        number (int): 待检测整数
    返回:
        bool: True 表示可能是素数, False 表示必然不是素数
    """
    if FermatPrimalityTest(number) and MillerRabinPrimalityTest(number):
        return True
    else:
        return False


def FermatPrimalityTest(number):
    """Fermat (费马) 素数概率检测

    参数:
        number (int): 待检测整数
    返回:
        bool: 通过费马小定理的概率测试则为 True
    """
    if number > 1:
        for time in range(3):
            randomNumber = random.randint(2, number) - 1
            if pow(randomNumber, number - 1, number) != 1:
                return False
        return True
    else:
        return False


def MillerRabinPrimalityTest(number):
    """Miller-Rabin 素数概率测试

    参数:
        number (int): 待检测整数
    返回:
        bool: 通过 Miller-Rabin 测试则为 True
    """
    if number == 2:
        return True
    elif number == 1 or number % 2 == 0:
        return False
    oddPartOfNumber = number - 1
    timesTwoDividNumber = 0
    while oddPartOfNumber % 2 == 0:
        oddPartOfNumber = oddPartOfNumber // 2
        timesTwoDividNumber = timesTwoDividNumber + 1

    for time in range(3):
        while True:
            randomNumber = random.randint(2, number) - 1
            if randomNumber != 0 and randomNumber != 1:
                break

        randomNumberWithPower = pow(randomNumber, oddPartOfNumber, number)

        if (randomNumberWithPower != 1) and (randomNumberWithPower != number - 1):
            iterationNumber = 1

            while (iterationNumber <= timesTwoDividNumber - 1) and (randomNumberWithPower != number - 1):
                randomNumberWithPower = pow(randomNumberWithPower, 2, number)
                iterationNumber = iterationNumber + 1
            if randomNumberWithPower != (number - 1):
                return False

    return True
