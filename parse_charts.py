import os.path
import pickle
import re

import pandas as pd
import torch
from matplotlib import pyplot as plt
from torch.nn.utils.rnn import pad_sequence
from glob import glob
from tqdm import tqdm

# --- 1. 定义所有可能的元素列表 ---

# 判定区列表生成
# 1-8

list_num = [str(i) for i in range(1, 9)]
# A1-E8 (C只有C1, C2)
list_alpha = [f'{l}{i}' for l in 'ABDE' for i in range(1, 9)] + \
             [f'C{i}' for i in range(1, 3)]

all_regions = list_num + list_alpha

# 【关键修改】按长度降序排列，确保先匹配 'A1' 而不是 'A' (虽然此场景下无所谓，但习惯更好)
# 并转义以防万一
all_regions.sort(key=len, reverse=True)

# 构建判定区正则：(A1|A2|...|1|2|...)
# 注意：这里不再在两端加 \b，因为 '3bx' 中 '3' 和 'b' 之间没有边界
REGION_PATTERN = '|'.join(all_regions)

# 特性 (Trait): b, x, f。允许出现多次，例如 'bx'
TRAIT_CHARS = r'[bxf]'
# 匹配 0 次或多次特性字符
TRAIT_PATTERN = rf'(?P<trait1>{TRAIT_CHARS}*)'

# 形状 (Shape)
SHAPE_PATTERN = r'(pp|qq|p|q|h|-|<|>|\^|v|V|w|s|z)'
SHAPE_PATTERN_DICT = ['-', '<', '>', 'p', 'q', '^', 'v', 'V', 'w', 's', 'z', 'pp', 'qq', 'h']

# 时间格式: [num1:num2]
TIME_PATTERN = r'\[\d+:\d+\]'

# 尾部特性: 同样的 b, x, f 组合
TRAIT2_PATTERN = rf'(?P<trait2>{TRAIT_CHARS}*)'
# --- 2. 预编译主正则 (前缀提取) ---
# 逻辑：
# ^\s* : 开头空格
# (?P<region>...) : 匹配列表中的任意一个判定区
# (?P<trait1>...) : 紧接着匹配任意个特性字符 (贪婪匹配判定区后，剩下的就是特性)
# (?P<rest>.*)    : 捕获剩余字符串
PREFIX_REGEX = re.compile(
    rf'^\s*(?P<region>{REGION_PATTERN}){TRAIT_PATTERN}(?P<rest>.*)$'
)

# --- 3. 预编译后续正则 ---
# 匹配: 接续形状 + 时间 + 尾部特性
HEAD_REGEX = re.compile(
    r'^\s*'
    r'(?P<continuation_shape>\d+(?:-\d+)*)\s*'
    rf'{TRAIT2_PATTERN}\s*'
    rf'(?P<time>{TIME_PATTERN})\s*'
    rf'(?P<trait2_1>{TRAIT_CHARS}*)$'  # 必须匹配到结尾
)

# 尾部正则：也不包含 '*' (因为 split 已经把它拿走了)
TAIL_PART_REGEX = re.compile(
    r'^\s*'
    rf'(?P<shape_2>{SHAPE_PATTERN})\s*'
    r'(?P<continuation_shape_2>\d+(?:-\d+)*)\s*'
    rf'(?P<trait2_21>{TRAIT_CHARS}*)\s*'
    rf'(?P<time_2>{TIME_PATTERN})\s*'
    rf'(?P<trait2_22>{TRAIT_CHARS}*)\s*$'  # 必须匹配到结尾
)

current_time, bpm = 0, -1


def parse_continuation_shape(continuation_shape, start_time, shared_time, t_tail):
    lane = continuation_shape[0]

    results = []
    if len(continuation_shape) == 1:
        return [(start_time + shared_time, t_tail + 'se', lane, shared_time, '0')]
    if continuation_shape[1].isdigit():
        results.append((start_time + shared_time, 'sm', lane, shared_time, '0'))
        res = parse_continuation_shape(continuation_shape[1:], start_time + shared_time, shared_time, t_tail)
    elif len(continuation_shape) >= 4 and continuation_shape[1:3] in SHAPE_PATTERN_DICT:
        results.append((start_time + shared_time, 'sm', lane, shared_time, continuation_shape[1:3]))
        res = parse_continuation_shape(continuation_shape[3:], start_time + shared_time, shared_time, t_tail)
    elif continuation_shape[1] in SHAPE_PATTERN_DICT:
        results.append((start_time + shared_time, 'sm', lane, shared_time, continuation_shape[1:2]))
        res = parse_continuation_shape(continuation_shape[2:], start_time + shared_time, shared_time, t_tail)
    else:
        print(continuation_shape)
        raise ValueError
    results.extend(res)
    return results


def parse_continuation_split(text, cur_time):
    results = []

    # --- 步骤 1：使用 '*' 分割字符串 ---
    parts = text.split('*')

    # --- 步骤 2：处理第一部分 (必需部分) ---
    head_text = parts[0]
    head_match = HEAD_REGEX.match(head_text)

    if not head_match:
        return None

    match = re.search(r"\[(\d+):(\d+)\]", head_match['time'])
    if match:
        a_str = match.group(1)
        b_str = match.group(2)
        a, b = int(a_str), int(b_str)
        duration = b * bpm / a
    else:
        raise ValueError

    N = len(re.findall(r'\d', head_match['continuation_shape']))
    t_tail = head_match['trait2'] + head_match['trait2_1']
    stack_res = parse_continuation_shape(head_match['continuation_shape'], cur_time, duration / N, t_tail)
    results.extend(stack_res)

    # --- 步骤 3：处理后续部分 (可选部分) ---
    # 遍历 parts[1:]，即除了第一个之外的所有部分
    for index, part_text in enumerate(parts[1:], start=1):
        # 去除首尾空白，防止 ' * ' 这种写法导致的空白干扰
        part_text = part_text.strip()

        # 如果是空字符串（例如字符串末尾多写了一个 *），可以选择跳过或报错
        if not part_text:
            continue

        tail_match = TAIL_PART_REGEX.match(part_text)

        if not tail_match:
            continue

        match = re.search(r"\[(\d+):(\d+)\]", tail_match['time_2'])
        if match:
            a_str = match.group(1)
            b_str = match.group(2)
            a, b = int(a_str), int(b_str)
            duration = b * bpm / a
        else:
            continue

        N = len(re.findall(r'\d', head_match['continuation_shape']))
        t_tail = tail_match['trait2_21'] + tail_match['trait2_22']
        stack_res = parse_continuation_shape(tail_match['continuation_shape_2'], cur_time, duration / N, t_tail)
        results.extend(stack_res)

    return results


def parse_complex_string(input_str: str) -> list:
    # Time (时间点), Type (类型), Lane (轨道/位置), Duration (持续时长), Shape (形状):

    global current_time, bpm
    # --- 4. 解析逻辑 ---
    # 匹配bpm
    match = re.match(r'(\(\d{1,5}(\.\d+)?\))?', input_str)
    if match.group(1):
        p_str = match.group(1)
        bpm = float(p_str[1:-1])
        input_str = input_str[match.end():]
    # 匹配基础分音
    match = re.match(r'^\{(\d{1,4})\}', input_str)
    if match:
        p_str = match.group(1)
        TIME_BASE = int(p_str)
        rest_str = input_str[match.end():]
    else:
        return []

    tokens = re.split(r'([/,])', rest_str)
    results = []

    for i in range(0, len(tokens), 2):
        part_str = tokens[i].strip()
        delimiter = tokens[i + 1] if i + 1 < len(tokens) else None
        adder = delimiter == ','

        if not part_str:
            current_time += bpm / TIME_BASE if adder else 0
            continue

        # Step 1: 提取判定区和前置特性
        match = PREFIX_REGEX.match(part_str)
        if not match:
            continue

        lane = match.group('region')
        # 如果 trait1 是空字符串，设为 None
        trait = match.group('trait1') if match.group('trait1') else ''
        trait = trait.rstrip('f')
        rest_str = match.group('rest').strip()

        if not rest_str:
            # 返回tap
            results.append((current_time, trait + 't', lane, 0, '0'))
            current_time += bpm / TIME_BASE if adder else 0
            continue

        # Step 2: 提取形状
        shape_match = re.match(SHAPE_PATTERN, rest_str)
        if not shape_match:
            continue

        shape = shape_match.group(0)
        rest_of_shape = rest_str[len(shape):].strip()

        # Step 3: 根据形状处理剩余部分

        # Case A: 形状 'h' -> 必须有时间，无接续形状
        if shape == 'h':
            # 匹配纯时间
            time_match = re.match(rf'^\s*({TIME_PATTERN})\s*$', rest_of_shape)
            if time_match:
                duration = time_match.group(1)
                match = re.search(r"\[(\d+):(\d+)\]", duration)
                if match:
                    a_str = match.group(1)
                    b_str = match.group(2)
                    a, b = int(a_str), int(b_str)
                    duration = b * bpm / a
                else:
                    raise ValueError
            else:
                raise ValueError
            # 返回hold
            results.append((current_time, trait + 'h', lane, duration, '0'))
            current_time += bpm / TIME_BASE
            continue

        # Case B: 普通形状 -> 接续形状 + 时间 + 可选特性
        res = parse_continuation_split(rest_of_shape, current_time)
        if res is None:
            continue
        results.extend(res)

        # 返回单条星星（不包括V）
        results.append((current_time, trait + 's', lane, 0, shape))
        current_time += bpm / TIME_BASE if adder else 0
    results = sorted(results, key=lambda x: x[0])
    return results


# time,type,lane,dur,shape
type_dict = ['t', 'bt', 'xt', 'bxt', 'h', 'bh', 'xh', 'bxh', 's', 'bs', 'xs',
             'bxs', 'se', 'bse', 'sm', 'bsm']
type_dict = {l: i + 1 for i, l in enumerate(type_dict)}
lane_dict = ['1', '2', '3', '4', '5', '6', '7', '8', 'A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'B1', 'B2',
             'B3',
             'B4', 'B5', 'B6', 'B7', 'B8', 'C1', 'C2', 'D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'E1', 'E2',
             'E3',
             'E4', 'E5', 'E6', 'E7', 'E8']
lane_dict = {l: i + 1 for i, l in enumerate(lane_dict)}
shape_dict = ['0', '-', '<', '>', 'p', 'q', '^', 'v', 'V', 'w', 's', 'z', 'pp', 'qq', 'h']
shape_dict = {s: i + 1 for i, s in enumerate(shape_dict)}


def encode_data(data):
    encoded_batch = []
    for bar in data:
        if len(bar) == 0:
            continue
        bar_list = []
        for note in bar:
            bar_list.append(torch.tensor([note[0], type_dict[note[1]], lane_dict[note[2]], note[3] + 1,
                                          shape_dict[note[4]]], dtype=torch.long))
        encoded_batch.append(torch.stack(bar_list))
    return encoded_batch


def parse_chart_info(path):
    name = path.split('\\')[-2]
    _id = name.split('_')[0]
    name = '_'.join(name.split('_')[1:])

    expert_start_marker, master_start_marker, remaster_start_marker = "&inote_4=", "&inote_5=", "&inote_6="
    expert_flag, master_flag, remaster_flag = 0, 0, 0
    end_marker = "E"
    expert, master, remaster = [], [], []
    is_recording = False

    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            stripped_line = line.strip()

            # 1. 检查是否到达结束标记 "E" (单独一行)
            if is_recording and stripped_line == end_marker:
                is_recording = False
                if expert_flag == 0:
                    expert_flag = 1
                elif master_flag == 0:
                    master_flag = 1
                elif remaster_flag == 0:
                    remaster_flag = 1
                continue  # 找到E，停止读取

            # 2. 如果处于记录状态，保存该行
            if is_recording and expert_flag == 0:
                expert.append(line)
            elif is_recording and master_flag == 0:
                master.append(line)
            elif is_recording and remaster_flag == 0:
                remaster.append(line)

            # 3. 检查是否到达开始标记 (放在最后是为了不包含起始行本身)
            if expert_start_marker in stripped_line:
                is_recording = True
            elif master_start_marker in stripped_line:
                is_recording = True
            elif remaster_start_marker in stripped_line:
                is_recording = True

    # expert = "\n".join(expert)
    # master = "\n".join(master)
    # remaster = "\n".join(remaster)

    return path, _id, name, expert, master, remaster


def get_levels(all_chart_path):
    target_keys = ['4', '5', '6']
    level_map = {'4': 'expert', '5': 'master', '6': 'remaster'}
    all_data_file = []
    all_data_level = []

    for filepath in (all_chart_path):
        name = filepath.split('\\')[-2]
        _id = name.split('_')[0]
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

            for key in target_keys:
                pattern = f"&lv_{key}=([^&\s]*)"
                match = re.search(pattern, content)

                if match:
                    # 获取捕获的值
                    value = match.group(1)
                    if value == "":
                        continue
                    else:
                        all_data_file.append(_id+'_'+level_map[key])
                        all_data_level.append(value)
                else:
                    continue
    pd.DataFrame({'chart':all_data_file,'level':all_data_level}).to_csv('./labels.csv')


if __name__ == '__main__':
    all_charts = glob('./charts/*/*/*')
    for chart in tqdm(all_charts):
        _, _id, _, expert, master, remaster = parse_chart_info(chart)
        # ------------------------ expert -----------------------
        if len(expert) == 0 or os.path.exists(f'./charts/parsed_charts/{_id}_expert.pt'):
            continue
        all_data = []
        for line in expert:
            parsed_data = parse_complex_string(line)
            all_data.append(parsed_data)

        indexed_data = encode_data(all_data)
        padded_input = pad_sequence(indexed_data, batch_first=True, padding_value=0)

        torch.save(padded_input, f'./charts/parsed_charts/{_id}_expert.pt')

        # ------------------------ master -----------------------
        if len(master) == 0 or os.path.exists(f'./charts/parsed_charts/{_id}_master.pt'):
            continue
        all_data = []
        for line in master:
            parsed_data = parse_complex_string(line)
            all_data.append(parsed_data)

        indexed_data = encode_data(all_data)
        padded_input = pad_sequence(indexed_data, batch_first=True, padding_value=0)
        torch.save(padded_input, f'./charts/parsed_charts/{_id}_master.pt')

        # ------------------------ remaster -----------------------
        if len(remaster) == 0 or os.path.exists(f'./charts/parsed_charts/{_id}_remaster.pt'):
            continue
        all_data = []
        for line in remaster:
            parsed_data = parse_complex_string(line)
            all_data.append(parsed_data)

        indexed_data = encode_data(all_data)
        padded_input = pad_sequence(indexed_data, batch_first=True, padding_value=0)
        torch.save(padded_input, f'./charts/parsed_charts/{_id}_remaster.pt')

    get_levels(all_charts)
