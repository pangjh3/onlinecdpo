import sys

LLMCodeBase = '/apdcephfs_qy3/share_301812049/long-thought/code/LLMCodeBase/'
sys.path.append(LLMCodeBase)

from utils.openmathinst_utils import math_equal
import re


def dirty_rules(answer):
    if "$\\boxed 2$" in answer:
        return "2"
    if "\\boxed 9" in answer:
        return "9"
    if "####" in answer:
        answer = answer.split("####")[1].strip()
        if "The answer is:" in answer:
            answer = answer.split("The answer is:")[1].strip()
        return answer
    if "The answer is:" in answer:
        answer = answer.split("The answer is:")[1].strip()
        return answer
    return "None"
    # print(answer)
    # raise ValueError("No rule for this answer")




# 示例字符串
text = r"This is an example with \boxed{content1} and \boxed{content2}."

def extract_answer(answer):

    # 正则表达式
    pattern = r"\\boxed\{(.*?)\}"

    # 使用 re.findall 匹配内容
    matches = re.findall(pattern, answer)

    if len(matches) > 0:
        
        return matches[-1]
    else:
        return "None"

# print("Matched content:", matches)

# def extract_answer(answer):
#     start_idx = answer.find("\\boxed{")
#     if start_idx != -1:
#         start_idx += len("\\boxed{")
#         unmatched_bracket = 1
#         idx = start_idx
#         while unmatched_bracket > 0:
#             if answer[idx] == '{':
#                 unmatched_bracket += 1
#             elif answer[idx] == '}':
#                 unmatched_bracket -= 1
#             idx += 1
#         gt_answer = answer[start_idx:idx-1].strip()
#     else:
#         gt_answer = dirty_rules(answer)
#     return gt_answer
