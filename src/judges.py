from typing import List, Optional, Union

from trl import BaseBinaryJudge

import evaluate

from .tools import *

class BinaryGTJudge(BaseBinaryJudge):

    # need gold_completions

    def __init__(self, **kwargs):

        super().__init__(**kwargs)
    
    def judge(
        self,
        prompts: List[str],
        completions: List[str],
        gold_completions: Optional[List[str]],
        shuffle_order: bool = True,
    ) -> List[int]:

        results = []
        for idx, (prompt, completion, gt_completion) in enumerate(zip(prompts, completions, gold_completions)):
            # print("completion####****====\n")
            # print(completion)
            # print("gt_completion####****====\n")
            # print(gt_completion)
            pred_ans = extract_answer(completion)
            gt_ans = extract_answer(gt_completion)

            if math_equal(pred_ans, gt_ans):
                results.append(1)
            else:
                results.append(0)

        return results


class BinaryDiffJudge(BaseBinaryJudge):

    # need gold_completions, for wer

    def __init__(self, wer_thred = 0.3, **kwargs):

        super().__init__(**kwargs)
        self.wer = evaluate.load("/apdcephfs_qy3/share_301812049/jianhuipang/evaluate/metrics/wer/wer.py")
        self.wer_thred = wer_thred
    
    def judge(
        self,
        prompts: List[str],
        completions: List[str],
        gold_completions: Optional[List[str]],
        shuffle_order: bool = True,
    ) -> List[int]:

        results = []
        for idx, (prompt, completion, gt_completion) in enumerate(zip(prompts, completions, gold_completions)):
            pred = [completion]
            ref = [gt_completion]
            wer_score = self.wer.compute(predictions=pred, references=ref)

            if wer_score > self.wer_thred:
                results.append(1)
            else:
                results.append(0)

        return results

class BinaryCorrectionJudge(BaseBinaryJudge):

    # need gold_completions, for wer

    def __init__(self, wer_thred = 0.3, **kwargs):

        super().__init__(**kwargs)
        self.prefer_words = ["Wait,"]
    
    def judge(
        self,
        prompts: List[str],
        completions: List[str],
        gold_completions: Optional[List[str]] = None,
        shuffle_order: bool = True,
    ) -> List[int]:

        results = []
        for idx, (prompt, completion) in enumerate(zip(prompts, completions)):
            if any([word in completion for word in self.prefer_words]):
                results.append(1)
            else:
                results.append(0)

        return results


class BinaryDifficultyJudge(BaseBinaryJudge):

    # need gold_completions, for wer

    def __init__(self, **kwargs):

        super().__init__(**kwargs)
        self.prefer_words = ["Wait,"]
    
    def judge(
        self,
        prompts: List[str],
        completions: List[str],
        diffculties: List[float],
        gold_completions: Optional[List[str]] = None,
        shuffle_order: bool = True,
    ) -> List[int]:

        results = []
        for idx, (prompt, completion, diffculty) in enumerate(zip(prompts, completions, diffculties)):
            if diffculty > 2 and any([word in completion for word in self.prefer_words]):
                results.append(1)
            else:
                results.append(0)

        return results
