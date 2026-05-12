# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Shared preprocessing helpers for math competition evaluation datasets."""

import argparse
import os
from collections.abc import Mapping

import datasets

from verl.utils.hdfs_io import copy, makedirs

INSTRUCTION_FOLLOWING = "\nPlease reason step by step, and put your final answer within \\boxed{{}}."


def _as_answer_string(answer) -> str:
    if isinstance(answer, list):
        if len(answer) == 0:
            return ""
        return str(answer[0])
    return str(answer)


def run_math_competition_preprocess(
    *,
    hf_dataset: str,
    data_source: str,
    split: str,
    default_local_dir: str,
    config_name: str = "default",
) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default=default_local_dir)
    parser.add_argument("--hdfs_dir", default=None)

    args = parser.parse_args()

    print(f"Loading the {hf_dataset} dataset from huggingface...", flush=True)
    dataset = datasets.load_dataset(hf_dataset, config_name)

    source_dataset = dataset[split]
    train_dataset = source_dataset
    test_dataset = source_dataset

    def make_map_fn(output_split):
        def process_fn(example: Mapping, idx: int):
            question = str(example["problem"]) + INSTRUCTION_FOLLOWING
            answer = _as_answer_string(example["answer"])

            if idx == 0:
                print("Question: ", question)
                print("Answer: ", type(answer), " ", answer)

            extra_info = {"split": output_split, "index": idx}
            if "problem_idx" in example:
                extra_info["problem_idx"] = int(example["problem_idx"])
            if "id" in example:
                extra_info["source_id"] = str(example["id"])
            if "problem_type" in example:
                extra_info["problem_type"] = example["problem_type"]

            return {
                "data_source": data_source,
                "id": idx,
                "prompt": [
                    {
                        "role": "user",
                        "content": question,
                    },
                ],
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": answer},
                "extra_info": extra_info,
            }

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True)

    local_dir = os.path.expanduser(args.local_dir)
    os.makedirs(local_dir, exist_ok=True)
    hdfs_dir = args.hdfs_dir

    train_dataset.to_parquet(os.path.join(local_dir, "train.parquet"))
    test_dataset.to_parquet(os.path.join(local_dir, "test.parquet"))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_dir, dst=hdfs_dir)
