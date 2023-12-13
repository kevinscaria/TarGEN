import os
import json
import time
import numpy as np
from tqdm.auto import tqdm

# Type Hints
from typing import Optional

from langchain.chat_models import ChatOpenAI
from ._prompt_builder import PromptBuilder


class Generate:
    def __init__(self,
                 api_key: str,
                 n: Optional[int] = None
                 ):
        self.prompt_builder = PromptBuilder()
        self.model = ChatOpenAI(openai_api_key=api_key)
        self.n = n or 2

    def create_synthetic_data(self,
                              step1_human_prompt: str,
                              step2_human_prompt: str,
                              step3_human_prompt: str,
                              step4_human_prompt: str,
                              output_path: str = None,
                              n_samples: Optional[int] = 100,
                              step3_parser: Optional = None
                              ):

        # Dump path
        if not os.path.exists(output_path):
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            samples = []
            start_idx = 0
            with open(output_path, mode='w', encoding='utf-8') as f:
                json.dump(samples, f)
        else:
            with open(output_path) as user_file:
                file_contents = user_file.read()
            samples = json.loads(file_contents)
            start_idx = len(samples)

        pbar = tqdm(range(start_idx, n_samples))
        counter = 0
        start_time = time.time()

        # Build the prompt
        contexts_prompt = self.prompt_builder.get_contexts(step1_prompt=step1_human_prompt)
        step1_output = self.model.invoke(input=contexts_prompt)
        contexts = self.prompt_builder.csv_output_parser.parse(step1_output.content)
        if "1" in contexts[0]:
            contexts = self.prompt_builder.numbered_list_parser.parse(step1_output.content)

        for idx in pbar:
            pbar.set_description(desc=f"{idx + 1}")
            DOMAIN = np.random.choice(contexts, size=1, replace=True)[0]
            instance_seeds_prompt = self.prompt_builder.get_instance_seed(step2_prompt=step2_human_prompt,
                                                                          domain=DOMAIN,
                                                                          n=self.n
                                                                          )

            step2_output = self.model.invoke(input=instance_seeds_prompt)
            instance_seeds = self.prompt_builder.numbered_list_parser.parse(step2_output.content)

            # Rate limiter
            counter += 1
            time_delta = time.time() - start_time

            if counter == 15 and time_delta < 60:
                time.sleep(20)

                # Reset the counter and start time after sleeping
                counter = 0
                start_time = time.time()

            INSTANCE_SEED = np.random.choice(instance_seeds, size=1, replace=True)[0]
            samples_prompt = self.prompt_builder.get_samples(step3_prompt=step3_human_prompt,
                                                             sentence=INSTANCE_SEED
                                                             )
            step3_output = self.model.invoke(input=samples_prompt)
            instance_sample = step3_parser(step3_output)
            instance_sample["premise"] = INSTANCE_SEED
            instance_sample["domain"] = DOMAIN

            # Iteratively dump json
            with open(output_path, mode='w', encoding='utf-8') as append_json:
                samples.append(instance_sample)
                json.dump(samples, append_json, indent=2)

            pbar.update(1)

            if len(samples) == n_samples+1:
                print('Reached target sample count. Terminating..')
                return samples
