import os
import json
import time
from tqdm.auto import tqdm
from typing import Optional
from langchain_openai import ChatOpenAI


class Generate:
    def __init__(self,
                 api_key: str,
                 ):
        self.model = ChatOpenAI(openai_api_key=api_key)

    def create_synthetic_data(self,
                              generator_function,
                              output_path: str = None,
                              n_samples: Optional[int] = 100,
                              step1_human_prompt: Optional[str] = None,
                              step2_human_prompt: Optional[str] = None,
                              step3_human_prompt: Optional[str] = None,
                              step4_human_prompt: Optional[str] = None,
                              overwrite: Optional[bool] = False
                              ):

        # Dump path
        if not os.path.exists(output_path) or overwrite:
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
        start_time = time.time()
        counter = 0

        for idx in pbar:
            pbar.set_description(desc=f"{idx + 1}")
            instance_sample = generator_function(self.model, step1_human_prompt, step2_human_prompt,
                                                 step3_human_prompt, step4_human_prompt)

            # Rate limiter
            counter += 1
            time_delta = time.time() - start_time

            if counter == 15 and time_delta < 60:
                time.sleep(20)

                # Reset the counter and start time after sleeping
                counter = 0
                start_time = time.time()

            # Iteratively dump json
            with open(output_path, mode='w', encoding='utf-8') as append_json:
                samples.append(instance_sample)
                json.dump(samples, append_json, indent=2)

            pbar.update(1)

            if len(samples) == n_samples + 1:
                print('Reached target sample count. Terminating..')
                return samples
