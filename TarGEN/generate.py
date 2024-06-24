import os
import json
import time
import uuid

from tqdm.auto import tqdm
from typing import Optional

from TarGEN.base_experiment import BaseExperiment, NotAValidExperiment


class Generate:
    def __init__(self, experiment_object):
        if not isinstance(experiment_object, BaseExperiment):
            raise NotAValidExperiment(
                f"The experiment object {experiment_object.__name__} does not meet required "
                f"definition of BaseExperiments"
            )
        else:
            self.experiment_object = experiment_object

    def create_synthetic_data(self,
                              output_path: str = None,
                              n_samples: Optional[int] = 100,
                              overwrite: Optional[bool] = False,
                              num_instance_seeds: Optional[int] = 1,
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
            instance_sample = self.experiment_object.generator_function(
                self.experiment_object.get_config()["step1_prompt_items"],
                self.experiment_object.get_config()["step2_prompt_items"],
                self.experiment_object.get_config()["step3_prompt_items"],
                self.experiment_object.get_config()["step4_prompt_items"],
                num_instance_seeds
            )

            # Adding identifiers
            instance_sample["s_no"] = start_idx+idx
            instance_sample["id"] = str(uuid.uuid4())

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
