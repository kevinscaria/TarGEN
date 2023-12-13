import configparser

from TarGEN import Generate
from experiments.copa import copa_config, custom_copa_parser

config = configparser.ConfigParser()
config.read('./config.ini')
API_KEY = config.get('targen', 'OPEN_AI_KEY')

# Load TarGEN
targen = Generate(api_key=API_KEY)

# Task 1 - COPA
step1_human_prompt = copa_config["step1_prompt"]
step2_human_prompt = copa_config["step2_prompt"]
step3_human_prompt = copa_config["step3_prompt"]
step4_human_prompt = copa_config["step4_prompt"]

targen.create_synthetic_data(step1_human_prompt, step2_human_prompt, step3_human_prompt,
                             step4_human_prompt, n_samples=15, step3_parser=custom_copa_parser,
                             output_path="./outputs/copa_sample.json"
                             )

# Task 2
step1_human_prompt = copa_config["step1_prompt"]
step2_human_prompt = copa_config["step2_prompt"]
step3_human_prompt = copa_config["step3_prompt"]
step4_human_prompt = copa_config["step4_prompt"]

targen.create_synthetic_data(step1_human_prompt, step2_human_prompt, step3_human_prompt,
                             step4_human_prompt, n_samples=15, step3_parser=custom_copa_parser,
                             output_path="./outputs/copa_sample.json"
                             )

# Task 3
step1_human_prompt = copa_config["step1_prompt"]
step2_human_prompt = copa_config["step2_prompt"]
step3_human_prompt = copa_config["step3_prompt"]
step4_human_prompt = copa_config["step4_prompt"]

targen.create_synthetic_data(step1_human_prompt, step2_human_prompt, step3_human_prompt,
                             step4_human_prompt, n_samples=15, step3_parser=custom_copa_parser,
                             output_path="./outputs/copa_sample.json"
                             )

# Task 4
step1_human_prompt = copa_config["step1_prompt"]
step2_human_prompt = copa_config["step2_prompt"]
step3_human_prompt = copa_config["step3_prompt"]
step4_human_prompt = copa_config["step4_prompt"]

targen.create_synthetic_data(step1_human_prompt, step2_human_prompt, step3_human_prompt,
                             step4_human_prompt, n_samples=15, step3_parser=custom_copa_parser,
                             output_path="./outputs/copa_sample.json"
                             )

# Task 5
step1_human_prompt = copa_config["step1_prompt"]
step2_human_prompt = copa_config["step2_prompt"]
step3_human_prompt = copa_config["step3_prompt"]
step4_human_prompt = copa_config["step4_prompt"]

targen.create_synthetic_data(step1_human_prompt, step2_human_prompt, step3_human_prompt,
                             step4_human_prompt, n_samples=15, step3_parser=custom_copa_parser,
                             output_path="./outputs/copa_sample.json"
                             )

# Task 6
step1_human_prompt = copa_config["step1_prompt"]
step2_human_prompt = copa_config["step2_prompt"]
step3_human_prompt = copa_config["step3_prompt"]
step4_human_prompt = copa_config["step4_prompt"]

targen.create_synthetic_data(step1_human_prompt, step2_human_prompt, step3_human_prompt,
                             step4_human_prompt, n_samples=15, step3_parser=custom_copa_parser,
                             output_path="./outputs/copa_sample.json"
                             )

# Task 7
step1_human_prompt = copa_config["step1_prompt"]
step2_human_prompt = copa_config["step2_prompt"]
step3_human_prompt = copa_config["step3_prompt"]
step4_human_prompt = copa_config["step4_prompt"]

targen.create_synthetic_data(step1_human_prompt, step2_human_prompt, step3_human_prompt,
                             step4_human_prompt, n_samples=15, step3_parser=custom_copa_parser,
                             output_path="./outputs/copa_sample.json"
                             )

# Task 8
step1_human_prompt = copa_config["step1_prompt"]
step2_human_prompt = copa_config["step2_prompt"]
step3_human_prompt = copa_config["step3_prompt"]
step4_human_prompt = copa_config["step4_prompt"]

targen.create_synthetic_data(step1_human_prompt, step2_human_prompt, step3_human_prompt,
                             step4_human_prompt, n_samples=15, step3_parser=custom_copa_parser,
                             output_path="./outputs/copa_sample.json"
                             )
