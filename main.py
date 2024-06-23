import configparser
from TarGEN import Generate
from experiments.copa import SyntheticCopa
from TarGEN.base_experiments import BaseExperiments

config = configparser.ConfigParser()
config.read('./config.ini')
API_KEY = config.get('targen', 'OPEN_AI_KEY')
TARGET_DATA_STYLE = "COPA"

# Load TarGEN
targen = Generate(api_key=API_KEY)

# Load orchestrator
if TARGET_DATA_STYLE == "COPA":
    target_data_orchestrator = SyntheticCopa()
else:
    target_data_orchestrator = BaseExperiments()

targen.create_synthetic_data(generator_function=target_data_orchestrator.generator_function,
                             output_path="outputs/copa_sample.json",
                             n_samples=6,
                             step1_human_prompt=target_data_orchestrator.get_config()["step1_prompt"],
                             step2_human_prompt=target_data_orchestrator.get_config()["step2_prompt"],
                             step3_human_prompt=target_data_orchestrator.get_config()["step3_prompt"],
                             step4_human_prompt=target_data_orchestrator.get_config()["step4_prompt"],
                             overwrite=True
                             )
