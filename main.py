import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from TarGEN import Generate
from experiments.copa import SyntheticCopa
from TarGEN.base_experiment import BaseExperiment

load_dotenv("./.env")
API_KEY = os.getenv("OPEN_AI_KEY")
TARGET_DATA_STYLE = "COPA"

# Load Model
openai_llm = ChatOpenAI(openai_api_key=API_KEY)

# Load orchestrator
if TARGET_DATA_STYLE == "COPA":
    experiment_object = SyntheticCopa(model=openai_llm)
else:
    experiment_object = BaseExperiment(llm=openai_llm)

# Load TarGEN
targen = Generate(experiment_object=experiment_object)
targen.create_synthetic_data(output_path="outputs/copa_sample.json",
                             n_samples=8,
                             overwrite=True,
                             num_instance_seeds=1
                             )
