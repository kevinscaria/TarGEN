# 💥 Additions
- Adding generation.py script (WIP)
- Adding analysis submodule to generate all analysis points related to bias, dataset difficulty, diversity, correctness etc. along lines of previous research (WIP)
- Modularizing the repository as a package for quick replication (WIP)


# TarGEN: Targeted Data Generation with Large Language Models

This is the official repository of the paper: [TarGEN: Targeted Data Generation with Large Language Models](https://arxiv.org/abs/2310.17876)

# How To?

**-Step 1: Import Packages & Add API_KEYS in the config.ini file in the root directory:**
``` python
import configparser

from TarGEN import Generate
from experiments.copa import copa_config, custom_copa_parser

config = configparser.ConfigParser()
config.read('./config.ini')
API_KEY = config.get('targen', 'OPEN_AI_KEY')
```

**- Step 2: Instantiate TarGEN object*:*
```
# Load TarGEN
targen = Generate(api_key=API_KEY)
```

**- Step 3: In the experiments directory add the prompts for all the steps:**
> [!IMPORTANT]  
> Support for self-correction will be added shortly in this package.
```
copa_config = {
    "step1_prompt": """ADD CUSTOM STAGE 1 PROMPT""",

    "step2_prompt": """ADD CUSTOM STAGE 2 PROMPT""",

    "step3_prompt": """ADD CUSTOM STAGE 3 PROMPT""",

    "step4_prompt": """ADD CUSTOM STAGE 4 PROMPT"""
}

def custom_copa_parser(inference_output):
    """Write output parser logic"""
```

**- Step 4: Load the prompts from the config and use method `create_synthetic_data` ti run the TarGEN pipeline:**
```
step1_human_prompt = copa_config["step1_prompt"]
step2_human_prompt = copa_config["step2_prompt"]
step3_human_prompt = copa_config["step3_prompt"]
step4_human_prompt = copa_config["step4_prompt"]

targen.create_synthetic_data(step1_human_prompt, step2_human_prompt, step3_human_prompt,
                             step4_human_prompt, n_samples=15, step3_parser=custom_copa_parser,
                             output_path="./outputs/copa_sample.json"
                             )
```

### If you find our work useful, please cite the paper: 

```bibtex
@article{gupta2023targen,
  title={TarGEN: Targeted Data Generation with Large Language Models},
  author={Gupta, Himanshu and Scaria, Kevin and Anantheswaran, Ujjwala and Verma, Shreyas and Parmar, Mihir and Sawant, Saurabh Arjun and Mishra, Swaroop and Baral, Chitta},
  journal={arXiv preprint arXiv:2310.17876},
  year={2023}
}
```
