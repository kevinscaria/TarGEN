# 💥 What's New?
- Added the self-correction step logic for COPA.
- Modularized the repository as a package for quick replication. Currently added for SyntheticCopa. For other datasets, shortly added.

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

**-Step 2: Instantiate TarGEN object:**
```
# Load TarGEN
targen = Generate(api_key=API_KEY)
```

**- Step 3: In the experiments directory add the prompts for all the steps:**
> [!IMPORTANT]
> Check out [sample SyntheticCopa class](https://github.com/kevinscaria/TarGEN/blob/main/experiments/copa.py) which extends BaseExperiments class as a systematic way to enforce required methods.

**- Step 4: Load the prompts from the config and use method `create_synthetic_data()` to run the TarGEN pipeline:**
```
# Load TarGEN
targen = Generate(api_key=API_KEY)

target_data_orchestrator = SyntheticCopa()

targen.create_synthetic_data(generator_function=target_data_orchestrator.generator_function,
                             output_path="outputs/copa_sample.json",
                             n_samples=6,
                             step1_human_prompt=target_data_orchestrator.get_config()["step1_prompt"],
                             step2_human_prompt=target_data_orchestrator.get_config()["step2_prompt"],
                             step3_human_prompt=target_data_orchestrator.get_config()["step3_prompt"],
                             step4_human_prompt=target_data_orchestrator.get_config()["step4_prompt"],
                             overwrite=True
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
