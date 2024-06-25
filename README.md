# 💥 What's New?
- Added the self-correction support in BaseExperiment.
- Modularized the repository as a package for quick replication. Currently added for SyntheticCopa. 
Other experiment objects for Synthetic SuperGLUE will be shortly added.

# TarGEN: Targeted Data Generation with Large Language Models

This is the official repository of the paper: [TarGEN: Targeted Data Generation with Large Language Models](https://arxiv.org/abs/2310.17876)

# How To?

**-Step 1: Import Packages & Add API_KEYS in the 
`.env` file that should be created in the <ROOT_DIRECTORY>:**

The ability to control model objects has also been introduced in the 
`main.py` script abstracting away from internal classes to provide flexibility.
Any langchain supported API can be used for experimentation.
``` python
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from TarGEN import Generate
from experiments.copa import SyntheticCopa, SyntheticCb

load_dotenv(<ROOT_DIRECTORY>)
API_KEY = os.getenv("OPEN_AI_KEY")
TARGET_DATA_STYLE = "COPA"

# Load Model
openai_llm = ChatOpenAI(openai_api_key=API_KEY)
```

**- Step 2: In the experiments directory add the configs for all the steps:**
> [!IMPORTANT]
> Check out [sample SyntheticCopa class](https://github.com/kevinscaria/TarGEN/blob/main/experiments/copa.py) which extends BaseExperiments 
> class as a systematic way to enforce required methods.

This file requires defining the pydantic object class of the instance sample, 
where each field and description is explicitly mentioned. There are few global
variables that will available during runtime for the generator to access such as 
`DOMAIN, N, SENTENCE` etc. Changes to global variables are currently frozen and requires
changes in the BAseExperiment class. In the next update we will provide accessibility
as a local runtime variable that can be configured as required by the prompt engineer.


**- Step 3: Load experiment object in `main.py`

Once the experiment object has been designed as detailed in step 2, 
it has to be loaded in the runtime.

```python
# Load orchestrator
if TARGET_DATA_STYLE == "COPA":
    experiment_object = SyntheticCopa(model=openai_llm)
else:
    experiment_object = BaseExperiment(llm=openai_llm)
```

**- Step 4: The generator only requires the experiment object.
The 
`create_synthetic_data()` method will orchestrate the generation of samples
based on the [`get_config()`](https://github.com/kevinscaria/TarGEN/blob/main/experiments/copa.py)
method defined in the experiment-specific class:**
```
targen = Generate(experiment_object=experiment_object)
targen.create_synthetic_data(output_path="outputs/copa_sample.json",
                             n_samples=8,
                             overwrite=True,
                             num_instance_seeds=1
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
