import numpy as np
from langchain_core.messages import SystemMessage, HumanMessage
from langchain.prompts import HumanMessagePromptTemplate
from langchain.output_parsers import CommaSeparatedListOutputParser, NumberedListOutputParser

from TarGEN.base_experiments import BaseExperiments


class SyntheticCopa(BaseExperiments):
    def __init__(self, ):
        super().__init__()

    def get_config(self, ):
        copa_config = {
            "step1_prompt": """Generate domains or settings in which events can take places. 
                        Do not return a numbered list. It should be comma separated.""",

            "step2_prompt": """Generate {N} sentence(s) describing events 
                       that could take place in the domain {DOMAIN}
                       """,

            "step3_prompt": ["""For the given sentence, generate
                        hypotheses (Hypothesis 1, Hypothesis 2) , such
                        that one hypothesis is a probable cause of the
                        sentence and the other hypothesis is very unlikely to be the
                        cause of the sentence.
                        Example:
                        Premise: I cast a long shadow.
                        Hypothesis 1: The sun was low in the sky.
                        Hypothesis 2: The grass was tall.
                        Explanation: Hypothesis 1, the low position of the
                        sun is more likely to cause a long shadow. The
                        height of the grass has nothing to do with the long
                        shadow, and thus is unlikely to be a cause.
                        Example:
                        Premise: The man had knee pain.
                        Hypothesis 1: The man liked eating fruits.
                        Hypothesis 2: The man was old.
                        Explanation: Hypothesis 2, the main being old
                        is more likely to cause of a knee pain. The
                        man liking fruits has nothing do with the knee pain,
                        and thus is unlikely to be a cause.
                        Sentence: {SENTENCE}""",
                             """
                        For the given sentence, generate 2 hypotheses
                        (Hypothesis 1, Hypothesis 2) , such that
                        one hypothesis is a probable result of the sentence
                        and the other hypothesis is very unlikely to be the result of the
                        sentence.
                        Example:
                        Premise: I fell down the stairs.
                        Hypothesis 1: I injured myself.
                        Hypothesis 2: My mother bought a new car.
                        Explanation: Hypothesis 1, the injury is more
                        likely to be a result of the fall. The buying of a car
                        is not implied by the sentence which talks about
                        falling down stairs - hence it is less likely to be the
                        result of the sentence.
                        Example:
                        Premise: I went to the beach.
                        Hypothesis 1: I played video games.
                        Hypothesis 2: I went for a swim.
                        Explanation: Hypothesis 2, the swimming activity is likely 
                        to be a result of going to the beach. The playing video games
                        is not implied by the sentence which talks about going to the beach,
                        and thus is unlikely to be a result of the sentence.
                        Sentence: {SENTENCE}
                        """],

            "step4_prompt": """You are given a relation, premise, 2 possible hypotheses (Choice 1 and Choice 2) 
                        and the predicted label as input. 
                        Recheck if the label is correct else correct it.
                        ONLY RETURN THE CODE "C1" FOR Choice 1 or "C2" FOR Choice 2 AS CORRECTED LABELS"
                        """
        }
        return copa_config

    @staticmethod
    def custom_parser(inference_output):
        _instance_sample = {}
        for response in inference_output.content.split("\n"):
            if "Explanation" in response:
                _instance_sample['explanation'] = response.split("Explanation: ")[-1]
            elif "Hypothesis 1" in response:
                _instance_sample['choice1'] = response.split("Hypothesis 1: ")[-1]
            elif "Hypothesis 2" in response:
                _instance_sample['choice2'] = response.split("Hypothesis 2: ")[-1]
        if all([expected_keys in list(_instance_sample.keys()) for \
                expected_keys in ["explanation", "choice1", "choice2"]]):
            return True, _instance_sample
        else:
            return False, {}

    def generator_function(
            self,
            model,
            step1_prompt=None,
            step2_prompt=None,
            step3_prompt=None,
            step4_prompt=None
    ):
        messages = [
            SystemMessage(content="You are a creative instruction following assistant and your name is TarGEN.")
        ]

        # Step 1
        step1_prompt = HumanMessage(content=step1_prompt)
        contexts_prompt = messages + [step1_prompt]
        step1_output = model.invoke(input=contexts_prompt)
        contexts = CommaSeparatedListOutputParser().parse(step1_output.content)
        if "1" in contexts[0]:
            contexts = NumberedListOutputParser().parse(step1_output.content)

        # Step 2
        N = 1
        DOMAIN = np.random.choice(contexts, size=1, replace=True)[0]
        instance_seeds_prompt = HumanMessagePromptTemplate.from_template(template=step2_prompt,
                                                                         input_variables=["DOMAIN", "N"])
        instance_seeds_prompt = instance_seeds_prompt.format(DOMAIN=DOMAIN, N=N)
        instance_seeds_prompt = messages + [instance_seeds_prompt]
        step2_output = model.invoke(input=instance_seeds_prompt)
        if N > 1:
            instance_seeds = NumberedListOutputParser().parse(step2_output.content)
        else:
            instance_seeds = [step2_output.content]

        # Step 3
        INSTANCE_SEED = np.random.choice(instance_seeds, size=1, replace=True)[0]
        relation_idx = np.random.choice(len(step3_prompt), size=1, replace=True)[0]
        RELATION = "CAUSE" if relation_idx == 0 else "EFFECT"
        step3_prompt = step3_prompt[relation_idx]
        step3_prompt = HumanMessagePromptTemplate.from_template(template=step3_prompt,
                                                                input_variables=["SENTENCE"])
        instance_sample_prompt = step3_prompt.format(SENTENCE=INSTANCE_SEED)
        instance_sample_prompt = messages + [instance_sample_prompt]

        # Fail gracefully if model response does not follow the required structure after 5 attempts
        is_parsed = False
        ignore_sample = False
        num_attempts = 1
        instance_sample = {}
        while not is_parsed:
            if num_attempts > 6:
                print("[FAILING GRACEFULLY] Tried 5 times to get structured output. "
                      "Ignoring this sample and moving to next one.")
                ignore_sample = True
            step3_output = model.invoke(input=instance_sample_prompt)
            is_parsed, instance_sample = self.custom_parser(step3_output)
            num_attempts += 1

        if not ignore_sample:
            instance_sample["premise"] = INSTANCE_SEED
            instance_sample["question"] = RELATION
            instance_sample["domain"] = DOMAIN

            label_extraction_prompt = """
                If the explanation suggests that Hypothesis 1 follows the RELATION and PREMISE then
                PRINT "C1" else PRINT "C2".
                """
            label_before_self_correction = model.invoke(input=instance_sample["explanation"] + label_extraction_prompt)
            instance_sample["initial_label"] = label_before_self_correction.content

            # Step 4
            instance_correction_prompt = step4_prompt + \
                                         f"For the given sample, the relation is {RELATION}" + \
                                         f"Premise: {INSTANCE_SEED}" + \
                                         f"Choice 1: {instance_sample['choice1']}" + \
                                         f"Choice 1: {instance_sample['choice2']}"
            self_correction_response = model.invoke(input=instance_correction_prompt)
            instance_sample["self_corrected_label"] = self_correction_response.content
            return instance_sample
        else:
            return instance_sample
