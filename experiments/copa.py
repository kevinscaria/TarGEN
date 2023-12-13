copa_config = {
    "step1_prompt": """Generate domains or settings in which events can take places. 
                Do not return a numbered list. It should be comma separated.""",

    "step2_prompt": """Generate {N} sentences describing events 
               that could take place in the domain {DOMAIN}
               """,

    "step3_prompt": """query = CAUSE. Add "What was the
                CAUSE of this?" to the premise during post
                processing For the given sentence, generate
                hypotheses (Hypothesis 1, Hypothesis 2) , such
                that Hypothesis 1 is a probable cause of the
                sentence. Hypothesis 2 is very unlikely to be the
                cause of the sentence.
                Example:
                Sentence: I cast a long shadow.
                Hypothesis 1: The sun was low in the sky.
                Hypothesis 2: The grass was tall.
                Explanation: Hypothesis 1, the low position of the
                sun is more likely to cause a long shadow. The
                height of the grass has nothing to do with the long
                shadow, and thus is unlikely to be a cause.
                Sentence: {SENTENCE}
                query = RESULT. Add "What was the RESULT of
                this?" to the premise during post-processing
                For the given sentence, generate 2 hypotheses
                (Hypothesis 1, Hypothesis 2) , such that
                Hypothesis 1 is a probable result of the sentence.
                Hypothesis 2 is very unlikely to be the result of the
                sentence.
                Example:
                Sentence: I fell down the stairs.
                Hypothesis 1: I injured myself.
                Hypothesis 2: My mother bought a new car.
                Explanation: Hypothesis 1, the injury is more
                likely to be a result of the fall. The buying of a car
                is not implied by the sentence which talks about
                falling down stairs - hence it is less likely to be the
                result of the sentence.
                Sentence: {SENTENCE}
                """,

    "step4_prompt": "Kevin"
}


def custom_copa_parser(inference_output):
    instance_sample = {}
    for response in inference_output.content.split("\n"):
        if "Explanation" in response:
            instance_sample['explanation'] = response.split("Explanation: ")[-1]
        elif "Hypothesis 1" in response:
            instance_sample['hypothesis_1'] = response.split("Hypothesis 1: ")[-1]
        elif "Hypothesis 2" in response:
            instance_sample['hypothesis_2'] = response.split("Hypothesis 2: ")[-1]
    return instance_sample
