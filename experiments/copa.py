from langchain_core.output_parsers import NumberedListOutputParser, JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.runnables import RunnableSequence

from TarGEN.base_experiment import BaseExperiment


class SyntheticCopaSample(BaseModel):
    choice1: str = Field(description="generated choice 1")
    choice2: str = Field(description="generated choice 2")
    premise: str = Field(description="the proposition that leads to a cause or is an effect of the choices")
    explanation: str = Field(description="explanation of the chosen choice as the label for the premise and question")
    question: str = Field(description="the relational query between the choices and the premise")
    domain: str = Field(description="the domain of the sample")
    initial_label: str = Field(description="the chosen choice as the label. C1 for Choice 1 and C2 for Choice 2")


class SyntheticCopa(BaseExperiment):
    def __init__(self, model):
        super().__init__(model)
        self.step1_chain = RunnableSequence(first=model, last=NumberedListOutputParser())
        self.step2_chain = RunnableSequence(first=model, last=NumberedListOutputParser())
        self.step3_chain = RunnableSequence(first=model, last=JsonOutputParser(pydantic_object=SyntheticCopaSample))

    def get_config(self, ):
        copa_config = {
            "step1_prompt_items":
                {"template": """"Generate domains or settings in which events can take places.
                        It can be fictional, imaginary, ordinary life events or historically accurate events. 
                        RETURN A NUMBERED LIST.""",
                 },

            "step2_prompt_items":
                {"template": """"Generate {N} sentence(s) describing events 
                       that could take place in the domain {DOMAIN}.
                       RETURN A NUMBERED LIST.
                       """,
                 "input_variables": ["DOMAIN", "N"]
                 },

            "step3_prompt_items":
                {"template": ["""For the given sentence, generate
                        choices (Choice 1, Choice 2) , such
                        that one choice is a probable cause of the
                        sentence and the other choice is very unlikely to be the
                        cause of the sentence.
                        Example:
                        Premise: I cast a long shadow.
                        Choice 1: The sun was low in the sky.
                        Choice 2: The grass was tall.
                        Question: cause
                        Explanation: Choice 1, the low position of the
                        sun is more likely to cause a long shadow. The
                        height of the grass has nothing to do with the long
                        shadow, and thus is unlikely to be a cause.
                        Initial Label: C1
                        Example:
                        Premise: The man had knee pain.
                        Choice 1: The man liked eating fruits.
                        Choice 2: The man was old.
                        Question: cause
                        Explanation: Choice 2, the main being old
                        is more likely to cause of a knee pain. The
                        man liking fruits has nothing do with the knee pain,
                        and thus is unlikely to be a cause.
                        Initial Label: C2
                        
                        Premise: {SENTENCE}
                        The domain is: {DOMAIN}
                        """,
                              """
                        For the given sentence, generate 2 choices
                        (Choice 1, Choice 2) , such that
                        one choice is a probable result of the sentence
                        and the other choice is very unlikely to be the result of the
                        sentence.
                        Example:
                        Premise: I fell down the stairs.
                        Choice 1: I injured myself.
                        Choice 2: My mother bought a new car.
                        Question: effect
                        Explanation: Choice 1, the injury is more
                        likely to be a result of the fall. The buying of a car
                        is not implied by the sentence which talks about
                        falling down stairs - hence it is less likely to be the
                        result of the sentence.
                        Initial Label: C1
                        Example:
                        Premise: I went to the beach.
                        Choice 1: I played video games.
                        Choice 2: I went for a swim.
                        Question: effect
                        Explanation: Choice 2, the swimming activity is likely 
                        to be a result of going to the beach. The playing video games
                        is not implied by the sentence which talks about going to the beach,
                        and thus is unlikely to be a result of the sentence.
                        Initial Label: C2
    
                        Premise: {SENTENCE}
                        The domain is: {DOMAIN}
                        Also return the EFFECT of the premise as either "C1" for Choice 1 or "C2" for Choice 2.
                        """],
                 "input_variables": ["SENTENCE", "DOMAIN"]
                 },

            "step4_prompt_items": {
                "template": """"You are given a relation, premise, 2 possible choices (Choice 1 and Choice 2) 
                        and the predicted label as input. 
                        Recheck if the label is correct else correct it.
                        ONLY RETURN THE CODE "C1" FOR Choice 1 or "C2" FOR Choice 2 AS CORRECTED LABELS"
                        f"For the given sample, the relation is {question},
                        Premise: {SENTENCE}, 
                        Choice 1: {choice1}, 
                        Choice 2: {choice2},
                        Existing Explanation: {explanation},
                        Current Label: {initial_label}
                        """,
                "input_variables": ["question", "SENTENCE", "choice1", "choice2", "explanation", "initial_label"],
            }
        }
        return copa_config
