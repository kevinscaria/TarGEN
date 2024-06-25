import abc
import numpy as np
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.prompts import (
    SystemMessagePromptTemplate, HumanMessagePromptTemplate,
    ChatPromptTemplate, PromptTemplate
)
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnableSequence


class NotAValidExperiment(Exception):
    pass


class SelfCorrectionSample(BaseModel):
    self_corrected_explanation: str = Field(description="the explanation for thought for self-correction")
    self_corrected_label: str = Field(description="the self corrected label")


class BaseExperiment(object):
    """
    Todo: Write detailed explanation.
    But for now, important point to note is that the GLOBAL variables DOMAIN and SENTENCE
    is available for template use in all subclasses extending from BaseExperiment class.
    """
    def __init__(self, llm):
        self.experiment_name = self.__class__.__name__
        self.self_correction_parser = JsonOutputParser(pydantic_object=SelfCorrectionSample)
        self.system_template = SystemMessagePromptTemplate.from_template(
            "You are a creative instruction following assistant and your name is TarGEN."
        )
        self.step1_chain = llm
        self.step2_chain = llm
        self.step3_chain = llm
        self.step4_chain = RunnableSequence(first=llm, last=JsonOutputParser(pydantic_object=SelfCorrectionSample))

    @abc.abstractmethod
    def get_config(self, ):
        raise NotImplementedError

    def generator_function(
            self,
            step1_prompt_items,
            step2_prompt_items,
            step3_prompt_items,
            step4_prompt_items,
            num_instance_seeds,
    ):

        instance_sample = {}

        # Step 1
        step1_prompt = ChatPromptTemplate.from_messages(
            [
                self.system_template,
                HumanMessagePromptTemplate.from_template(step1_prompt_items["template"])
            ]
        ).format_messages()
        contexts = self.step1_chain.invoke(step1_prompt)

        # Step 2
        instance_sample["DOMAIN"] = np.random.choice(contexts, size=1, replace=True)[0]
        step2_prompt = ChatPromptTemplate.from_messages(
            [
                self.system_template,
                HumanMessagePromptTemplate.from_template(step2_prompt_items["template"],
                                                         input_variables=step2_prompt_items["input_variables"])
            ]
        ).format_messages(DOMAIN=instance_sample["DOMAIN"], N=num_instance_seeds)
        step2_output = self.step2_chain.invoke(step2_prompt)

        # Step 3
        for INSTANCE_SEED in step2_output:
            instance_sample["SENTENCE"] = INSTANCE_SEED
            if isinstance(step3_prompt_items["template"], list):
                relation_idx = np.random.choice(len(step3_prompt_items["template"]), size=1, replace=True)[0]
            else:
                relation_idx = step3_prompt_items["template"]

            step3_prompt = ChatPromptTemplate.from_messages(
                [
                    self.system_template,
                    HumanMessagePromptTemplate(
                        prompt=PromptTemplate(
                            template=step3_prompt_items["template"][relation_idx] + '\n {format_instructions}',
                            input_variables=step3_prompt_items["input_variables"],
                            partial_variables={
                                "format_instructions":
                                    self.step3_chain.last.get_format_instructions() if hasattr(self.step3_chain, "last")
                                    else ""}
                        )
                    )
                ],
            ).format_messages(**instance_sample)
            try:
                instance_sample.update(self.step3_chain.invoke(step3_prompt))
            except:
                return {}

            # Step 4
            instance_correction_prompt = PromptTemplate(
                            template=step4_prompt_items["template"] + '\n {format_instructions}',
                            input_variables=step4_prompt_items["input_variables"],
                            partial_variables={
                                "format_instructions":
                                    self.step4_chain.last.get_format_instructions() if hasattr(self.step4_chain, "last")
                                    else ""}
                        ).format(**instance_sample)
            instance_sample.update(self.step4_chain.invoke(instance_correction_prompt))
            return instance_sample
