# agents/omics_annotator_aggregate.py
from pathlib import Path
from agents.base_agent import BaseAgent
from langchain.schema import SystemMessage, HumanMessage


class OmicsAnnotatorAggregate(BaseAgent):
    def __init__(self):
        BaseAgent.__init__(self, "omics_annotator_aggregate")

    def run(self, context_vars):
        dialogue = []

        sys_vars = self.get_prompt_template("system").input_variables
        sys_msg = SystemMessage(content=self.format_prompt("system", **{
            k: context_vars[k] for k in sys_vars
        }))
        dialogue.append(sys_msg)

        context_msg = HumanMessage(content=self.format_prompt("context", **{
            "context": context_vars["context"]
        }))
        dialogue.append(context_msg)

        input_vars = self.get_prompt_template("input").input_variables
        input_msg = HumanMessage(content=self.format_prompt("input", **{
            k: context_vars[k] for k in input_vars
        }))
        dialogue.append(input_msg)

        instruction_msg = HumanMessage(content=self.format_prompt("instruction"))
        dialogue.append(instruction_msg)

        res = self.chat(dialogue)
        dialogue.append(res)

        return dialogue

    # def aggregate_dict(self, outputs_by_cl, tissue_types, major_phenotypes, context):
    #     aggregated = {}
    #     tissue_types_str = ", ".join(tissue_types).lower()
    #     for (region_id, cl), outputs in outputs_by_cl.items():
    #         context_vars = {
    #             "tissue_types": tissue_types_str,
    #             "major_phenotypes": major_phenotypes,
    #             "context": context,
    #             "n_obs": len(outputs),
    #             "results": "\n===\n".join(outputs)
    #         }

        #     dialogue = self.run(context_vars)
        #     aggregated[(region_id, cl)] = dialogue[-1].content
        # return aggregated