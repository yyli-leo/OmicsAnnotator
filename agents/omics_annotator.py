from agents.base_agent import BaseAgent
from langchain.schema import SystemMessage, HumanMessage

class OmicsAnnotator(BaseAgent):
    def __init__(self):
        super(OmicsAnnotator, self).__init__("omics_annotator")

    def run(self, context_vars):
        """
        Dialogue:
        - System message (task setup)
        - User message: biological context
        - User message: cell summaries (biomarkers, cell types, segmentation)
        - User message: instruction
        """
        dialogue = []

        # 1. System message
        sys_vars = self.get_prompt_template("system").input_variables
        sys_msg = SystemMessage(content=self.format_prompt("system", **{
            k: context_vars[k] for k in sys_vars
        }))
        dialogue.append(sys_msg)

        # 2. User context message
        context_msg = HumanMessage(content=self.format_prompt("user_context", **{
            "context": context_vars["context"]
        }))
        dialogue.append(context_msg)

        # 3. User data message (summaries)
        summary_vars = self.get_prompt_template("user_input").input_variables
        summary_msg = HumanMessage(content=self.format_prompt("user_input", **{
            k: context_vars[k] for k in summary_vars
        }))
        dialogue.append(summary_msg)

        # 4. User instruction message
        instruction_msg = HumanMessage(content=self.format_prompt("user_instruction"))
        dialogue.append(instruction_msg)

        # 5. LLM call
        res = self.chat(dialogue)
        dialogue.append(res)

        return dialogue