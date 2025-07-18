from agents.base_agent import BaseAgent
from langchain.schema import SystemMessage, HumanMessage

class ContextCurator(BaseAgent):
    def __init__(self):
        super(ContextCurator, self).__init__("context_curator")

    # def build_dialogue(self, context_vars, step):
    #     sys_msg = SystemMessage(content=self.format_prompt("system", **context_vars))
    #     user_msg = HumanMessage(content=self.format_prompt("user_structures", **context_vars))
    #     return [sys_msg, user_msg]

    def run(self, context_vars):
        """
        Executes the full dialogue flow:
        system → user_structures → AI → user_biomarkers → AI
        """
        dialogue = []

        # 1. System message
        sys_inputs = self.get_prompt_template("system").input_variables
        sys_msg = SystemMessage(content=self.format_prompt("system", **{
            k: context_vars[k] for k in sys_inputs
        }))
        dialogue.append(sys_msg)

        # 2. User: ask for multicellular structures
        struct_inputs = self.get_prompt_template("user_structures").input_variables
        user_struct = HumanMessage(content=self.format_prompt("user_structures", **{
            k: context_vars[k] for k in struct_inputs
        }))
        dialogue.append(user_struct)

        # 3. AI response to structure query
        res_struct = self.chat(dialogue)
        dialogue.append(res_struct)

        # 4. User: ask about biomarkers
        bio_inputs = self.get_prompt_template("user_biomarkers").input_variables
        user_biomarkers = HumanMessage(content=self.format_prompt("user_biomarkers", **{
            k: context_vars[k] for k in bio_inputs
        }))
        dialogue.append(user_biomarkers)

        # 5. AI response to biomarkers
        res_biomarker = self.chat(dialogue)
        dialogue.append(res_biomarker)

        return dialogue