# agents/pathology_scorer.py
from agents.base_agent import BaseAgent
from langchain.schema import SystemMessage, HumanMessage, AIMessage


phenotype_desc = """In Diabetic Kidney Disease, microscopic changes in kidney tissue reflect progressive damage caused by chronic hyperglycemia and subsequent metabolic and hemodynamic abnormalities. These changes are observed under light microscopy, electron microscopy, and immunohistochemistry. Below are the key microscopic features:

1. Glomerular Changes:
Mesangial Expansion: Enlargement of the mesangial area due to the accumulation of extracellular matrix (ECM), such as collagen and fibronectin.
Glomerular Basement Membrane (GBM) Thickening: Widening of the GBM due to increased deposition of glycoproteins like type IV collagen.
Podocyte Injury: Loss or detachment of podocytes (specialized cells lining the GBM), leading to impaired filtration barrier and proteinuria.
Glomerulosclerosis: Formation of nodular glomerulosclerosis (Kimmelstiel-Wilson nodules), a hallmark of advanced DKD, as well as diffuse glomerulosclerosis.

2. Tubulointerstitial Changes:
Tubular Atrophy: Loss of tubular epithelial cells, leading to thinning and dysfunction of the tubules.
Interstitial Fibrosis: Excessive deposition of ECM in the interstitial space, contributing to nephron loss and kidney function decline.
Inflammatory Cell Infiltration: Presence of monocytes, macrophages, or lymphocytes in the interstitial area.

3. Vascular Changes:
Arteriolar Hyalinosis: Hyaline deposits in the walls of afferent and efferent arterioles, leading to narrowing and impaired blood flow.
Capillary Rarefaction: Loss of peritubular capillaries, worsening ischemia and fibrosis.

4. Protein Deposits:
Immunofluorescence Findings: Deposition of IgG and albumin in the GBM or mesangium due to altered permeability.
Non-enzymatic Glycation Products: Accumulation of advanced glycation end-products (AGEs) contributes to structural and functional damage.

These microscopic changes collectively lead to impaired filtration, proteinuria, and progressive kidney function decline. Early detection and intervention are critical to managing DKD and preventing progression to end-stage renal disease (ESRD)."""

class PathologyScorer(BaseAgent):
    def __init__(self):
        BaseAgent.__init__(self, "pathology_scorer")

    def run(self, context_vars):
        dialogue = []

        sys_vars = self.get_prompt_template("system").input_variables
        sys_msg = SystemMessage(content=self.format_prompt("system", **{
            k: context_vars[k] for k in sys_vars
        }))
        dialogue.append(sys_msg)

        user_input_msg = HumanMessage(content=context_vars["user_input"])
        dialogue.append(user_input_msg)

        ai_msg = AIMessage(content=context_vars["ai_output"])
        dialogue.append(ai_msg)

        user_final_input = HumanMessage(content=self.format_prompt("user_final_input", **{
            "diagnosis": context_vars["diagnosis"],
            # "phenotype_desc": context_vars["phenotype_desc"]
            "phenotype_desc": phenotype_desc,
        }) + self.format_prompt("user_instruction"))
        dialogue.append(user_final_input)

        res = self.chat(dialogue)
        dialogue.append(res)

        return dialogue