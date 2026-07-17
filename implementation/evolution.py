from typing import List, Tuple, Any, Optional
from implementation.hst import HierarchicalScenarioTree, EntityNode, ConstraintNode
from src.llm_client import BaseLLMClient
from src.prompt_manager import PromptManager
from src.config_loader import ConfigLoader

class HSTEvolutionEngine:
    """
    Implements the tree evolving operators for generating complex problems:
    - Similarity Identification (S_sem and S_func)
    - Scenario Union (Subtree Merging)
    - Scenario Transfer (Entity Adaptation)
    - Knowledge Fogging
    """
    def __init__(self, llm_client: Optional[BaseLLMClient] = None, 
                 prompt_manager: Optional[PromptManager] = None, 
                 logger: Optional[Any] = None):
        self.llm_client = llm_client
        self.prompt_manager = prompt_manager
        self.logger = logger
        
        # Load epsilon parameter from config
        config_loader = ConfigLoader()
        try:
            train_config = config_loader.load_config("training")
            self.epsilon = train_config.get("evolution", {}).get("epsilon", 0.5)
        except Exception:
            self.epsilon = 0.5


    def compute_s_func(self, tree_i: HierarchicalScenarioTree, tree_j: HierarchicalScenarioTree) -> float:
        """
        Computes functional compatibility S_func between two trees based on entity attribute overlap.
        S_func = (1 / |E_i|) * sum_{e_m in E_i} max_{e_n in E_j} ( |A(e_m) intersect A(e_n)| / |A(e_m)| )
        """
        if not tree_i.entities or not tree_j.entities:
            return 0.0
            
        total_max_overlap = 0.0
        for e_m in tree_i.entities:
            max_overlap = 0.0
            attrs_m = set(e_m.attributes)
            if not attrs_m:
                continue
                
            for e_n in tree_j.entities:
                attrs_n = set(e_n.attributes)
                overlap = len(attrs_m.intersection(attrs_n)) / len(attrs_m)
                if overlap > max_overlap:
                    max_overlap = overlap
            total_max_overlap += max_overlap
            
        return total_max_overlap / len(tree_i.entities)

    def compute_s_sem(self, tree_i: HierarchicalScenarioTree, tree_j: HierarchicalScenarioTree) -> int:
        """
        Computes semantic thematic compatibility S_sem (1 if same domain, 0 otherwise).
        Utilizes LLM classifier: S_sem = LLM(Prompt_cmp, R_i, R_j).
        """
        if not self.llm_client or not self.prompt_manager:
            # Fallback to direct domain check if client is absent
            if tree_i.domain != "Unknown" and tree_i.domain == tree_j.domain:
                return 1
            return 0

        prompt = self.prompt_manager.get_prompt(
            "Prompt_cmp", 
            scenario_1=tree_i.scenario_context, 
            scenario_2=tree_j.scenario_context
        )
        if self.logger:
            self.logger.info("Computing S_sem via LLM...")
        
        response = self.llm_client.generate(prompt)
        if self.logger:
            self.logger.info(f"S_sem LLM response: {response.strip()}")

        if "yes" in response.lower():
            return 1
        return 0

    def scenario_union(self, tree_i: HierarchicalScenarioTree, tree_j: HierarchicalScenarioTree) -> HierarchicalScenarioTree:
        """
        Operator I: Scenario Union via Subtree Merging
        Merges problems in the same domain (S_sem = 1).
        """
        s_sem = self.compute_s_sem(tree_i, tree_j)
        if s_sem != 1:
            raise ValueError("Scenario Union requires thematic compatibility (S_sem = 1).")

        # 1. Merge entities
        merged_entities = list(tree_i.entities)
        for e_j in tree_j.entities:
            attrs_j = set(e_j.attributes)
            # Check for functionally equivalent entity in tree_i
            is_duplicate = False
            for e_i in tree_i.entities:
                attrs_i = set(e_i.attributes)
                if attrs_i and attrs_j and len(attrs_i.intersection(attrs_j)) / len(attrs_j) == 1.0:
                    is_duplicate = True
                    break
            if not is_duplicate:
                merged_entities.append(e_j)

        # 2. Aggregate constraints
        merged_constraints = tree_i.constraints + tree_j.constraints

        # 3. Use LLM to weave into a single problem description
        if self.llm_client and self.prompt_manager:
            prompt = self.prompt_manager.get_prompt(
                "Prompt_syn",
                scenario_1=tree_i.scenario_context,
                scenario_2=tree_j.scenario_context,
                merged_entities=str(merged_entities),
                merged_constraints=str(merged_constraints)
            )
            if self.logger:
                self.logger.info("Weaving scenario union via LLM...")
            merged_scenario_context = self.llm_client.generate(prompt)
        else:
            merged_scenario_context = f"{tree_i.scenario_context}\nAdditionally: {tree_j.scenario_context}"

        merged_evidence = list(set(tree_i.retrieval_evidence + tree_j.retrieval_evidence))

        return HierarchicalScenarioTree(
            problem_name=f"{tree_i.problem_name}_UNION_{tree_j.problem_name}",
            domain=tree_i.domain,
            scenario_context=merged_scenario_context,
            objective=f"Multi-objective: {tree_i.objective} and {tree_j.objective}",
            entities=merged_entities,
            constraints=merged_constraints,
            retrieval_evidence=merged_evidence
        )

    def scenario_transfer(self, source_tree: HierarchicalScenarioTree, target_tree: HierarchicalScenarioTree) -> HierarchicalScenarioTree:
        """
        Operator II: Scenario Transfer and Entity Adaptation
        Applies mathematical structure from source_tree to target_tree (S_sem = 0, S_func >= epsilon).
        """
        s_sem = self.compute_s_sem(source_tree, target_tree)
        s_func = self.compute_s_func(source_tree, target_tree)
        
        if s_sem == 1 or s_func < self.epsilon:
            raise ValueError(f"Scenario Transfer requires S_sem = 0 and S_func >= {self.epsilon}.")

        # Adapt source entities to fit target scenario mathematically
        adapted_entities = target_tree.entities + [e for e in source_tree.entities]
        transferred_constraints = target_tree.constraints + source_tree.constraints

        # Weave logic using LLM
        if self.llm_client and self.prompt_manager:
            prompt = self.prompt_manager.get_prompt(
                "Prompt_trans",
                source_scenario=source_tree.scenario_context,
                target_scenario=target_tree.scenario_context,
                source_entities=str(source_tree.entities),
                source_constraints=str(source_tree.constraints)
            )
            if self.logger:
                self.logger.info("Transferring scenario logic via LLM...")
            transferred_context = self.llm_client.generate(prompt)
        else:
            transferred_context = f"Target Base: {target_tree.scenario_context}\nTransferred Logic: {source_tree.scenario_context}"
        
        merged_evidence = list(set(target_tree.retrieval_evidence + source_tree.retrieval_evidence))

        return HierarchicalScenarioTree(
            problem_name=f"{source_tree.problem_name}_TRANSFERRED_TO_{target_tree.problem_name}",
            domain=target_tree.domain,
            scenario_context=transferred_context,
            objective=target_tree.objective,
            entities=adapted_entities,
            constraints=transferred_constraints,
            retrieval_evidence=merged_evidence
        )

    def knowledge_fogging(self, tree: HierarchicalScenarioTree) -> HierarchicalScenarioTree:
        """
        Operator III: Knowledge Fogging
        Masks explicit domain common knowledge, replacing them with concise terminology to compel information seeking.
        """
        fogged_constraints = []
        if self.llm_client and self.prompt_manager:
            if self.logger:
                self.logger.info("Fogging scenario details and constraints via LLM...")
            # Fog global context
            fogged_scenario_context = self.llm_client.generate(
                self.prompt_manager.get_prompt("Prompt_abs", constraints=tree.scenario_context)
            )
            # Fog each constraint
            for c in tree.constraints:
                c_prompt = self.prompt_manager.get_prompt("Prompt_abs", constraints=c.description)
                fogged_desc = self.llm_client.generate(c_prompt)
                fogged_constraints.append(ConstraintNode(c.name, fogged_desc, c.related_entities, c.mathematical_formulation))
        else:
            fogged_scenario_context = f"[FOGGED] {tree.scenario_context}"
            for c in tree.constraints:
                fogged_constraints.append(ConstraintNode(c.name, f"[FOGGED] {c.description}", c.related_entities, c.mathematical_formulation))
            
        return HierarchicalScenarioTree(
            problem_name=f"{tree.problem_name}_FOGGED",
            domain=tree.domain,
            scenario_context=fogged_scenario_context,
            objective=tree.objective,
            entities=tree.entities,
            constraints=fogged_constraints,
            retrieval_evidence=tree.retrieval_evidence
        )
