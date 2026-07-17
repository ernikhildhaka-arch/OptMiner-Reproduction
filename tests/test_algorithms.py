import pytest
from typing import Optional
from implementation.hst import HierarchicalScenarioTree, EntityNode, ConstraintNode
from implementation.evolution import HSTEvolutionEngine
from implementation.agent import OptMinerAgent
from src.llm_client import MockClient
from src.prompt_manager import PromptManager

@pytest.fixture
def dummy_trees():
    e1 = EntityNode("DeliveryVan", ["Mover", "Capacity"], "Standard delivery van")
    c1 = ConstraintNode("TimeWindow", "Deliveries must occur within operating hours.", ["DeliveryVan"], "sum(x_i) <= max_hours")
    hst1 = HierarchicalScenarioTree(
        problem_name="Urban_Logistics_V1",
        domain="Logistics",
        scenario_context="A company needs to deliver goods.",
        objective="Minimize total travel time.",
        entities=[e1],
        constraints=[c1],
        retrieval_evidence=["VRPTW"]
    )
    
    e2 = EntityNode("RefrigeratedTruck", ["Mover", "Capacity", "Temp"], "Truck for cold goods")
    c2 = ConstraintNode("TemperatureControl", "Goods must remain below 4C.", ["RefrigeratedTruck"], "temp_t <= 4")
    hst2 = HierarchicalScenarioTree(
        problem_name="ColdChain_Logistics_V1",
        domain="Logistics",
        scenario_context="A distributor must deliver cold goods.",
        objective="Minimize cooling cost.",
        entities=[e2],
        constraints=[c2],
        retrieval_evidence=["ColdChain"]
    )
    
    e3 = EntityNode("AssemblyLine", ["Mover", "Capacity"], "Factory conveyor belt")
    c3 = ConstraintNode("Throughput", "Line must process 50 units/hr.", ["AssemblyLine"], "rate >= 50")
    hst3 = HierarchicalScenarioTree(
        problem_name="Factory_Assembly_V1",
        domain="Manufacturing",
        scenario_context="A factory needs to schedule batches.",
        objective="Maximize daily throughput.",
        entities=[e3],
        constraints=[c3],
        retrieval_evidence=["JobShop"]
    )
    return hst1, hst2, hst3

def test_hst_structural_decoupling(dummy_trees):
    hst1, _, _ = dummy_trees
    # Map 'DeliveryVan' to 'AutonomousDrone'
    mapped_tree = hst1.map_entities({"DeliveryVan": "AutonomousDrone"})
    assert mapped_tree.constraints[0].related_entities == ["AutonomousDrone"]
    assert mapped_tree.problem_name == "Urban_Logistics_V1_mapped"

def test_evolution_s_func(dummy_trees):
    hst1, hst2, _ = dummy_trees
    engine = HSTEvolutionEngine()
    
    # DeliveryVan overlap RefrigeratedTruck: 2/2 = 1.0
    s_func = engine.compute_s_func(hst1, hst2)
    assert s_func == 1.0

def test_evolution_with_mocks(dummy_trees, tmp_path):
    hst1, hst2, hst3 = dummy_trees
    
    # Setup temporary prompts
    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir()
    (prompts_dir / "Prompt_cmp.txt").write_text("Domain Equivalent: yes", encoding="utf-8")
    (prompts_dir / "Prompt_syn.txt").write_text("Merged Description text", encoding="utf-8")
    (prompts_dir / "Prompt_trans.txt").write_text("Adapted Description text", encoding="utf-8")
    (prompts_dir / "Prompt_abs.txt").write_text("Abstracted text", encoding="utf-8")
    
    prompt_manager = PromptManager(prompts_dir=str(prompts_dir))
    llm_client = MockClient(response_text="yes\nMerged Description text\nAdapted Description text\nAbstracted text")
    
    engine = HSTEvolutionEngine(llm_client=llm_client, prompt_manager=prompt_manager)
    
    # Semantic similarity
    assert engine.compute_s_sem(hst1, hst2) == 1
    
    # Union
    union_tree = engine.scenario_union(hst1, hst2)
    assert "UNION" in union_tree.problem_name
    
    # Transfer
    # We mock LLM to say 'no' (0) for semantic domain, and s_func overlap is 1.0
    llm_client_no = MockClient(response_text="no")
    engine_no = HSTEvolutionEngine(llm_client=llm_client_no, prompt_manager=prompt_manager)
    transferred = engine_no.scenario_transfer(hst2, hst3)
    assert "TRANSFERRED" in transferred.problem_name

def test_agent_inference(tmp_path):
    # Setup prompt
    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir()
    (prompts_dir / "Prompt_research.txt").write_text("Prompt research {problem_description} {max_turns}", encoding="utf-8")
    prompt_manager = PromptManager(prompts_dir=str(prompts_dir))
    
    # Mock LLM to return search tags first, then python solver code execution
    class MultiMockClient(MockClient):
        def __init__(self):
            super().__init__()
            self.turn = 0
        def generate(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> str:
            self.turn += 1
            if self.turn == 1:
                return "<search>Vehicle Routing</search>"
            else:
                return "<python>\n# Dummy python code\nprint('Optimal solution found')\n</python>"
                
    client = MultiMockClient()
    agent = OptMinerAgent(llm_client=client, prompt_manager=prompt_manager, max_turns=2)
    
    result = agent.run_inference_loop("Delivery of 10 goods")
    assert "<search>Vehicle Routing</search>" in result
    assert "Optimal solution found" in result
