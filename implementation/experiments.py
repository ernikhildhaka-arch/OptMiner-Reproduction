import json
import os
from implementation.hst import HierarchicalScenarioTree, EntityNode, ConstraintNode
from implementation.evolution import HSTEvolutionEngine
from implementation.agent import OptMinerAgent

def run_synthesis_experiments():
    print("=== Opt-Miner Data Synthesis Pipeline Test ===")
    
    # 1. Create Base Problem 1 (Logistics)
    e1_p1 = EntityNode("DeliveryVan", ["Mover", "Capacity", "Time"], "Standard delivery van")
    c1_p1 = ConstraintNode("TimeWindow", "Deliveries must occur within operating hours.", ["DeliveryVan"], "sum(x_i) <= max_hours")
    hst1 = HierarchicalScenarioTree(
        problem_name="Urban_Logistics_V1",
        domain="Logistics",
        scenario_context="A company needs to deliver goods to 10 customers within urban time windows.",
        objective="Minimize total travel time.",
        entities=[e1_p1],
        constraints=[c1_p1],
        retrieval_evidence=["Vehicle Routing Problem with Time Windows (VRPTW)"]
    )
    
    # 2. Create Base Problem 2 (Logistics - Cold Chain)
    e1_p2 = EntityNode("RefrigeratedTruck", ["Mover", "Capacity", "Temp", "Time"], "Truck for cold goods")
    c1_p2 = ConstraintNode("TemperatureControl", "Goods must remain below 4C.", ["RefrigeratedTruck"], "temp_t <= 4")
    hst2 = HierarchicalScenarioTree(
        problem_name="ColdChain_Logistics_V1",
        domain="Logistics",
        scenario_context="A distributor must deliver perishable goods while maintaining strict temperature controls.",
        objective="Minimize energy cost for cooling while routing.",
        entities=[e1_p2],
        constraints=[c1_p2],
        retrieval_evidence=["Cold Chain Vehicle Routing Problem"]
    )

    # 3. Create Base Problem 3 (Manufacturing - Unrelated Domain)
    e1_p3 = EntityNode("AssemblyLine", ["Mover", "Capacity", "Temp", "Time"], "Factory conveyor belt")
    c1_p3 = ConstraintNode("Throughput", "Line must process 50 units/hr.", ["AssemblyLine"], "rate >= 50")
    hst3 = HierarchicalScenarioTree(
        problem_name="Factory_Assembly_V1",
        domain="Manufacturing",
        scenario_context="A factory needs to schedule batches on an assembly line with thermal processing.",
        objective="Maximize daily throughput.",
        entities=[e1_p3],
        constraints=[c1_p3],
        retrieval_evidence=["Job Shop Scheduling"]
    )
    
    engine = HSTEvolutionEngine()
    
    # Test Union (Same domain: Logistics)
    print("\n[+] Testing Scenario Union (Logistics + Cold Chain Logistics)")
    union_hst = engine.scenario_union(hst1, hst2)
    print(f"Union Problem Name: {union_hst.problem_name}")
    print(f"Entities Count: {len(union_hst.entities)}")
    print(f"Constraints Count: {len(union_hst.constraints)}")
    
    # Test Transfer (Different domain, similar attributes: Cold Chain -> Manufacturing)
    print("\n[+] Testing Scenario Transfer (Cold Chain -> Manufacturing)")
    transfer_hst = engine.scenario_transfer(hst2, hst3)
    print(f"Transfer Problem Name: {transfer_hst.problem_name}")
    print(f"Entities Count: {len(transfer_hst.entities)}")
    
    # Test Fogging
    print("\n[+] Testing Knowledge Fogging on Union Result")
    fogged_hst = engine.knowledge_fogging(union_hst)
    print(f"Fogged Constraint 1: {fogged_hst.constraints[0].description}")
    
    # Save synthetic data
    os.makedirs(r"d:\OptMiner-Reproduction\results", exist_ok=True)
    out_path = r"d:\OptMiner-Reproduction\results\synthetic_problems.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump([union_hst.to_dict(), transfer_hst.to_dict(), fogged_hst.to_dict()], f, indent=4)
    print(f"\nSynthetic data saved to {out_path}")

class MockLLMClient:
    def generate(self, context: str) -> str:
        # Simple mock that issues a search query on the first turn
        if "<search>" not in context:
            return "<|think|> I need to search for VRPTW formulations to find the constraints. </|think|>\n<search>Vehicle Routing Problem with Time Windows</search>"
        else:
            return "<|think|> I found the formulations. I will write Gurobi code now. </|think|>\n<python>\nimport gurobipy as gp\nm = gp.Model()\n# ... mock code ...\n</python>"

def run_agent_test():
    print("\n=== Opt-Miner Agentic Search Loop Test ===")
    mock_llm = MockLLMClient()
    agent = OptMinerAgent(llm_client=mock_llm, max_turns=2)
    
    problem = "A company needs to deliver goods to 10 customers within urban time windows."
    result = agent.run_inference_loop(problem)
    print("Agent interaction log:")
    print(result)

if __name__ == "__main__":
    run_synthesis_experiments()
    run_agent_test()
