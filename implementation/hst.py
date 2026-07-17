import json
from typing import List, Dict, Any, Optional

class EntityNode:
    """
    Represents an entity in the problem scenario (e.g., 'Vehicle', 'Warehouse').
    Contains functional attributes used to compute functional compatibility (S_func).
    """
    def __init__(self, name: str, attributes: List[str], description: str = ""):
        self.name = name
        self.attributes = attributes  # e.g., ['Mover', 'Temp', 'Capacity']
        self.description = description

    def __repr__(self):
        return f"EntityNode(name={self.name}, attributes={self.attributes})"

class ConstraintNode:
    """
    Represents a specific mathematical constraint or rule.
    """
    def __init__(self, name: str, description: str, related_entities: List[str], mathematical_formulation: str = ""):
        self.name = name
        self.description = description
        self.related_entities = related_entities
        self.mathematical_formulation = mathematical_formulation

    def __repr__(self):
        return f"ConstraintNode(name={self.name}, entities={self.related_entities})"

class HierarchicalScenarioTree:
    """
    Represents the Hierarchical Scenario Tree (HST) as described in the Opt-Miner paper.
    It contains:
    - R: Global scenario context (Scenario Description and Objective)
    - T_E: Entity subtree (list of EntityNodes)
    - T_C: Constraint subtree (list of ConstraintNodes)
    - Q: Ground-truth retrieval evidence (keywords/techniques)
    """
    def __init__(self, problem_name: str, domain: str, scenario_context: str, objective: str, 
                 entities: List[EntityNode], constraints: List[ConstraintNode], retrieval_evidence: List[str]):
        self.problem_name = problem_name
        self.domain = domain
        self.scenario_context = scenario_context
        self.objective = objective
        self.entities = entities
        self.constraints = constraints
        self.retrieval_evidence = retrieval_evidence

    def get_all_attributes(self) -> List[str]:
        attrs = set()
        for e in self.entities:
            attrs.update(e.attributes)
        return list(attrs)

    def map_entities(self, mapping: Dict[str, str]) -> 'HierarchicalScenarioTree':
        """
        Applies the Structural Decoupling injective mapping Phi: VE_i -> VE_j.
        Transplants constraint logical mappings to a new set of target entities.
        """
        new_constraints = []
        for c in self.constraints:
            new_related = [mapping.get(ent, ent) for ent in c.related_entities]
            new_constraints.append(
                ConstraintNode(c.name, c.description, new_related, c.mathematical_formulation)
            )
        return HierarchicalScenarioTree(
            problem_name=f"{self.problem_name}_mapped",
            domain=self.domain,
            scenario_context=self.scenario_context,
            objective=self.objective,
            entities=self.entities,
            constraints=new_constraints,
            retrieval_evidence=self.retrieval_evidence
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "problem_name": self.problem_name,
            "domain": self.domain,
            "scenario_context": self.scenario_context,
            "objective": self.objective,
            "entities": [{"name": e.name, "attributes": e.attributes, "description": e.description} for e in self.entities],
            "constraints": [{"name": c.name, "description": c.description, "related_entities": c.related_entities, "math": c.mathematical_formulation} for c in self.constraints],
            "retrieval_evidence": self.retrieval_evidence
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'HierarchicalScenarioTree':
        entities = [EntityNode(e['name'], e['attributes'], e.get('description', '')) for e in data['entities']]
        constraints = [ConstraintNode(c['name'], c['description'], c['related_entities'], c.get('math', '')) for c in data['constraints']]
        return cls(
            problem_name=data['problem_name'],
            domain=data.get('domain', 'Unknown'),
            scenario_context=data['scenario_context'],
            objective=data['objective'],
            entities=entities,
            constraints=constraints,
            retrieval_evidence=data.get('retrieval_evidence', [])
        )
