from src.modules.research.claim_graph import ClaimGraph, ClaimKind, build_claim
from src.modules.research.claim_resolution import (
    ConservativeClaimResolver,
    SemanticClaimRelation,
)
from src.modules.research.entity_identity import (
    EntityIdentityLayer,
    normalize_entity_name,
)


def test_legal_suffix_normalization_preserves_same_company():
    assert normalize_entity_name("Apple Inc.") == "apple"
    assert normalize_entity_name("Apple") == "apple"


def test_explicit_alias_map_can_join_ticker_and_company_name():
    layer = EntityIdentityLayer(alias_map={"AAPL": "Apple"})
    left = layer.resolve(
        claim_key="aapl.revenue.2025",
        statement="AAPL revenue was 100 billion dollars in 2025.",
    )
    right = layer.resolve(
        claim_key="apple.revenue.2025",
        statement="Apple revenue was 100 billion dollars in 2025.",
    )
    comparison = layer.compare(left, right)
    assert comparison.same_entity is True


def test_different_company_scopes_are_hard_semantic_boundary():
    graph = ClaimGraph()
    graph.register_claim(
        build_claim(
            claim_key="alpha.adjusted_ebitda_margin.2025",
            statement="Alpha adjusted EBITDA margin was 31 percent in 2025.",
            kind=ClaimKind.FACT,
        )
    )
    result = ConservativeClaimResolver().resolve(
        statement="Beta adjusted EBITDA margin was 31 percent in 2025.",
        claim_key="beta.adjusted_ebitda_margin.2025",
        graph=graph,
    )
    assert result.relation is SemanticClaimRelation.DISTINCT
    assert result.reason == "different_entity_identity"
    assert result.signals["entity_match"] is False


def test_different_entities_do_not_become_numeric_contradiction():
    graph = ClaimGraph()
    graph.register_claim(
        build_claim(
            claim_key="alpha.revenue.2025",
            statement="Alpha revenue was 40 billion dollars in 2025.",
            kind=ClaimKind.FACT,
        )
    )
    result = ConservativeClaimResolver().resolve(
        statement="Beta revenue was 39 billion dollars in 2025.",
        claim_key="beta.revenue.2025",
        graph=graph,
    )
    assert result.relation is SemanticClaimRelation.DISTINCT
    assert result.reason == "different_entity_identity"


def test_grounded_legal_name_metadata_matches_normalized_company_identity():
    graph = ClaimGraph()
    graph.register_claim(
        build_claim(
            claim_key="apple.revenue.2025",
            statement="Apple Inc. revenue was 100 billion dollars in 2025.",
            kind=ClaimKind.FACT,
            metadata={
                "entity_name": "Apple Inc.",
                "entity_type": "company",
                "entity_identity_source": "grounded_metadata",
            },
        )
    )
    result = ConservativeClaimResolver().resolve(
        statement="Apple revenue was 100 billion dollars in 2025.",
        claim_key="apple.revenue.2025",
        graph=graph,
        metadata={
            "entity_name": "Apple",
            "entity_type": "company",
        },
        quote="Apple revenue was 100 billion dollars in 2025.",
    )
    assert result.relation in {
        SemanticClaimRelation.EXACT,
        SemanticClaimRelation.PARAPHRASE,
    }
    assert result.signals["entity_match"] is True


def test_generic_research_namespace_does_not_invent_entity():
    layer = EntityIdentityLayer()
    identity = layer.resolve(
        claim_key="macro.inflation.2026-08",
        statement="Inflation was 3.2 percent in August 2026.",
    )
    assert identity is None
