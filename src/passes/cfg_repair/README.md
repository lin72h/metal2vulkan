# Retained-SPIR-V CFG repair

This subsystem owns the structured-control-flow repairs that remain load-bearing after native
emission and typed stage/access lowering:

- merge declarations displaced from their terminators;
- loop continues that pass through another branch target;
- selection merges that collide with loop-continue targets;
- loop-continue external predecessors and the coupled phi-edge reconciliation.

A complete primary no-retry census records actual mutations for merge placement, continue
pass-through, continue-selection splitting, and phi-edge reconciliation. The external-predecessor
member records no primary mutation but remains coupled to the phi fixpoint until retry-tier
reachability is measured.
