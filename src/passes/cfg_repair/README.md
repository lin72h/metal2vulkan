# Retained-SPIR-V CFG repair

This subsystem owns the structured-control-flow repairs that remain load-bearing after native
emission and typed stage/access lowering:

- merge declarations displaced from their terminators;
- loop continues that pass through another branch target;
- selection merges that collide with loop-continue targets;
- loop-continue external predecessors and the coupled phi-edge reconciliation.

The native structurizer owns primary CFG construction; the retry relooper owns wholesale fallback
restructuring. This directory is deliberately limited to the retained-module shapes above. Any
expansion needs a primary no-retry mutation census and validation-gated retry evidence so it does
not grow back into an unbounded post-hoc structurizer.
