# nn-from-scratch

Building a neural network from raw NumPy — no autograd, no frameworks — to build genuine
first-principles understanding of ML before moving on to paper replication and research
engineering work.

Each checkpoint below is a self-contained step. The goal isn't just working code — it's
being able to derive the math by hand before implementing it, and verifying the
implementation is actually correct (not just error-free).

## Checkpoints

- [ ] **Checkpoint 1 — Forward pass by hand**
  Implement `y = Wx + b` and MSE loss using only NumPy matrix ops.
- [ ] **Checkpoint 2 — Backprop by hand**
  Derive gradients for a 2-layer network (linear → ReLU → linear → loss) on paper first,
  then implement exactly that derivation.
- [ ] **Checkpoint 3 — Gradient checking**
  Implement numerical gradient checking to verify the analytical backprop is correct.
- [ ] **Checkpoint 4 — Train on real data**
  Train the from-scratch network on XOR, then a small MNIST subset.
- [ ] **Checkpoint 5 — Optimizers from scratch**
  Implement momentum and Adam by deriving and coding the update rules directly.

## Why this exists

Moving from applied/agentic AI engineering into research engineering roles at frontier
labs. This project is step one of that path: proving I can build ML systems from first
principles, not just integrate existing models and APIs.

## Progress notes

Each checkpoint folder contains a `notes.md` with the hand-derived math behind that step,
alongside the code.
