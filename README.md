# Consortium

### Evergreen best-of-best LLM generations at your fingertips

- Provide a list of prompts. Completions are not needed, but these prompts should come from actual use cases of your desired downstream task.
- Embeddings are created for each of the prompts. Then a collection is created of the most-**co**sine-DISsimilar-prompts to the average of the then-current collection, to create an even fence around the latent space of the possible prompts.
- The collection will grow up to a target number of samples, which naturally should be drastically fewer than the number of total prompts, or up to a select cosine similarity value which acts as a tripwire; when getting the next most-dissimilar-from-group prompt, if the cosine similarity is above this tripwire, no more are added and the collection is complete.
- Then, best-of-the-best LLM responses will be generated against those inputs through multi-model sampling followed by a multi-model judgement phase where the outputs are source-blind **sort**ed according to each judgement model's preference, then scored in aggregate.
- The final result is a highest-possible-quality data set of `prompt -> completion` pairs, across the most diverse possible prompts for your downstream task.
- This same **consortium**-style inference can also be used as a traditional inference client/SDK. It will categorically be slower and more expensive, but that tradeoff may be fine for some use cases.

## Goals & Intents

- High focus on performance. Latency and throughput are first-class targets to optimize.
- Resilience and retry behavior should be baked in.
- Early passes may take direct, concrete, non-generic approaches, but the long-term intent is to genericify key steps such as embeddings and judgement prompts.
- Offer a TypeScript-friendly client with `napi`.
- Likely also offer a Python-friendly client with `pyo3`.
- This crate will naturally become a good source of normalized API clients for multiple inference providers as a consequence of the main goal.
- The user-facing API should leverage Rust's strengths and make invalid states unrepresentable.
- It is **not** a design goal for user-facing clients to mirror an OpenAPI spec.

## Docs

- [Contributing](./CONTRIBUTING.md): code style, correctness, robustness, performance, and review expectations.
- [Lab](./lab/README.md): working notes, plans, decisions, and templates.
- [Current Implementation Plan](./lab/plans/2026-05-15-implementation-plan.md): active implementation sequencing.

## License

Licensed under either of:

- Apache License, Version 2.0 ([LICENSE-APACHE](./LICENSE-APACHE))
- MIT license ([LICENSE-MIT](./LICENSE-MIT))

at your option.


## Good Starting Points
[Contributing](./CONTRIBUTING.md) | [Lab](./lab/README.md) | [Plans](./lab/plans/README.md) | [Journal](./lab/journal/README.md) | [Decisions](./lab/decisions/README.md)
