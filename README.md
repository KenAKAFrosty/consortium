# Consortium

### Best-of-the-best LLM generations at your fingertips

- Provide a list of prompts. Completions are not needed, but these prompts should come from actual use cases of your desired downstream task.

- Embeddings are created for each of the prompts. Then a collection is created of the most-DISsimilar-prompts to the average of the then-current collection, to create an even fence around the latent space of the possible prompts.

- The collection will grow up to a target number of samples (which, naturally, should be drastically fewer than the number of total prompts), _or_ up to a select **co**sine similarity value which acts as a tripwire; when getting the next most-dissimilar-from-group prompt, if the cosine similarity is above this tripwire, no more are added and the collection is complete.

- Then, best-of-the-best LLM responses will be generated against those inputs, through multi-model sampling followed by a multi-model judgement phase where the outputs are source-blind **sort**ed according to each judgement model's preference, then scored in aggregate.

- The final result is a highest-possible-quality data set of `prompt -> completion` pairs, across the most diverse possible prompts for your downstream task.

- Also, this **consortium** style inference could be used as a traditional inference client/sdk as well. Naturally, this will be categorically slower - and expensive - but that may be fine for some use cases!

# Goals & Intents

- High focus on performance. Latency and throughput are first-class targets to optimize.

- Resilience and retry behavior baked in.

- The first passes at this may take direct/concrete/non-generic approaches, but the long-term intent is to genericify any of the key steps (for example: getting the embeddings, the system prompt for the judgement stage, etc.)

- Also will offer a Typescript-friendly client with napi.

- Should probably also consider Python-friendly client with pyo3.

- This crate will also naturally end up being a great source for API clients for all sorts of model inference providers, just to serve the main purpose; as such, the other part of this crate's namesake is having the consortium of LLMs to choose from for whatever application needs you may have.

- User-facing API surface should leverage Rust's strengths, and make invalid states unrepresentable. It is **not** a design goal for the user-facing clients to use an OpenAPI spec

# License

Licensed under either of

- Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
- MIT license (http://opensource.org/licenses/MIT)

at your option.
