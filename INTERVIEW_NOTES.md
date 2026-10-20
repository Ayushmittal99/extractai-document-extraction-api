# Interview Notes — ExtractAI

## Architecture in 60 seconds
"ExtractAI is a FastAPI service with one real endpoint, POST /extract. The request is validated by Pydantic (type enum, non-empty text) and a size limit. The extractor wraps the text in a data-only block, calls Gemini through the official google-genai SDK with a response schema and temperature 0, retries only transient errors with bounded exponential backoff, then validates in three steps: JSON parse, Pydantic schema, and report-only semantic checks such as identifier grounding and invoice-total consistency. Errors map to specific HTTP codes with safe messages. The Gemini call sits behind a small backend interface so every test uses a fake and no credits are spent."

## Ten questions
1. **Why structured output instead of 'return JSON' in the prompt?** Schema-constrained decoding makes valid shape far more likely, but I still validate, because it can fail (truncation, safety blocks, schema subsets) and it is not a correctness guarantee.
2. **JSON validation vs Pydantic vs semantic correctness?** Syntax = parseable; Pydantic = right fields/types/ranges; semantics = true to the document. Only the first two can be enforced by code; the third needs checks, evaluation and sometimes humans.
3. **How do you stop hallucinated fields?** Prompt says extract only explicit info, null/[] for missing, temperature 0, and grounding warnings flag emails/phones/invoice numbers not found in the source. It reduces but does not eliminate the risk.
4. **Prompt injection?** Document is untrusted data in delimiters, delimiter-escape neutralised, system instruction forbids following embedded commands, output is only data. Not a guarantee; I'd add adversarial evaluation.
5. **Why Decimal for money?** Floats cause rounding errors (0.1+0.2). Decimal is exact; serialised as string so JSON consumers don't re-float it. Gemini sees it as a number via a JSON-schema override.
6. **Why not fix invoice totals?** Silently "correcting" financial data is dangerous. I report mismatches as warnings and keep original values.
7. **Retry strategy?** Retry 429, 5xx, timeouts, network errors with 1s, 2s, 4s waits, max 3 retries; auth and malformed output are not retried. Bounded to avoid cost spirals (jitter is a next step).
8. **Why different status codes (502 vs 422)?** 422/413 are client mistakes; 502/504 mean the upstream model failed or returned unusable output; 503 means server misconfiguration; 429 passes quota pressure on.
9. **How do you test without Gemini?** Dependency injection: Extractor takes a backend with `generate()`; FastAPI dependency override swaps the extractor. Failure modes become deterministic. Gap: can't measure real accuracy.
10. **How would you evaluate quality?** Labelled synthetic resumes/invoices, field-level precision/recall and exact match for identifiers/amounts, tracked per model version, plus adversarial cases (injection, ambiguous dates, missing fields).
