# A/B evaluation protocol

Use this protocol when testing whether the Skill improves copy quality. The goal is to measure the instruction package, not to make two hand-written versions look different.

## Experimental arms

- **Control A — brief only:** same model receives the task, facts, and output constraints, plus a neutral request to write the copy. Do not include this Skill, its references, or examples.
- **Treatment B — Skill:** same model receives the identical task, facts, and constraints, with this Skill available and the relevant platform references loaded.

Generate the arms in separate contexts. Creating A after reading the Skill contaminates the control. Do not let either arm see the other's output.

## Hold constant

Keep model and revision, system policy, temperature, seed where available, token limit, language, data version, and number of attempts the same. Record failures and latency instead of silently retrying one arm more often.

The evaluation set must not reuse examples from `references/examples.md`. Do not tune the Skill on the final test set; iterate on a separate validation set.

## Judge

Score both arms with the same rubric:

- instruction following;
- factual grounding;
- platform/brand fit;
- usefulness and specificity;
- safety and claim discipline.

Blind the Judge to arm labels and randomize whether A or B appears first. If possible, use a different model family as Judge and calibrate against human double-labelled samples. Require human review for medical, mental-health, financial, or other high-risk claims.

## Report

Compare:

- pass rate and mean score by dimension;
- category and difficulty slices;
- paired win / tie / loss counts;
- new regressions introduced by the Skill;
- badcase examples with evidence.

For a small smoke test, report descriptive results without claiming general improvement. A meaningful claim needs a larger representative sample and uncertainty estimates such as a paired bootstrap confidence interval.

## Current local project

When `/Users/avecsally/Desktop/bench/xhs-copy-eval` is available, its fixed synthetic dataset and five-dimensional rubric can be used as a smoke-test harness. Keep the existing replay report labelled as a pipeline demonstration; create new control and treatment runs for the Skill comparison rather than overwriting it.

