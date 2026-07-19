# R002y Failed Actions

## Eager Host Import

The first client draft imported `transformers` at module load. The benchmark
image includes it, but the lightweight host helper test does not. Static tests
failed before any server or GPU run. The import now occurs only inside the
container-executed `run()` path; pure prompt helpers remain dependency-free.

## Missing Executable Bit

The first smoke command exited `126` before Docker because the newly added
shell script lacked its executable bit. No server or GPU process started. The
scripts were marked executable before the identical smoke was repeated.

## First-Touch Smoke Is Not A Control

The successful smoke measured turn-0/turn-1 GPU boundaries at
`875.840/196.519ms`, then `82.632-112.966ms` for turns 2-5. The cache
progression was valid, but startup had not exercised the six append-only
prefill shapes. These first-touch values are validation-only and excluded.
Every measured process now runs an unarmed six-turn shape warmup first.

## Empty-Text First Token

The first four-repeat control stopped after 17 valid panels when a successful
response reported one completion token whose decoded text was empty. The
client required non-empty text and therefore left TTFT unset. This process is
excluded in full. Requests now set `return_token_ids=true`; receiving a token
ID marks first token even when its decoded string is empty.

## One-Conversation Warmup Missed A Shape

The first complete A/Kineto/B bracket warmed only conversation 0. Both timing
controls then showed the same first-use outlier at conversation 17 turn 5:
`176.231ms` and `173.775ms` for the first `24,993`-token prompt, followed by
`110.881ms` and `118.672ms` for another `24,993`-token prompt. This is shape
specialization, not random drift. All three processes are excluded. Final
processes replay all 15 public primer conversations (`90` requests) unarmed.

## Root-Owned Bind-Mount Panels

The first host analysis attempt could read every panel but could not write a
summary beside files created by the root user inside the client container.
No raw artifact changed. The analyzer now writes compact host-owned summaries
under `analysis/<run>/`, leaving bind-mounted measurements immutable.
