# Werewolf agent — deterministic, model-free

Entry for make.fun `werewolf-agent-01`: build the best Werewolf agent.
30 games a trial, 8 players (2 werewolves, seer, doctor, 4 villagers),
against seven `claude-haiku-5-5` house players. Score is
`1000 * (wolf wins/15 + village wins/15) / 2`.

## What this is

A single-file TypeScript program (`main.ts`, Node 22 stdlib only) that
makes **no model calls at all**. Every decision is a pure function of
the view the env served for that one game, the trial seed
(`MAKEFUN_TRIAL_SEED`) and the game number. No wall clock, no
`Math.random`, no cross-game state, no network except the env at
`MAKEFUN_ENV_URL`. A trial therefore replays byte-for-byte.

Run: `node /bot/main.ts` (the grader's command). It reads
`MAKEFUN_PROMPT_FILE` when present but no decision depends on it.

## Strategy

All inference is rebuilt from the view on every decision — no mutable
per-game state, so nothing can leak between games or between the live
pass and the replay.

**Suspicion scores** (rebuilt per decision from the public log):
- Day votes judged against roles revealed since: voting for a revealed
  wolf is village-like (−22, −30 if first voter); voting out a revealed
  villager is wolfy (+14, +22 if first); voting out the seer/doctor
  is worse (+18/+26). Abstaining while a villager is exiled: +3.
- Speech: accusations (+5 to the accused, −2 to the accuser),
  defences (+3 to the defender, −4 to the defended), role claims,
  and passes (+4 each, capped). Accusing a revealed wolf clears the
  accuser (−18); accusing a revealed villager damns them (+10);
  defending a revealed wolf damns the defender (+20).
- Seer claims: a claim is +6 by itself; naming a revealed villager as
  a wolf is +40; naming a revealed wolf is −30. Two live claimants
  naming different wolves both get +12 (later claimant +8). A claim
  contradicting our own inspections (as seer) is +60. The parser also
  reads the inspection phrasing ("I inspected seat 6 on night 1: a
  werewolf") -- the earlier version only read "seat X is a wolf", so
  that whole class of claims was missed. A dead seer-claimant's named
  wolf is now a hard fact, not +25: the village always votes that
  seat, and says so out loud citing the dead seer. A dead accuser's
  last target gets +12 (a dead wolf's gets −8).
- Hard facts override scores: a seer-found wolf is +1000 (always
  voted/revealed), a seer-cleared seat and the werewolf partner are
  −1000 (never voted, accused or killed).

**Votes** join the emerging plurality: score + 10 × votes already cast
for that seat today, +40 if a credible seer claim names them. A tie
exiles nobody, so the village must not split.

**Werewolf kill:** the second voter always agrees with the partner's
pick (a kill needs unanimity). The first voter picks the most
dangerous villager: a seer claimant first (+45), then accurate /
persuasive villagers, minus 0.7 × suspicion (never waste a night on a
seat the village is about to exile anyway).

**Seer:** inspect the most suspicious uninspected seat (seer
claimants first, to catch liars). Reveal a found wolf at once, in the
open, with the night number — then keep pushing that seat.

**Doctor:** protect the likely seer (+50 for a claimant), then
accurate/persuasive villagers; never the previous night's seat (the
env excludes it anyway) and never self (the env excludes it).

**Speech:** short (max 114 chars in mock testing, mean 80 — under the
house prompt's 120-char clip threshold, so the point survives),
first-person, citing real log events (seat numbers, votes, deaths,
claims). A seer reveals a found wolf immediately; a wolf never claims
seer or doctor and deflects from a pressured partner; a doctor never
claims doctor. Day 1 with no lead, the agent reports the night and
stays engaged instead of accusing blindly.

**Ties** are broken by `sha256(trial seed, game, day, label, seat)`,
never by arrival order or a stateful RNG stream.

## Files

- `main.ts` — the whole agent (observe loop + analysis + decisions)
- `LICENSE` — MIT, copied from the kit's starter
- `README.md` — this file

## Local testing (mock only)

No Docker and no Anthropic key were available, so the real
`grade-local` was **not** run. Instead a scripted mock env
(`/tmp/werewolf-mock/mock-env.mjs`, not part of this repo) served the
30 practice trial-0 deals with deterministic hash-seeded house
players, and the bot was run against it:

- Run 1 (seed `test-seed-1`): 224 decisions, 0 fallbacks, 30/30 games
  finished in 0.7 s; no 400/409, no null targets, no statement over
  500 chars.
- Runs A/B (same seed, after a wording fix): 224 decisions each,
  **identical action logs and identical outcomes** — determinism
  confirmed.
- Seed 2: 221 decisions, 0 fallbacks, 30/30 finished, no crashes.
- Mock scores: 467 (seed 1: wolf 0.87, village 0.07) and 500 (seed 2:
  wolf 0.93, village 0.07). Village vote accuracy in the mock was
  0.43 (10/23) vs ~0.29 random. The mock house votes by hash, not by
  reasoning, so its village cannot coordinate at all — these scores
  understate play against the real Haiku house players, who do follow
  stated leads. They are not a prediction of the real score.
- v2 re-run (dead-seer hard fact + inspection-phrasing parser, 11 Nov):
  same mock scores, 467 and 500, 0 fallbacks, determinism re-verified
  (identical action logs on repeat). A unit test
  (/tmp/test-dead-seer.mjs, not in this repo) proves the new path:
  a claimant who died revealed as seer promotes their named wolf to
  a hard fact, the vote targets it, and the speech cites the dead
  seer -- while a claimant who died revealed as a villager does not.
  It later fired end-to-end in a full mock trial (trial-2, game 30),
  but the game was still lost: the mock house votes by hash and
  ignores speech, so only real Haiku players can show the benefit.
- All three practice trials (mock, seed test-seed-1): trial-0 467,
  trial-1 467 (wolf 13/15, village 1/15), trial-2 500 (wolf 13/15,
  village 2/15). No crashes or bad acts on any instance. Village
  is the bottleneck everywhere (1-2/15); wolf is stable at 13/15.

## Known weaknesses vs the Opus baseline

- The baseline is Claude Opus 5.5 reasoning freely each decision;
  this agent's speech is templated and short. Against real Haiku
  players, fluent, adaptive persuasion may coordinate the village
  better than fixed templates — especially on village side, where
  winning needs 6 strangers to vote together.
- Suspicion weights are hand-set, never fitted to real house play:
  no real trial has been run yet, so the weights are priors, not
  measurements.
- The agent never bluffs as seer/doctor and never counter-claims as
  a wolf; a wolf cornered late has no escape play beyond deflection.
- No use of the $15 model budget: if the heuristic underperforms
  Opus in a real grade-local run, adding a model call for the hardest
  decisions (with the same determinism rules) is the obvious next step.

## Still needed before any submission

1. A real `grade-local` run (needs Docker + an Anthropic key — the
   house players run on the entrant's key locally).
2. A make.fun account + a public GitHub repo created after the
   competition opened, with this code at the root, plus the
   coding-agent transcripts bundle for the transcript gate.
3. Nothing here creates an account, a keypair, an API key, a repo,
   or a submission — this directory is preparation only.
