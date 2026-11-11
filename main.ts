// Deterministic, model-free Werewolf agent for make.fun werewolf-agent-01.
//
// No model calls at all: every decision is a pure function of the view the
// env served for that one game, the trial seed and the game number. No wall
// clock, no Math.random, no cross-game state, no network except the env.
// That makes a trial replay byte-for-byte identical to its live pass.
//
// How it plays (all scores are rebuilt from the view on every decision, so
// no per-game mutable state can leak between games or between passes):
//  * Suspicion is Bayesian-ish bookkeeping over the public log: day votes
//    for/against seats whose roles were later revealed, accusations and
//    defences in speech, role claims and whether they were contradicted by
//    a death or by our own inspections, night deaths (a dead accuser's
//    lead is credible; a dead seer-claimant's named wolf is credible),
//    and passes (dodging the debate is suspicious).
//  * Seer results and the werewolf partner are hard facts, not scores:
//    a known wolf is always voted / revealed, a known non-wolf is never
//    voted, the partner is never voted, accused or killed. A seer
//    claimant who died and was revealed as the real seer is ground
//    truth too: the wolf they named is a hard fact, and the village
//    says so out loud, with the source, so the vote does not split.
//  * Votes join the emerging plurality: a suspect's score is boosted by
//    the votes already cast for them today, so the village does not split
//    and hand the wolves a tie (a tie exiles nobody).
//  * Werewolves coordinate the kill: the second voter always agrees with
//    the partner's pick (a kill needs unanimity); the first voter picks
//    the most dangerous villager -- a credible seer claimant first, then
//    accurate / persuasive villagers -- and never a seat the village is
//    about to exile anyway.
//  * Statements are short (first ~120 chars carry the point, because the
//    house prompt clips old speeches to 120 chars) and cite real log
//    events: seat numbers, votes, deaths, claims. A seer reveals a found
//    wolf at once; a wolf never claims seer or doctor; a doctor never
//    claims doctor; nobody claims to be the moderator.
//  * Ties are broken by a hash of (trial seed, game, day, label, seat),
//    never by arrival order or by a stateful RNG stream.
//
// Run by the grader as `node /bot/main.ts` (Node 22, stdlib only).

import { readFileSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { pathToFileURL } from 'node:url';

type Role = 'werewolf' | 'seer' | 'doctor' | 'villager';
type DecisionKind = 'speak' | 'vote' | 'kill' | 'protect' | 'inspect';

interface Player {
  seat: number;
  name: string;
  alive: boolean;
  role: string | null;
}

interface LogEntry {
  seq: number;
  day: number;
  kind: string;
  seat?: number;
  target?: number | null;
  text?: string;
  round?: number;
  died?: number | null;
  role?: string | null;
  tally?: { seat: number; votes: number }[];
  order?: number[];
  winner?: string | null;
  [k: string]: unknown;
}

interface View {
  day: number;
  phase: string;
  you: {
    seat: number;
    name: string;
    role: string;
    alive: boolean;
    partner: { seat: number; name: string; alive: boolean } | null;
  };
  players: Player[];
  seerResults: { night: number; seat: number; werewolf: boolean }[] | null;
  lastProtect: { night: number; seat: number | null } | null;
  wolfVotes: { night: number; seat: number; target: number | null }[] | null;
  log: LogEntry[];
  outcome: { winner: string | null; reason: string; day: number; winners: number[] } | null;
}

export interface Pending {
  game: number;
  decision: DecisionKind;
  options?: number[];
  max_chars?: number;
  view: View;
}

const isRecord = (v: unknown): v is Record<string, unknown> =>
  v !== null && typeof v === 'object' && !Array.isArray(v);

/** Fold a statement the env's way and cut it to maxChars code points. */
export function foldStatement(text: string, maxChars = 500): string {
  const folded = text
    .replace(/[\t\n\v\f\r\u0085\u2028\u2029]+/g, ' ')
    .replace(/\p{Cc}/gu, '')
    .trim();
  const chars = [...folded];
  return chars.length > maxChars ? chars.slice(0, maxChars).join('').trim() : folded;
}

/** Deterministic 32-bit hash of (seed, parts...); the only tie-breaker. */
export function hashInt(seed: string, ...parts: (string | number)[]): number {
  const h = createHash('sha256')
    .update([seed, ...parts.map(String)].join('|'), 'utf8')
    .digest();
  return h.readUInt32BE(0);
}

/** Pick the best item by score; ties go to the lowest hash (deterministic). */
function pickBest<T>(
  seed: string,
  game: number,
  day: number,
  label: string,
  items: readonly T[],
  score: (x: T) => number,
  tieKey: (x: T) => number,
): T | null {
  let best: T | null = null;
  let bestScore = -Infinity;
  let bestHash = Infinity;
  for (const item of items) {
    const s = score(item);
    const h = hashInt(seed, game, day, label, tieKey(item));
    if (s > bestScore || (s === bestScore && h < bestHash)) {
      best = item;
      bestScore = s;
      bestHash = h;
    }
  }
  return best;
}

const seatName = (view: View, seat: number): string =>
  view.players.find((p) => p.seat === seat)?.name ?? `seat ${seat}`;
const isAlive = (view: View, seat: number): boolean =>
  view.players.find((p) => p.seat === seat)?.alive ?? false;
const revealedRole = (view: View, seat: number): string | null =>
  view.players.find((p) => p.seat === seat)?.role ?? null;

// ---------------------------------------------------------------- analysis

export interface Claim {
  seat: number;
  day: number;
  claimedWolf: number | null;
  claimedClear: number | null;
  text: string;
}

export interface Analysis {
  scores: Map<number, number>;
  knownWolf: Set<number>;
  knownClear: Set<number>;
  seerClaimants: Claim[];
  doctorClaimants: Set<number>;
  accusedBy: Map<number, number[]>; // target -> accusers
  defendedBy: Map<number, number[]>; // target -> defenders
  accusedCount: Map<number, number>;
  passes: Map<number, number>;
  speeches: Map<number, number>;
  evidence: Map<number, string>;
  dayVotesToday: Map<number, number>;
  round: number;
  lastDawn: LogEntry | null;
  lastExile: LogEntry | null;
}

const WOLF_WORD = /\b(werewolf|wolf)\b/i;
const SEER_CLAIM = /\b(i am|i'm|im|i’m)\s+(the\s+)?seer\b|\bas\s+seer\b|\bseer\s+here\b|\bi\s+inspected\b|\bi\s+checked\b/i;
const DOCTOR_CLAIM = /\b(i am|i'm|im|i’m)\s+(the\s+)?doctor\b|\bi\s+protected\b/i;

/**
 * Rebuild every inference from the view alone. Scores are relative and
 * deliberately small except for hard facts (known wolf / clear), so one
 * noisy day-1 vote cannot dominate a later, better-evidenced read.
 */
export function analyze(view: View): Analysis {
  const me = view.you.seat;
  const living = view.players.filter((p) => p.alive).map((p) => p.seat);
  const scores = new Map<number, number>();
  for (const s of living) scores.set(s, 0);
  const bump = (seat: number, n: number): void => {
    if (scores.has(seat)) scores.set(seat, (scores.get(seat) ?? 0) + n);
  };

  const knownWolf = new Set<number>();
  const knownClear = new Set<number>();
  for (const r of view.seerResults ?? []) {
    if (r.werewolf) knownWolf.add(r.seat);
    else knownClear.add(r.seat);
  }
  // Our own role is known to us; a werewolf's partner is known clear to us.
  if (view.you.role === 'werewolf' && view.you.partner) knownClear.add(view.you.partner.seat);

  const seerClaimants: Claim[] = [];
  const doctorClaimants = new Set<number>();
  const accusedBy = new Map<number, number[]>();
  const defendedBy = new Map<number, number[]>();
  const accusedCount = new Map<number, number>();
  const passes = new Map<number, number>();
  const speeches = new Map<number, number>();
  const evidence = new Map<number, string>();
  const setEvidence = (seat: number, text: string, strong: boolean): void => {
    if (!evidence.has(seat) || strong) evidence.set(seat, text);
  };

  // --- day votes, judged against roles revealed since ---
  const votesByDay = new Map<number, LogEntry[]>();
  for (const e of view.log) {
    if (e.kind === 'vote' && typeof e.seat === 'number') {
      const list = votesByDay.get(e.day) ?? [];
      list.push(e);
      votesByDay.set(e.day, list);
    }
  }
  for (const [day, votes] of votesByDay) {
    const exile = view.log.find((e) => e.kind === 'exile' && e.day === day);
    votes.forEach((v, idx) => {
      const voter = v.seat as number;
      const first = idx === 0;
      if (v.target === null || v.target === undefined) {
        // Abstaining while a villager was exiled helps the wolves a little.
        if (exile && exile.seat !== null && revealedRole(view, exile.seat) !== 'werewolf') bump(voter, 3);
        else if (exile && exile.seat !== null) bump(voter, -3);
        return;
      }
      const role = revealedRole(view, v.target);
      if (role === 'werewolf') bump(voter, first ? -30 : -22);
      else if (role === 'villager') bump(voter, first ? 22 : 14);
      else if (role === 'seer' || role === 'doctor') bump(voter, first ? 26 : 18);
      // Voting for a seat that was exiled that day is on the record either way.
      if (exile && exile.seat === v.target && role === 'villager') {
        setEvidence(voter, `voted seat ${v.target} out on day ${day} and they were a villager`, true);
      }
    });
  }

  // --- speech: claims, accusations, defences, passes ---
  // Track each accuser's targets with the day, to judge them once roles reveal.
  const accusations: { by: number; target: number; day: number }[] = [];
  const defences: { by: number; target: number; day: number }[] = [];
  for (const e of view.log) {
    if (e.kind === 'pass' && typeof e.seat === 'number') {
      passes.set(e.seat, (passes.get(e.seat) ?? 0) + 1);
      bump(e.seat, 4);
      continue;
    }
    if (e.kind !== 'speech' || typeof e.seat !== 'number' || typeof e.text !== 'string') continue;
    const seat = e.seat;
    const text = e.text;
    speeches.set(seat, (speeches.get(seat) ?? 0) + 1);
    if (SEER_CLAIM.test(text)) {
      let claimedWolf: number | null = null;
      let claimedClear: number | null = null;
      const wolfM = text.match(/seat\s+(\d+)\s+(is|was)\s+(a\s+)?(werewolf|wolf)/i)
        ?? text.match(/(?:seat\s+)?(\d+)\s+is\s+(a\s+)?(werewolf|wolf)/i)
        // The canonical inspection phrasing -- "I inspected seat 6 on
        // night 1: a werewolf" -- names no "is": match the inspected
        // seat when a wolf word follows it in the same sentence.
        ?? text.match(/\b(?:inspected|checked)\s+(?:seat\s+)?(\d+)[^.]*\b(?:werewolf|wolf)\b/i);
      if (wolfM) claimedWolf = Number(wolfM[1]);
      const clearM = text.match(/seat\s+(\d+)\s+(is|was)\s+not\s+(a\s+)?(werewolf|wolf)/i)
        ?? text.match(/seat\s+(\d+)\s+(is|was)\s+(a\s+)?villager/i)
        ?? text.match(/\b(?:inspected|checked)\s+(?:seat\s+)?(\d+)[^.]*\b(?:not\s+(?:a\s+)?(?:werewolf|wolf)|villager)\b/i);
      if (clearM) claimedClear = Number(clearM[1]);
      seerClaimants.push({ seat, day: e.day, claimedWolf, claimedClear, text });
      bump(seat, 6); // a claim may be a wolf's lie; keep the prior small
    }
    if (DOCTOR_CLAIM.test(text)) doctorClaimants.add(seat);

    // Accusations: "seat X is a wolf", "vote seat X", "exile seat X".
    const accuseTargets = new Set<number>();
    for (const m of text.matchAll(/seat\s+(\d+)\s+(is|was)\s+(a\s+)?(werewolf|wolf)/gi)) accuseTargets.add(Number(m[1]));
    for (const m of text.matchAll(/\b(?:vote|exile)\s+(?:seat\s+)?(\d+)/gi)) accuseTargets.add(Number(m[1]));
    for (const target of accuseTargets) {
      if (target === seat || target < 1 || target > 8) continue;
      accusations.push({ by: seat, target, day: e.day });
      accusedCount.set(target, (accusedCount.get(target) ?? 0) + 1);
      const list = accusedBy.get(target) ?? [];
      if (!list.includes(seat)) list.push(seat);
      accusedBy.set(target, list);
      bump(target, 5);
      bump(seat, -2); // naming a suspect is village-like work
    }
    // Defences: "seat X is not a wolf", "seat X is a villager", "trust seat X".
    const defendTargets = new Set<number>();
    for (const m of text.matchAll(/seat\s+(\d+)\s+(is|was)\s+not\s+(a\s+)?(werewolf|wolf)/gi)) defendTargets.add(Number(m[1]));
    for (const m of text.matchAll(/seat\s+(\d+)\s+(is|was)\s+(a\s+)?villager/gi)) defendTargets.add(Number(m[1]));
    for (const m of text.matchAll(/\btrust\s+(?:seat\s+)?(\d+)/gi)) defendTargets.add(Number(m[1]));
    for (const target of defendTargets) {
      if (target === seat || target < 1 || target > 8) continue;
      defences.push({ by: seat, target, day: e.day });
      const list = defendedBy.get(target) ?? [];
      if (!list.includes(seat)) list.push(seat);
      defendedBy.set(target, list);
      bump(target, -4);
      bump(seat, 3);
    }
    if (/\b(i am|i'm|im|i’m)\s+(not\s+)?(a\s+)?villager\b/i.test(text) || /\bi'?m\s+not\s+(a\s+)?wolf\b/i.test(text)) {
      bump(seat, 3); // protesting innocence unprompted is slightly wolfy
    }
  }

  // --- judge accusations / defences once the target's role is revealed ---
  for (const a of accusations) {
    const role = revealedRole(view, a.target);
    if (role === 'werewolf') {
      bump(a.by, -18);
      setEvidence(a.by, `called seat ${a.target} a wolf and they were one`, false);
    } else if (role === 'villager') {
      bump(a.by, 10);
      setEvidence(a.by, `accused seat ${a.target}, who turned out a villager`, true);
    } else if (role === 'seer' || role === 'doctor') {
      bump(a.by, 14);
    }
  }
  for (const d of defences) {
    const role = revealedRole(view, d.target);
    if (role === 'werewolf') {
      bump(d.by, 20);
      setEvidence(d.by, `defended seat ${d.target}, who turned out a werewolf`, true);
    } else if (role === 'villager') {
      bump(d.by, -10);
    }
  }

  // --- judge seer claims ---
  for (const c of seerClaimants) {
    if (c.claimedWolf !== null) {
      const role = revealedRole(view, c.claimedWolf);
      if (role === 'villager' || role === 'seer' || role === 'doctor') {
        bump(c.seat, 40);
        setEvidence(c.seat, `claimed seer and named seat ${c.claimedWolf}, who was a ${role}`, true);
      } else if (role === 'werewolf') {
        bump(c.seat, -30);
        bump(c.claimedWolf, 25);
      }
      // Contradiction with our own inspections (we are the seer).
      if (view.you.role === 'seer' && knownClear.has(c.claimedWolf)) bump(c.seat, 60);
      if (view.you.role === 'seer' && knownWolf.has(c.claimedWolf)) bump(c.seat, -20);
    }
    if (c.claimedClear !== null && view.you.role === 'seer' && knownWolf.has(c.claimedClear)) {
      bump(c.seat, 60);
    }
    // A dead claimant's role settles their claim. A revealed seer's
    // inspection is ground truth, not a hint: if they died as the seer,
    // the seat they named a wolf IS a wolf (and the seat they cleared
    // IS clear), unless our own direct inspection says otherwise --
    // our own seer results win any conflict. Promoting this to a hard
    // fact means the village always votes that seat and never votes
    // the cleared one, instead of merely leaning that way (+25).
    const role = revealedRole(view, c.seat);
    if (role === 'seer' && c.claimedWolf !== null) {
      if (!knownClear.has(c.claimedWolf)) knownWolf.add(c.claimedWolf);
      else bump(c.claimedWolf, 25);
    }
    if (role === 'seer' && c.claimedClear !== null) {
      if (!knownWolf.has(c.claimedClear)) knownClear.add(c.claimedClear);
    }
    if (role === 'werewolf' && c.claimedWolf !== null) bump(c.claimedWolf, -20);
  }
  // Two different seer claims cannot both be true.
  const distinctClaimants = [...new Set(seerClaimants.map((c) => c.seat))];
  if (distinctClaimants.length >= 2) {
    for (const s of distinctClaimants) bump(s, 12);
    // The later claimant is the likelier counter-claim.
    const sorted = [...seerClaimants].sort((a, b) => a.day - b.day || a.seat - b.seat);
    const last = sorted[sorted.length - 1];
    if (last) bump(last.seat, 8);
  }
  // A seer claimant who survives night after night while others die is
  // either protected every night or a wolf faking the role.
  if (view.day >= 3) {
    for (const s of distinctClaimants) {
      if (isAlive(view, s) && view.you.role !== 'seer') bump(s, 8);
    }
  }

  // --- night deaths: a dead player's last accusation is credible ---
  for (const e of view.log) {
    if (e.kind !== 'dawn' || e.died === null || e.died === undefined) continue;
    const deadRole = e.role ?? revealedRole(view, e.died);
    const lastAcc = [...accusations].reverse().find((a) => a.by === e.died && a.day <= e.day);
    if (lastAcc) {
      if (deadRole === 'werewolf') bump(lastAcc.target, -8);
      else if (deadRole) bump(lastAcc.target, 12);
    }
    // A seer claimant killed at night was probably the real seer.
    if (deadRole === 'seer') {
      const claim = seerClaimants.find((c) => c.seat === e.died && c.claimedWolf !== null);
      if (claim?.claimedWolf) bump(claim.claimedWolf, 25);
    }
  }

  // Cap the pass penalty so a quiet house player is not damned by it alone.
  for (const [seat, n] of passes) {
    if (n > 3) bump(seat, -(n - 3) * 4); // undo the excess above the cap
  }

  // Hard facts override scores.
  for (const s of knownWolf) scores.set(s, 1000);
  for (const s of knownClear) scores.set(s, -1000);
  scores.set(me, -1000);

  // Evidence strings for the top suspects, from the strongest fact on file.
  for (const s of living) {
    if (s === me) continue;
    if (evidence.has(s)) continue;
    const accusers = accusedBy.get(s) ?? [];
    if (accusers.length >= 2) {
      evidence.set(s, `was named a wolf by seat ${accusers[0]} and seat ${accusers[1]}`);
    } else if (accusers.length === 1) {
      evidence.set(s, `was named a wolf by seat ${accusers[0]}`);
    } else if ((passes.get(s) ?? 0) >= 2) {
      evidence.set(s, `passed ${passes.get(s)} times instead of talking`);
    } else if ((speeches.get(s) ?? 0) === 0 && view.day >= 2) {
      evidence.set(s, `has said almost nothing for ${view.day - 1} days`);
    } else {
      evidence.set(s, `has votes that do not line up with who turned out to be a wolf`);
    }
  }

  // Today's votes so far, and which debate round this is.
  const dayVotesToday = new Map<number, number>();
  for (const e of view.log) {
    if (e.kind === 'vote' && e.day === view.day && typeof e.target === 'number') {
      dayVotesToday.set(e.target, (dayVotesToday.get(e.target) ?? 0) + 1);
    }
  }
  const turnsToday = view.log.filter(
    (e) => (e.kind === 'speech' || e.kind === 'pass') && e.day === view.day,
  ).length;
  const livingCount = living.length || 1;
  const round = 1 + Math.floor(turnsToday / livingCount);
  const dawns = view.log.filter((e) => e.kind === 'dawn');
  const exiles = view.log.filter((e) => e.kind === 'exile');
  return {
    scores,
    knownWolf,
    knownClear,
    seerClaimants,
    doctorClaimants,
    accusedBy,
    defendedBy,
    accusedCount,
    passes,
    speeches,
    evidence,
    dayVotesToday,
    round: Math.min(2, Math.max(1, round)),
    lastDawn: dawns.length > 0 ? dawns[dawns.length - 1]! : null,
    lastExile: exiles.length > 0 ? exiles[exiles.length - 1]! : null,
  };
}

// ---------------------------------------------------------------- decisions

/** The most credible seer claim naming a living seat, or null. */
function credibleSeerClaim(view: View, a: Analysis): Claim | null {
  const live = a.seerClaimants.filter(
    (c) =>
      c.claimedWolf !== null &&
      isAlive(view, c.claimedWolf) &&
      isAlive(view, c.seat) &&
      !a.knownWolf.has(c.seat) &&
      c.seat !== view.you.seat,
  );
  if (live.length === 0) return null;
  // If two live claims name different wolves, trust the less suspicious claimant.
  let chosen: Claim | null = null;
  let chosenScore = Infinity;
  for (const c of live) {
    const s = a.scores.get(c.seat) ?? 0;
    if (s < chosenScore) {
      chosenScore = s;
      chosen = c;
    }
  }
  // A claim is only credible if no equally live claim contradicts it, or
  // this claimant is clearly the less suspicious of the two.
  const contradicting = live.filter((c) => c.claimedWolf !== chosen?.claimedWolf);
  if (contradicting.length > 0) {
    const otherScore = Math.min(...contradicting.map((c) => a.scores.get(c.seat) ?? 0));
    if (chosenScore > otherScore - 10) return null;
  }
  return chosen;
}

function livingOthers(view: View): number[] {
  return view.players.filter((p) => p.alive && p.seat !== view.you.seat).map((p) => p.seat);
}

/** Vote: known wolf first, else highest suspicion, boosted by today's tally. */
export function chooseVote(view: View, a: Analysis, options: number[], seed: string, game: number): number | null {
  if (options.length === 0) return null;
  for (const s of options) if (a.knownWolf.has(s)) return s;
  const claim = view.you.role === 'werewolf' ? null : credibleSeerClaim(view, a);
  const scored = options.filter((s) => !a.knownClear.has(s));
  const pool = scored.length > 0 ? scored : options;
  return pickBest(
    seed, game, view.day, 'vote', pool,
    (s) => (a.scores.get(s) ?? 0)
      + 10 * (a.dayVotesToday.get(s) ?? 0)
      + (claim && claim.claimedWolf === s ? 40 : 0),
    (s) => s,
  );
}

/** Kill: agree with the partner's pick tonight, else the most dangerous villager. */
export function chooseKill(view: View, a: Analysis, options: number[], seed: string, game: number): number | null {
  if (options.length === 0) return null;
  const partner = view.you.partner?.seat;
  if (partner !== undefined) {
    const partnerVote = (view.wolfVotes ?? []).find(
      (v) => v.night === view.day && v.seat === partner && v.target !== null,
    );
    if (partnerVote?.target !== undefined && partnerVote.target !== null && options.includes(partnerVote.target)) {
      return partnerVote.target; // unanimity is what makes a kill happen
    }
  }
  return pickBest(
    seed, game, view.day, 'kill', options,
    (s) => {
      let v = 0;
      if (a.seerClaimants.some((c) => c.seat === s)) v += 45; // a seer claimant is the top target
      if (a.doctorClaimants.has(s)) v += 15;
      // Accurate, persuasive villagers are dangerous: they find wolves.
      for (const e of view.log) {
        if (e.kind === 'vote' && e.seat === s && typeof e.target === 'number' && revealedRole(view, e.target) === 'werewolf') v += 20;
      }
      v += Math.min(15, (a.speeches.get(s) ?? 0) * 3);
      // Never waste a night on a seat the village is about to exile anyway.
      v -= 0.7 * (a.scores.get(s) ?? 0);
      if ((a.scores.get(s) ?? 0) > 25) v -= 10;
      return v;
    },
    (s) => s,
  );
}

/** Protect: the likely seer first, then accurate / persuasive villagers. */
export function chooseProtect(view: View, a: Analysis, options: number[], seed: string, game: number): number | null {
  if (options.length === 0) return null;
  return pickBest(
    seed, game, view.day, 'protect', options,
    (s) => {
      let v = 0;
      if (a.seerClaimants.some((c) => c.seat === s)) v += 50;
      if (a.doctorClaimants.has(s)) v += 5;
      for (const e of view.log) {
        if (e.kind === 'vote' && e.seat === s && typeof e.target === 'number' && revealedRole(view, e.target) === 'werewolf') v += 25;
      }
      v += Math.min(10, (a.speeches.get(s) ?? 0) * 2);
      v -= 0.5 * (a.scores.get(s) ?? 0);
      // A seat that survived last night while nobody died may be the wolves'
      // repeat target -- but if we protected them last night we cannot repeat.
      const last = view.lastProtect;
      if (last && last.seat === s) v -= 1000; // env already excludes this; belt and braces
      return v;
    },
    (s) => s,
  );
}

/** Inspect: the most suspicious uninspected seat; check seer claimants first. */
export function chooseInspect(view: View, a: Analysis, options: number[], seed: string, game: number): number | null {
  if (options.length === 0) return null;
  const inspected = new Set((view.seerResults ?? []).map((r) => r.seat));
  const pool = options.filter((s) => !inspected.has(s) && !a.knownClear.has(s));
  const use = pool.length > 0 ? pool : options;
  return pickBest(
    seed, game, view.day, 'inspect', use,
    (s) => (a.scores.get(s) ?? 0) + (a.seerClaimants.some((c) => c.seat === s) ? 40 : 0),
    (s) => s,
  );
}

function mySpeechToday(view: View): string[] {
  return view.log
    .filter((e) => e.kind === 'speech' && e.day === view.day && e.seat === view.you.seat && typeof e.text === 'string')
    .map((e) => e.text as string);
}

/** One statement (or null to pass). Short, first-person, citing real events. */
export function chooseSpeak(view: View, a: Analysis, seed: string, game: number, maxChars: number): string | null {
  const me = view.you.seat;
  const role = view.you.role as Role;
  const others = livingOthers(view);
  if (others.length === 0) return null;
  const suspect = pickBest(
    seed, game, view.day, 'speak', others,
    (s) => a.scores.get(s) ?? 0,
    (s) => s,
  );
  const ev = (s: number): string => a.evidence.get(s) ?? 'their votes do not line up';
  const saidToday = mySpeechToday(view);
  const alreadySaidSeer = saidToday.some((t) => /i am the seer/i.test(t));

  // --- seer with a found wolf: reveal at once, then keep pushing it ---
  if (role === 'seer') {
    const wolf = [...a.knownWolf].filter((s) => isAlive(view, s)).sort((x, y) => x - y)[0];
    if (wolf !== undefined) {
      const result = (view.seerResults ?? []).find((r) => r.seat === wolf);
      const night = result ? result.night : view.day - 1;
      if (alreadySaidSeer) return foldStatement(`Seat ${wolf} is a werewolf, I inspected them on night ${night}. Vote seat ${wolf}.`, maxChars);
      return foldStatement(`I am the seer. I inspected seat ${wolf} on night ${night}: a werewolf. Vote seat ${wolf}.`, maxChars);
    }
  }

  // --- village / doctor: a dead revealed seer is ground truth ---
  // If a seer claimant died and was revealed as the real seer, the wolf
  // they named is a fact, not a suspicion. Say so plainly, with the
  // source, so the house players vote that seat instead of splitting:
  // a live claim can be a wolf's lie, a dead seer's cannot.
  if (role !== 'werewolf' && role !== 'seer') {
    for (const c of a.seerClaimants) {
      if (
        c.claimedWolf !== null &&
        isAlive(view, c.claimedWolf) &&
        revealedRole(view, c.seat) === 'seer' &&
        a.knownWolf.has(c.claimedWolf)
      ) {
        return foldStatement(
          `Seat ${c.seat} was the seer and named seat ${c.claimedWolf} a wolf before they died. Vote seat ${c.claimedWolf}.`,
          maxChars,
        );
      }
    }
  }

  // --- village / doctor: echo a credible seer claim ---
  if (role !== 'werewolf' && role !== 'seer') {
    const claim = credibleSeerClaim(view, a);
    if (claim?.claimedWolf !== null && claim?.claimedWolf !== undefined) {
      const w = claim.claimedWolf;
      return foldStatement(
        `Seat ${claim.seat} says they inspected seat ${w} as a wolf, and ${ev(w)}. I agree, vote seat ${w}.`,
        maxChars,
      );
    }
    const claimants = [...new Set(a.seerClaimants.map((c) => c.seat))].filter((s) => isAlive(view, s));
    if (claimants.length >= 2) {
      const sorted = [...a.seerClaimants]
        .filter((c) => c.claimedWolf !== null && isAlive(view, c.claimedWolf as number))
        .sort((x, y) => (a.scores.get(x.seat) ?? 0) - (a.scores.get(y.seat) ?? 0));
      if (sorted.length >= 2) {
        const first = sorted[0]!;
        const second = sorted[1]!;
        return foldStatement(
          `Two seer claims can't both be true. Seat ${first.seat} names seat ${first.claimedWolf}, seat ${second.seat} names seat ${second.claimedWolf}. I trust seat ${first.seat} more.`,
          maxChars,
        );
      }
    }
  }

  // --- werewolf: deflect from the partner when they are under pressure ---
  if (role === 'werewolf' && view.you.partner) {
    const partner = view.you.partner.seat;
    const partnerScore = a.scores.get(partner) ?? 0;
    const partnerAccusers = (a.accusedBy.get(partner) ?? []).length;
    if (isAlive(view, partner) && (partnerScore > 20 || partnerAccusers >= 2) && suspect !== null && suspect !== partner) {
      return foldStatement(
        `Seat ${partner} looks like a villager to me. Seat ${suspect} is the better lead: ${ev(suspect)}.`,
        maxChars,
      );
    }
  }

  // --- accuse the top suspect, with the strongest real evidence ---
  if (suspect !== null && (a.scores.get(suspect) ?? 0) >= 8) {
    if (a.round === 2 && saidToday.some((t) => t.includes(`seat ${suspect}`))) {
      return foldStatement(`Still seat ${suspect}: ${ev(suspect)}. Let's not split the vote.`, maxChars);
    }
    return foldStatement(`Seat ${suspect} ${ev(suspect)}. I think seat ${suspect} is a wolf.`, maxChars);
  }

  // --- day 1 / no lead: report the night, stay engaged, accuse nothing ---
  if (view.day === 1 && a.round === 1) {
    const dawn = a.lastDawn;
    if (dawn && dawn.died !== null && dawn.died !== undefined) {
      return foldStatement(
        `Seat ${dawn.died} (${seatName(view, dawn.died)}) was a ${dawn.role ?? 'villager'} last night. No clear lead yet, I am watching who pushes first.`,
        maxChars,
      );
    }
    return foldStatement('No one died last night, the doctor may have saved someone. Watching who pushes first today.', maxChars);
  }
  if (suspect !== null) {
    return foldStatement(`Seat ${suspect} ${ev(suspect)}. Worth a closer look today.`, maxChars);
  }
  return null;
}

/** The /act body for one pending decision: a pure function of the pending view. */
export function decideAction(p: Pending, seed: string): Record<string, unknown> {
  const view = p.view;
  const a = analyze(view);
  const options = p.options ?? [];
  switch (p.decision) {
    case 'speak': {
      const text = chooseSpeak(view, a, seed, p.game, p.max_chars ?? 500);
      return text === null ? { game: p.game, decision: 'speak', pass: true } : { game: p.game, decision: 'speak', text };
    }
    case 'vote':
      return { game: p.game, decision: 'vote', target: chooseVote(view, a, options, seed, p.game) };
    case 'kill':
      return { game: p.game, decision: 'kill', target: chooseKill(view, a, options, seed, p.game) };
    case 'protect':
      return { game: p.game, decision: 'protect', target: chooseProtect(view, a, options, seed, p.game) };
    case 'inspect':
      return { game: p.game, decision: 'inspect', target: chooseInspect(view, a, options, seed, p.game) };
  }
}

export function fallbackFor(p: Pick<Pending, 'game' | 'decision'>): Record<string, unknown> {
  return p.decision === 'speak'
    ? { game: p.game, decision: 'speak', pass: true }
    : { game: p.game, decision: p.decision, target: null };
}

// ---------------------------------------------------------------- run loop

interface Observed {
  games: number;
  finished: number;
  done: boolean;
  pending: Pending[];
}

const sleep = (ms: number): Promise<void> => new Promise((r) => setTimeout(r, ms));

export async function run(o: { envUrl?: string; seed?: string } = {}): Promise<{ decisions: number; fallbacks: number }> {
  const envUrl = (o.envUrl ?? process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
  const seed = o.seed ?? process.env.MAKEFUN_TRIAL_SEED ?? 'werewolf-agent-01';
  if (envUrl === '') throw new Error('MAKEFUN_ENV_URL is needed');
  // The brief is read when present (the grader always provides it); the
  // decisions below do not depend on it.
  try {
    const file = process.env.MAKEFUN_PROMPT_FILE ?? '';
    if (file !== '') readFileSync(file, 'utf8');
  } catch {
    // A missing brief does not stop a deterministic player.
  }
  let decisions = 0;
  let fallbacks = 0;

  const post = async (path: string, body: unknown): Promise<{ status: number; body: unknown }> => {
    const res = await fetch(`${envUrl}${path}`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify(body),
    });
    const text = await res.text();
    let parsed: unknown = null;
    try {
      parsed = JSON.parse(text);
    } catch {
      parsed = text;
    }
    return { status: res.status, body: parsed };
  };

  for (;;) {
    // /observe long-polls up to 20s when nothing waits; a transient fetch
    // failure is retried a few times (waiting changes timing, never a request).
    let obs: Observed | null = null;
    for (let attempt = 0; attempt < 5; attempt++) {
      try {
        const r = await post('/observe', {});
        if (r.status === 200 && isRecord(r.body)) {
          obs = r.body as unknown as Observed;
          break;
        }
        throw new Error(`POST /observe answered ${r.status}`);
      } catch (err) {
        if (attempt === 4) throw err;
        await sleep(250);
      }
    }
    if (obs === null) throw new Error('POST /observe gave no answer');
    if (obs.done) break;
    // Decisions are instant (no model calls), so answer them one at a time,
    // in the order the env lists them. One request at a time keeps the
    // replay order identical to the live order.
    for (const p of obs.pending) {
      decisions += 1;
      let act: Record<string, unknown>;
      try {
        act = decideAction(p, seed);
      } catch {
        act = fallbackFor(p);
        fallbacks += 1;
      }
      let posted: { status: number; body: unknown };
      try {
        posted = await post('/act', act);
      } catch {
        continue; // the next /observe will list it again
      }
      if (posted.status === 400 && JSON.stringify(act) !== JSON.stringify(fallbackFor(p))) {
        fallbacks += 1;
        try {
          await post('/act', fallbackFor(p));
        } catch {
          // Next /observe lists it again.
        }
      } else if (posted.status === 409) {
        // Already answered / moved on: nothing to do.
      } else if (posted.status !== 200) {
        fallbacks += 1;
      }
    }
  }
  process.stdout.write(`werewolf: ${decisions} decisions, ${fallbacks} fallbacks, done\n`);
  return { decisions, fallbacks };
}

const isMain = process.argv[1] !== undefined && import.meta.url === pathToFileURL(process.argv[1]).href;
if (isMain) {
  try {
    await run();
    process.exitCode = 0;
  } catch (err) {
    process.stderr.write(`werewolf: ${(err as Error).stack ?? String(err)}\n`);
    process.exitCode = 1;
  }
}
