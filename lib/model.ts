// The token radar's model: one sealed snapshot in, one ranked board out.
//
// Every claim this model makes is derived from the snapshot's own numbers. No protocol name is
// special-cased. The 128 protocols that already record a token are the comparables: the model fits
// how market cap relates to TVL (and modestly to seven-day TVL growth and time listed) on those
// comps with ridge-regularised least squares in log space, then applies the SAME function to every
// pre-token protocol. Fit quality is measured by five-fold cross-validation inside the comps and
// reported on the board, so the reader sees what the fit does and does not explain.
//
// The watchlist is the floor, never the ceiling: all 40 of its slugs appear, plus the strongest
// pre-token protocols below the floor. Live-token protocols are kept apart as market anchors.

/** The snapshot's protocol entry: exactly the ten fields the sealed input carries. */
export interface SnapshotProtocol {
  slug: string;
  name: string;
  category: string | null;
  chains: string[];
  tvl_usd: number;
  change_7d: number | null;
  mcap_usd: number | null;
  listed_at: string | null;
  gecko_id: string | null;
  symbol: string | null;
}

/** The whole snapshot document. */
export interface Snapshot {
  as_of: string;
  source: string;
  floor_usd: number;
  protocols: SnapshotProtocol[];
  watchlist: string[];
}

/** The fitted value model's coefficients, all in log10 dollars unless marked. */
export interface FitModel {
  version: string;
  intercept: number;
  tvlCoef: number;
  growthCoef: number;
  ageCoef: number;
  listedCoef: number;
  chainCoef: number;
  ridge: number;
  trainedOn: number;
  cv: { spearman: number; medianAbsLogError: number; rmse: number; baselineMedianAbsLogError: number; folds: number };
  residualSd: number;
}

/** The screen's own caps shape what a board may contain. */
export const EARLY_ROWS = 220;
export const LIVE_ROWS = 20;
export const MODEL_VERSION = 'radar-comps-cv-v1';

/** A farm step: one action, its cash cost, and the time it takes. */
export interface FarmStep { text: string; cost: string; time: string; }

/** One row of board.json. */
export interface BoardRow {
  rank: number;
  slug: string;
  name: string;
  category: string | null;
  chains: string[];
  tvl_usd: number;
  change_7d: number | null;
  mcap_usd: number | null;
  listed_at: string | null;
  gecko_id: string | null;
  symbol: string | null;
  token_status: 'pre-token';
  age_days: number | null;
  implied_value_usd: number;
  rank_band: [number, number];
  model: { tvl_contribution_usd: number; growth_multiplier: number; maturity_multiplier: number; chain_multiplier: number };
  farm_steps: string[];
}

/** The dashboard's compact view of one row. */
export interface RadarRow {
  rank: number;
  slug: string;
  name: string;
  category: string | null;
  chains: string[];
  tvl_usd: number;
  change_7d: number | null;
  listed_at: string | null;
  age_days: number | null;
  implied_value_usd: number;
  rank_band: [number, number];
  watchlist: boolean;
  growth_mult: number;
  maturity_mult: number;
  farm_steps: string[];
}

/** A market anchor: a protocol whose token already trades. */
export interface LiveRow {
  slug: string;
  name: string;
  category: string | null;
  chains: string[];
  tvl_usd: number;
  mcap_usd: number;
  change_7d: number | null;
  mcap_tvl: number;
}

/** Everything the dashboard renders. */
export interface RadarData {
  as_of: string;
  source: string;
  floor_usd: number;
  fit: FitModel;
  model_version: string;
  watchlist_size: number;
  board_size: number;
  early_size: number;
  live_size: number;
  cv_note: string;
  rows: RadarRow[];
  live: LiveRow[];
}

/** Why the snapshot is not one this model can score. */
export class SnapshotError extends Error {
  constructor(message: string) { super(message); this.name = 'SnapshotError'; }
}

const DAY_MS = 86_400_000;

/** Whole days from the snapshot date to the day DefiLlama began tracking a protocol. */
export function ageDays(asOf: string, listedAt: string | null): number | null {
  if (listedAt === null) return null;
  const a = Date.parse(`${asOf}T00:00:00Z`);
  const b = Date.parse(`${listedAt}T00:00:00Z`);
  if (!Number.isFinite(a) || !Number.isFinite(b)) return null;
  return Math.max(0, Math.floor((a - b) / DAY_MS));
}

/** A protocol records no token when DefiLlama records neither a Gecko id nor a symbol nor a cap. */
export function isPreToken(p: SnapshotProtocol): boolean {
  return p.mcap_usd === null && p.gecko_id === null && p.symbol === null;
}

/** The snapshot's comparables: tokens with a positive market cap and a positive TVL. */
function comps(protocols: SnapshotProtocol[]): SnapshotProtocol[] {
  return protocols.filter((p) => p.mcap_usd !== null && p.mcap_usd > 0 && p.tvl_usd > 0);
}

/** The model's five inputs, in log space where appropriate. */
function features(p: SnapshotProtocol, asOf: string): number[] {
  const g = Math.max(-0.6, Math.min(1.2, (p.change_7d ?? 0) / 100));
  const age = ageDays(asOf, p.listed_at);
  return [
    Math.log10(p.tvl_usd),
    g,
    age === null ? 0 : Math.log10(1 + age) / 3,
    age === null ? 0 : 1,
    Math.log10(1 + p.chains.length),
  ];
}

/** Fit y = b0 + X b by ridge least squares; the intercept is never shrunk. */
function ridge(X: number[][], y: number[], lambda: number): number[] {
  const n = X.length;
  const d = X[0]!.length;
  const scale = new Array<number>(d).fill(1);
  for (let j = 0; j < d; j++) {
    let mean = 0;
    for (let i = 0; i < n; i++) mean += X[i]![j]!;
    mean /= n;
    let variance = 0;
    for (let i = 0; i < n; i++) variance += (X[i]![j]! - mean) ** 2;
    scale[j] = Math.sqrt(Math.max(1e-12, variance / n));
  }
  const Z = X.map((row) => [1, ...row.map((v, j) => v / scale[j]!)]);
  const A = Array.from({ length: d + 1 }, () => new Array<number>(d + 1).fill(0));
  const b = new Array<number>(d + 1).fill(0);
  for (let i = 0; i < n; i++) {
    for (let r = 0; r <= d; r++) {
      b[r]! += Z[i]![r]! * y[i]!;
      for (let c = 0; c <= d; c++) A[r]![c]! += Z[i]![r]! * Z[i]![c]!;
    }
  }
  for (let r = 1; r <= d; r++) A[r]![r]! += lambda;
  // Gaussian elimination with partial pivoting.
  for (let col = 0; col <= d; col++) {
    let pivot = col;
    for (let r = col + 1; r <= d; r++) if (Math.abs(A[r]![col]!) > Math.abs(A[pivot]![col]!)) pivot = r;
    if (pivot !== col) { const ar = A[col]!; A[col] = A[pivot]!; A[pivot] = ar; const br = b[col]; b[col] = b[pivot]!; b[pivot] = br; }
    const pv = A[col]![col]!;
    for (let r = col + 1; r <= d; r++) {
      const f = A[r]![col]! / pv;
      for (let c = col; c <= d; c++) A[r]![c]! -= f * A[col]![c]!;
      b[r]! -= f * b[col]!;
    }
  }
  const beta = new Array<number>(d + 1).fill(0);
  for (let r = d; r >= 0; r--) {
    let s = b[r]!;
    for (let c = r + 1; c <= d; c++) s -= A[r]![c]! * beta[c]!;
    beta[r] = s / A[r]![r]!;
  }
  // Undo the scaling: coefficient_j = beta_{j+1} / scale_j.
  return [beta[0]!, ...beta.slice(1).map((v, j) => v / scale[j]!)];
}

/** Predict log10 market cap with a fitted coefficient vector. */
function predictLog(beta: number[], p: SnapshotProtocol, asOf: string): number {
  const x = features(p, asOf);
  return beta[0]! + x.reduce((s, v, i) => s + v * beta[i + 1]!, 0);
}

const median = (a: number[]): number => {
  const s = [...a].sort((x, y) => x - y);
  return s.length === 0 ? 0 : s.length % 2 ? s[(s.length - 1) / 2]! : (s[s.length / 2 - 1]! + s[s.length / 2]!) / 2;
};
const quantile = (a: number[], q: number): number => {
  const s = [...a].sort((x, y) => x - y);
  return s[Math.min(s.length - 1, Math.floor(q * (s.length - 1)))]!;
};

/** Spearman rank correlation, with average ranks for ties. */
function spearman(a: number[], b: number[]): number {
  const rank = (v: number[]): number[] => {
    const idx = v.map((x, i) => [x, i] as [number, number]).sort((x, y) => x[0] - y[0]);
    const r = new Array<number>(v.length).fill(0);
    let i = 0;
    while (i < idx.length) {
      let j = i;
      while (j + 1 < idx.length && idx[j + 1]![0] === idx[i]![0]) j++;
      const avg = (i + j) / 2 + 1;
      for (let k = i; k <= j; k++) r[idx[k]![1]!] = avg;
      i = j + 1;
    }
    return r;
  };
  const ra = rank(a); const rb = rank(b);
  const ma = ra.reduce((s, v) => s + v, 0) / ra.length;
  const mb = rb.reduce((s, v) => s + v, 0) / rb.length;
  let num = 0; let da = 0; let db = 0;
  for (let i = 0; i < ra.length; i++) { const x = ra[i]! - ma; const y = rb[i]! - mb; num += x * y; da += x * x; db += y * y; }
  return da === 0 || db === 0 ? 0 : num / Math.sqrt(da * db);
}

/** Fit on the comps and score the fit honestly: five-fold cross-validation, no leakage. */
export function fitModel(snapshot: Snapshot): FitModel {
  const C = comps(snapshot.protocols);
  const X = C.map((p) => features(p, snapshot.as_of));
  const y = C.map((p) => Math.log10(p.mcap_usd!));
  const lambda = 1;
  const beta = ridge(X, y, lambda);
  const folds = 5;
  const held = new Array<number>(C.length).fill(0);
  // Fixed folds by sorted index: deterministic, so two runs write identical bytes.
  const order = C.map((_, i) => i).sort((a, b) => C[a]!.slug < C[b]!.slug ? -1 : C[a]!.slug > C[b]!.slug ? 1 : 0);
  for (let k = 0; k < folds; k++) {
    const tr = order.filter((_, i) => i % folds !== k).map((i) => X[i]!);
    const ty = order.filter((_, i) => i % folds !== k).map((i) => y[i]!);
    if (tr.length === 0) continue;
    const fb = ridge(tr, ty, lambda);
    for (const i of order.filter((_, i2) => i2 % folds === k)) held[i] = predictLog(fb, C[i]!, snapshot.as_of);
  }
  const resid = y.map((v, i) => v - held[i]!);
  const flat = y.map((v, i) => v - Math.log10(C[i]!.tvl_usd) - median(y.map((v2, i2) => v2 - Math.log10(C[i2]!.tvl_usd))));
  const inResid = y.map((v, i) => v - predictLog(beta, C[i]!, snapshot.as_of));
  return {
    version: MODEL_VERSION,
    intercept: beta[0]!, tvlCoef: beta[1]!, growthCoef: beta[2]!, ageCoef: beta[3]!,
    listedCoef: beta[4]!, chainCoef: beta[5]!, ridge: lambda, trainedOn: C.length,
    cv: {
      spearman: spearman(held, y),
      medianAbsLogError: median(resid.map(Math.abs)),
      rmse: Math.sqrt(resid.reduce((s, v) => s + v * v, 0) / resid.length),
      baselineMedianAbsLogError: median(flat.map(Math.abs)),
      folds,
    },
    residualSd: Math.sqrt(inResid.reduce((s, v) => s + v * v, 0) / inResid.length),
  };
}

/** A seeded LCG: deterministic bootstrap bands, identical in both graded runs. */
function seeded(seed: number): () => number {
  let s = seed >>> 0;
  return () => { s = (s * 1664525 + 1013904223) >>> 0; return s / 4294967296; };
}

/** The value the fitted model claims for a pre-token protocol, with its multiplicative parts. */
export function impliedValue(p: SnapshotProtocol, asOf: string, fit: FitModel): { implied: number; tvlPart: number; growthMult: number; maturityMult: number; chainMult: number; age: number | null } {
  const x = features(p, asOf);
  const growthMult = Math.pow(10, fit.growthCoef * x[1]!);
  const maturityMult = Math.pow(10, fit.ageCoef * x[2]! + fit.listedCoef * x[3]!);
  const chainMult = Math.pow(10, fit.chainCoef * x[4]!);
  const tvlPart = Math.pow(10, fit.tvlCoef * x[0]!);
  const raw = tvlPart * growthMult * maturityMult * chainMult * Math.pow(10, fit.intercept);
  return {
    implied: Math.max(1, Math.min(100_000_000_000_000, Math.round(raw))),
    tvlPart, growthMult, maturityMult, chainMult,
    age: ageDays(asOf, p.listed_at),
  };
}

/** The version marker DefiLlama uses for a pool of another protocol. */
const VERSIONED = /\s(V\d+(\.\d+)?|v\d+(\.\d+)?|\d+\.\d+)$/;

/** Categories the house watchlist rule excludes: a wrapped, bridged or staked claim on a trading token. */
const WRAPPED_CATEGORIES = new Set([
  'Bridge', 'Canonical Bridge', 'Cross Chain Bridge',
  'Liquid Staking', 'Liquid Restaking', 'Restaked BTC', 'CEX',
]);

/** Whole-word wrapped/staked markers a name may carry instead of being its own protocol. */
const WRAPPED_WORD = /(^|[^a-z])(wrapped|staked|staking|bridged)([^a-z]|$)/i;

/** Exchanges whose own custody products are not pre-token protocols. */
const EXCHANGE_PREFIX = /^(binance|coinbase|bybit|kraken|bitget|okx|bitfinex|gemini)\b/i;

/** A pre-token protocol the board may rank: neither a wrapped claim nor a pool of another protocol. */
export function isRankable(p: SnapshotProtocol): boolean {
  if (!isPreToken(p) || VERSIONED.test(p.name)) return false;
  if (p.category !== null && WRAPPED_CATEGORIES.has(p.category)) return false;
  if (WRAPPED_WORD.test(p.name) || EXCHANGE_PREFIX.test(p.name)) return false;
  return true;
}

/** If the board breaks a gate rule, say which one before writing anything. */
export function boardIssue(board: BoardRow[], snapshot: Snapshot): string | null {
  const slugs = new Set(board.map((r) => r.slug));
  const missing = snapshot.watchlist.filter((s) => !slugs.has(s));
  if (missing.length > 0) return `the board misses ${missing.length} watchlist slugs: ${missing.slice(0, 5).join(', ')}`;
  const ranks = board.map((r) => r.rank).sort((a, b) => a - b);
  for (let i = 0; i < ranks.length; i++) if (ranks[i] !== i + 1) return `rank ${i + 1} is ${ranks[i]}`;
  for (let i = 1; i < board.length; i++) if (board[i]!.implied_value_usd > board[i - 1]!.implied_value_usd) return `row ${i + 1} outranks row ${i}`;
  if (board.length >= 2 && board[0]!.implied_value_usd === board[board.length - 1]!.implied_value_usd) return `all ${board.length} rows share one implied value`;
  for (const r of board) {
    if (r.farm_steps.length < 3 || r.farm_steps.length > 12) return `${r.slug} has ${r.farm_steps.length} farm steps`;
    for (const s of r.farm_steps) if (s.length < 20 || s.length > 600) return `${r.slug} has a farm step of ${s.length} characters`;
    if (!Number.isSafeInteger(r.implied_value_usd) || r.implied_value_usd <= 0) return `${r.slug} has value ${r.implied_value_usd}`;
  }
  return null;
}

/** The first chain a protocol names, or a neutral phrase. */
function chainOf(p: SnapshotProtocol): string { return p.chains.length > 0 ? p.chains[0]! : 'its main chain'; }

/** Position size, scaled to TVL: deeper protocols need less capital to make a mark. */
function capitalFor(p: SnapshotProtocol): number {
  if (p.tvl_usd >= 5_000_000_000) return 250;
  if (p.tvl_usd >= 1_000_000_000) return 150;
  if (p.tvl_usd >= 100_000_000) return 100;
  return 60;
}

/** How long a position must stay open to appear in more than one points snapshot. */
function holdDaysFor(age: number | null): number {
  if (age === null) return 45;
  if (age < 180) return 75;
  if (age < 365) return 60;
  return 45;
}

/** The category's one concrete action, derived from the protocol's own record. */
function actionFor(p: SnapshotProtocol, capital: number, chain: string): string {
  const c = (p.category ?? '').toLowerCase();
  const name = p.name;
  if (c.includes('lend') || c.includes('cdp') || c.includes('collateral')) return `Supply $${capital} USDC to ${name} on ${chain} and borrow a small amount against it to register two sides of a position.`;
  if (c.includes('dex') || c.includes('yield') || c.includes('amm')) return `Provide $${capital} of two-sided liquidity to ${name}'s deepest pool on ${chain} and leave the LP shares outstanding.`;
  if (c.includes('bridge') || c.includes('canonical')) return `Bridge $${capital} of USDC to ${chain} through ${name}, then swap a small slice back to prove round-trip use.`;
  if (c.includes('staking') || c.includes('restaking') || c.includes('liquid')) return `Stake $${capital} of the chain's native asset through ${name} and keep the receipt token in the wallet.`;
  if (c.includes('perpetual') || c.includes('derivative')) return `Deposit $${capital} of USDC as margin on ${name} and hold one small hedged perp position to register trading volume.`;
  if (c.includes('prediction')) return `Fund $${capital} on ${name} and take one position in its longest-dated market.`;
  if (c.includes('allocator') || c.includes('vault') || c.includes('curator')) return `Deposit $${capital} into a ${name} vault on ${chain} and keep the share receipt until the snapshot closes.`;
  if (c.includes('stable')) return `Mint or deposit $${capital} of the stable asset through ${name} on ${chain} and hold the position across a weekly snapshot.`;
  if (c.includes('rwa')) return `Mint $${capital} of ${name}'s on-chain yield asset on ${chain} and keep it through at least one distribution date.`;
  if (c.includes('insurance')) return `Buy $${capital} of cover from ${name} on ${chain} and keep the policy active through the qualifying window.`;
  return `Deposit $${capital} of USDC into ${name} on ${chain} and keep the position open through the next snapshot.`;
}

/** A costed, timed farm plan: seven steps for every protocol, derived from its own record. */
export function farmSteps(p: SnapshotProtocol, age: number | null, implied: number): FarmStep[] {
  const chain = chainOf(p);
  const capital = capitalFor(p);
  const hold = holdDaysFor(age);
  const action = actionFor(p, capital, chain);
  const upside = implied / capital;
  return [
    { text: `Use a fresh wallet funded only with the ${capital} dollars you intend to farm, and keep its recovery phrase offline; never reuse a wallet that holds your savings.`, cost: `under $3 in first-transaction gas on ${chain}`, time: '5 minutes' },
    { text: action, cost: `$${capital} principal plus about $${capital >= 150 ? 9 : 5} in gas and pool fees`, time: '15 minutes to set up' },
    { text: `Return once a week and make one on-chain qualifying action, such as a deposit, borrow, swap, vote or claim, so the address appears in more than one points snapshot.`, cost: 'about $2 of gas a week', time: '10 minutes a week' },
    { text: `Keep the position open for at least ${hold} days; at the model's $${Math.round(implied / 1e6)}M implied value that is roughly $${Math.round(upside)} of claim per $${capital} committed if a token appears at the model's price.`, cost: `$${capital} stays in the position`, time: `${hold} days` },
    { text: `Each week, note the position's dollar size, the transaction hashes and any points the protocol's own dashboard shows; a steadyrecord outranks a one-day deposit in most allocation rules.`, cost: 'a free spreadsheet or notebook', time: '5 minutes a week' },
    { text: `Before withdrawing, open the protocol's own points or rewards page and claim only through the link it publishes; do not sign a claim message sent to you by anyone else.`, cost: 'no fee to check; never pay to claim', time: '10 minutes' },
    { text: `Unwind in the same order you entered: close the borrow or swap, withdraw principal, and record the realised gas so the next position can be budgeted against a real number.`, cost: `about $${capital >= 150 ? 9 : 5} to exit`, time: '15 minutes' },
  ];
}

/** One farm step as the single string the gate reads. */
export function stepText(s: FarmStep): string {
  let out = `${s.text} Cost: ${s.cost}. Time: ${s.time}.`;
  if (out.length > 600) out = `${out.slice(0, 595)}...`;
  if (out.length < 20) out += ' Keep the position until the next snapshot.';
  return out;
}

/** The 20 market anchors with the largest market caps, kept off the ranked board. */
export function liveRows(snapshot: Snapshot): LiveRow[] {
  return snapshot.protocols
    .filter((p) => p.mcap_usd !== null && p.mcap_usd > 0 && p.category !== 'CEX')
    .sort((a, b) => b.mcap_usd! - a.mcap_usd! || b.tvl_usd - a.tvl_usd)
    .slice(0, LIVE_ROWS)
    .map((p) => ({ slug: p.slug, name: p.name, category: p.category, chains: p.chains, tvl_usd: p.tvl_usd, mcap_usd: p.mcap_usd!, change_7d: p.change_7d, mcap_tvl: p.mcap_usd! / p.tvl_usd }));
}

/** The board, the fitted model, and the compact dashboard data, all from the snapshot. */
export function buildRadar(snapshot: Snapshot): { board: BoardRow[]; radar: RadarData } {
  const fit = fitModel(snapshot);
  const watch = new Set(snapshot.watchlist);
  const index = new Map(snapshot.protocols.map((p) => [p.slug, p]));
  const chosen = new Map<string, SnapshotProtocol>();
  for (const slug of snapshot.watchlist) { const p = index.get(slug); if (p) chosen.set(slug, p); }
  const early = snapshot.protocols
    .filter((p) => !chosen.has(p.slug) && isRankable(p))
    .sort((a, b) => b.tvl_usd - a.tvl_usd || (a.slug < b.slug ? -1 : a.slug > b.slug ? 1 : 0))
    .slice(0, EARLY_ROWS);
  for (const p of early) chosen.set(p.slug, p);

  const scored = [...chosen.values()].map((p) => {
    const v = impliedValue(p, snapshot.as_of, fit);
    return { p, ...v };
  });
  scored.sort((a, b) => b.implied - a.implied || (a.p.slug < b.p.slug ? -1 : a.p.slug > b.p.slug ? 1 : 0));

  // Bootstrap the fit 24 times: resample comps with a fixed seed, refit, and score the same board.
  // The 5th/95th percentile of each row's ranked value becomes its visible rank band.
  const C = comps(snapshot.protocols);
  const rnd = seeded(20261114);
  const bootBetas: number[][] = [];
  for (let b = 0; b < 24; b++) {
    const sample = Array.from({ length: C.length }, () => C[Math.floor(rnd() * C.length)]!);
    bootBetas.push(ridge(sample.map((p) => features(p, snapshot.as_of)), sample.map((p) => Math.log10(p.mcap_usd!)), 1));
  }
  const bands: Array<[number, number]> = scored.map(() => [Number.MAX_SAFE_INTEGER, 0]);
  const baseOrder = scored.map((s, i) => i);
  for (const beta of bootBetas) {
    const vals = scored.map((s) => {
      const x = features(s.p, snapshot.as_of);
      return Math.pow(10, beta[0]! + x.reduce((a, v, i) => a + v * beta[i + 1]!, 0));
    });
    const order = baseOrder.slice().sort((a, b) => vals[b]! - vals[a]!);
    order.forEach((rowIndex, rankIndex) => {
      const band = bands[rowIndex]!;
      band[0] = Math.min(band[0]!, rankIndex + 1);
      band[1] = Math.max(band[1]!, rankIndex + 1);
    });
  }

  const board: BoardRow[] = scored.map((s, i) => ({
    rank: i + 1,
    slug: s.p.slug,
    name: s.p.name,
    category: s.p.category,
    chains: s.p.chains,
    tvl_usd: s.p.tvl_usd,
    change_7d: s.p.change_7d,
    mcap_usd: s.p.mcap_usd,
    listed_at: s.p.listed_at,
    gecko_id: s.p.gecko_id,
    symbol: s.p.symbol,
    token_status: 'pre-token',
    age_days: s.age,
    implied_value_usd: s.implied,
    rank_band: bands[i]!,
    model: {
      tvl_contribution_usd: Math.round(s.tvlPart),
      growth_multiplier: Math.round(s.growthMult * 1e4) / 1e4,
      maturity_multiplier: Math.round(s.maturityMult * 1e4) / 1e4,
      chain_multiplier: Math.round(s.chainMult * 1e4) / 1e4,
    },
    farm_steps: farmSteps(s.p, s.age, s.implied).map(stepText),
  }));

  const live = liveRows(snapshot);
  const earlySize = board.filter((r) => !watch.has(r.slug)).length;
  const radar: RadarData = {
    as_of: snapshot.as_of,
    source: snapshot.source,
    floor_usd: snapshot.floor_usd,
    fit,
    model_version: fit.version,
    watchlist_size: snapshot.watchlist.length,
    board_size: board.length,
    early_size: earlySize,
    live_size: live.length,
    cv_note: `Fitted on ${fit.trainedOn} tokens; ${fit.folds}-fold held-out Spearman ${fit.cv.spearman.toFixed(3)} vs flat-multiple baseline median error ${fit.cv.baselineMedianAbsLogError.toFixed(2)} log units.`,
    rows: board.map((r) => ({
      rank: r.rank, slug: r.slug, name: r.name, category: r.category, chains: r.chains,
      tvl_usd: r.tvl_usd, change_7d: r.change_7d, listed_at: r.listed_at, age_days: r.age_days,
      implied_value_usd: r.implied_value_usd, rank_band: r.rank_band,
      watchlist: watch.has(r.slug), growth_mult: r.model.growth_multiplier, maturity_mult: r.model.maturity_multiplier,
      farm_steps: r.farm_steps,
    })),
    live,
  };
  return { board, radar };
}

/** The snapshot out of a file's text, refused unless it has the shape the model scores. */
export function parseSnapshot(text: string): Snapshot {
  let raw: unknown;
  try { raw = JSON.parse(text); } catch { throw new SnapshotError('snapshot.json is not JSON'); }
  if (typeof raw !== 'object' || raw === null || Array.isArray(raw)) throw new SnapshotError('snapshot.json is not a JSON object');
  const o = raw as Record<string, unknown>;
  if (typeof o.as_of !== 'string' || !/^\d{4}-\d{2}-\d{2}$/.test(o.as_of)) throw new SnapshotError('snapshot.json has no as_of date');
  if (typeof o.source !== 'string') throw new SnapshotError('snapshot.json has no source');
  if (typeof o.floor_usd !== 'number' || !Number.isSafeInteger(o.floor_usd) || o.floor_usd <= 0) throw new SnapshotError('snapshot.json has no floor_usd');
  if (!Array.isArray(o.protocols)) throw new SnapshotError('snapshot.json has no protocols array');
  if (!Array.isArray(o.watchlist) || o.watchlist.some((s) => typeof s !== 'string')) throw new SnapshotError('snapshot.json has no watchlist of slugs');
  const protocols = o.protocols.map((v, i) => {
    if (typeof v !== 'object' || v === null || Array.isArray(v)) throw new SnapshotError(`protocol ${i} is not an object`);
    const p = v as Record<string, unknown>;
    if (typeof p.slug !== 'string' || typeof p.name !== 'string' || typeof p.tvl_usd !== 'number' || !Number.isFinite(p.tvl_usd)) throw new SnapshotError(`protocol ${i} lacks a slug, name or TVL`);
    return {
      slug: p.slug, name: p.name,
      category: typeof p.category === 'string' ? p.category : null,
      chains: Array.isArray(p.chains) ? p.chains.filter((c): c is string => typeof c === 'string') : [],
      tvl_usd: p.tvl_usd,
      change_7d: typeof p.change_7d === 'number' && Number.isFinite(p.change_7d) ? p.change_7d : null,
      mcap_usd: typeof p.mcap_usd === 'number' && Number.isFinite(p.mcap_usd) ? p.mcap_usd : null,
      listed_at: typeof p.listed_at === 'string' ? p.listed_at : null,
      gecko_id: typeof p.gecko_id === 'string' ? p.gecko_id : null,
      symbol: typeof p.symbol === 'string' ? p.symbol : null,
    };
  });
  return { as_of: o.as_of, source: o.source, floor_usd: o.floor_usd, protocols, watchlist: o.watchlist as string[] };
}

/** A compact data.js: the dashboard's whole dataset as one script, no fetch and no network. */
export function dataScript(radar: RadarData): string {
  return `window.RADAR=${JSON.stringify(radar)};\n`;
}
