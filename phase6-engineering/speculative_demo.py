"""
Module 17c -- Speculative decoding on toy distributions, checked exactly.

Decoding is limited by memory traffic, not arithmetic: a big model takes about as
long to score 5 positions in one forward pass as to score 1. Speculative decoding
(Leviathan et al., 2023; Chen et al., 2023) uses that slack. A small, fast DRAFT
model guesses several tokens. The big TARGET model checks all of them in ONE pass.
A rejection rule keeps or replaces each guess so that the final text is distributed
EXACTLY as if the target had sampled every token itself.

Our "models" are tiny Markov chains over 6 tokens: the next-token distribution
depends only on the last token. That is enough to test the claim that matters.

What you will see when you run it:

  1. One speculative round, traced: proposals, p and q, accept or reject.
  2. The one-step identity, to machine precision:
         P(output = x) = min(p, q) + (1 - sum min(p, q)) * residual(x) = p(x)
  3. An empirical check over 100,000 three-token sequences: the speculative
     sampler is as close to the target's true distribution as the target's own
     sampler is. Sampling from the draft alone is not.
  4. Tokens produced per target pass, as the draft gets worse, measured and exact.
"""
import numpy as np

V = 6            # vocabulary size
GAMMA = 4        # tokens the draft proposes per round


# ----------------------------------------------------------------------
# Two toy "language models": rows are next-token distributions
# ----------------------------------------------------------------------
def make_models(eps, seed=3):
    """Target P is a random peaked Markov chain. The draft Q is P blended with an
    unrelated chain R: eps = 0 is a perfect draft, eps = 1 an unrelated one."""
    rng = np.random.default_rng(seed)
    P = rng.dirichlet(np.full(V, 0.4), size=V)
    R = rng.dirichlet(np.full(V, 0.4), size=V)
    Q = (1 - eps) * P + eps * R
    return P, Q


# ----------------------------------------------------------------------
# The algorithm
# ----------------------------------------------------------------------
def draw(p, rng):
    """Sample an index from the distribution p (inverse CDF; much faster than rng.choice)."""
    return min(len(p) - 1, int(np.searchsorted(np.cumsum(p), rng.random(), side="right")))


def residual(p, q):
    """Where the target wants MORE probability than the draft gave: max(0, p - q), normalised."""
    r = np.maximum(0.0, p - q)
    total = r.sum()
    return r / total if total > 0 else p    # p == q: never reached by a rejection, any choice works


def speculative_round(P, Q, last, rng, gamma=GAMMA, trace=None):
    """One round = gamma cheap draft steps + ONE target pass. Returns the new tokens (1 to gamma + 1)."""
    # 1. the draft proposes gamma tokens, one after another
    proposals, ctx = [], last
    for _ in range(gamma):
        x = draw(Q[ctx], rng)
        proposals.append(x)
        ctx = x
    # 2. the target scores every position in one pass. Here that is a row lookup per
    #    position; in a real model it is one forward pass over gamma + 1 positions.
    out, ctx = [], last
    for x in proposals:
        p, q = P[ctx], Q[ctx]
        accept_prob = min(1.0, p[x] / q[x])
        accepted = rng.random() < accept_prob
        if trace is not None:
            trace.append({"ctx": ctx, "token": x, "p": p[x], "q": q[x], "accept_prob": accept_prob, "accepted": accepted})
        if not accepted:
            # 3. first rejection: replace it with a draw from the residual, and stop
            y = draw(residual(p, q), rng)
            if trace is not None:
                trace.append({"ctx": ctx, "resampled": y})
            out.append(y)
            return out
        out.append(x)
        ctx = x
    # 4. everything accepted: the target pass also scored the NEXT position, so take a free token
    bonus = draw(P[ctx], rng)
    if trace is not None:
        trace.append({"ctx": ctx, "bonus": bonus})
    out.append(bonus)
    return out


def generate_speculative(P, Q, start, n, rng):
    out, passes = [], 0
    while len(out) < n:
        out.extend(speculative_round(P, Q, out[-1] if out else start, rng))
        passes += 1
    return out[:n], passes


def generate_plain(P, start, n, rng):
    out = []
    for _ in range(n):
        out.append(draw(P[out[-1] if out else start], rng))
    return out


# ----------------------------------------------------------------------
# Exact results to compare against
# ----------------------------------------------------------------------
def one_step_output_distribution(p, q):
    """Distribution of the FIRST token a round emits, by the law of total probability."""
    accept = np.minimum(p, q)                      # q(x) * min(1, p(x) / q(x))
    return accept + (1.0 - accept.sum()) * residual(p, q)


def exact_joint(P, start, n):
    """The target's true distribution over all V**n sequences of length n."""
    joint = np.zeros((V,) * n)
    for idx in np.ndindex(*joint.shape):
        prob, ctx = 1.0, start
        for x in idx:
            prob *= P[ctx, x]
            ctx = x
        joint[idx] = prob
    return joint


def measured_tokens_per_pass(P, Q, rng, rounds=30_000, gamma=GAMMA):
    """Average length of a round, each started from a uniformly random last token."""
    return sum(len(speculative_round(P, Q, int(rng.integers(V)), rng, gamma)) for _ in range(rounds)) / rounds


def expected_tokens_per_pass(P, Q, gamma=GAMMA):
    """Exact, averaged uniformly over the last token. M[a, b] = min(P, Q) is the chance that b is
    proposed after a AND accepted, so (M^i 1)[a] is the chance the first i proposals all
    survive. Each survivor adds a token; the pass always yields one more (resample or bonus)."""
    M = np.minimum(P, Q)
    alive = np.ones(V)
    total = np.ones(V)
    for _ in range(gamma):
        alive = M @ alive
        total = total + alive
    return float(total.mean())


def acceptance_rate(P, Q):
    """alpha: the chance one proposal is accepted = sum_x min(p(x), q(x)), averaged over contexts."""
    return float(np.minimum(P, Q).sum(axis=1).mean())


def tv_distance(a, b):
    """Total variation distance: half the sum of absolute differences. 0 = identical."""
    return 0.5 * float(np.abs(a - b).sum())


def empirical_joint(sampler, n_samples, n=3):
    counts = np.zeros((V,) * n)
    for _ in range(n_samples):
        counts[tuple(sampler())] += 1
    return counts / n_samples


def banner(title):
    print("\n" + "=" * 74)
    print(title)
    print("=" * 74)


def main():
    P, Q = make_models(eps=0.4)

    banner("1. One speculative round, traced (draft blend eps = 0.4, 4 proposals)")
    rng = np.random.default_rng(16)
    trace = []
    out = speculative_round(P, Q, last=0, rng=rng, trace=trace)
    for t in trace:
        if "token" in t:
            print(f"  after token {t['ctx']}: draft proposes {t['token']}   q = {t['q']:.3f}  p = {t['p']:.3f}   "
                  f"accept with min(1, p/q) = {t['accept_prob']:.3f}  ->  {'ACCEPT' if t['accepted'] else 'REJECT'}")
        elif "resampled" in t:
            print(f"  rejected: draw a replacement from normalise(max(0, p - q))  ->  {t['resampled']}, and stop")
        else:
            print(f"  all accepted: the target pass already scored the next position  ->  bonus token {t['bonus']}")
    print(f"  this round produced {len(out)} tokens for ONE pass of the big model: {out}")

    banner("2. The one-step identity: the first emitted token is distributed exactly as p")
    p, q = P[0], Q[0]
    got = one_step_output_distribution(p, q)
    print(f"  target p     {np.array2string(p, precision=4, floatmode='fixed')}")
    print(f"  draft  q     {np.array2string(q, precision=4, floatmode='fixed')}")
    print(f"  speculative  {np.array2string(got, precision=4, floatmode='fixed')}")
    print(f"  largest difference from p: {np.abs(got - p).max():.1e}")
    print("  Why: a token is kept with probability q * min(1, p/q) = min(p, q). The missing mass,\n"
          "  max(0, p - q), is exactly what the residual hands back. min + max(0, p - q) = p.")

    banner("3. Empirical check: 100,000 sequences of 3 tokens against the exact distribution")
    N = 100_000
    truth = exact_joint(P, 0, 3)
    rng = np.random.default_rng(1)
    tv_spec = tv_distance(truth, empirical_joint(lambda: generate_speculative(P, Q, 0, 3, rng)[0], N))
    tv_plain = tv_distance(truth, empirical_joint(lambda: generate_plain(P, 0, 3, rng), N))
    tv_draft = tv_distance(truth, empirical_joint(lambda: generate_plain(Q, 0, 3, rng), N))
    print("  sampler                          total variation distance from the target")
    print(f"  target, one token at a time      {tv_plain:.4f}   <- pure sampling noise at this N")
    print(f"  speculative (draft + verify)     {tv_spec:.4f}   <- no larger than the noise: no bias")
    print(f"  draft alone                      {tv_draft:.4f}   <- a different distribution")

    banner("4. Tokens per target pass versus draft quality (4 proposals per round, so at most 5)")
    print("   eps   acceptance   tokens/pass   tokens/pass   (1 - a^5) / (1 - a)")
    print("         rate a       measured      exact         Leviathan et al., i.i.d. approximation")
    for eps in (0.0, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0):
        P, Q = make_models(eps)
        rng = np.random.default_rng(2)
        measured = measured_tokens_per_pass(P, Q, rng)
        a = acceptance_rate(P, Q)
        formula = GAMMA + 1 if a == 1 else (1 - a ** (GAMMA + 1)) / (1 - a)
        print(f"  {eps:>4.1f}{a:>11.3f}{measured:>13.2f}{expected_tokens_per_pass(P, Q):>14.2f}{formula:>14.2f}")
    print("  A perfect draft gives 5 tokens per pass. An unrelated one still gives more than 1,\n"
          "  and the output distribution is the target's either way: a bad draft costs speed,\n"
          "  never correctness. Whether it is a net WIN depends on what the draft costs to run.")


if __name__ == "__main__":
    main()
