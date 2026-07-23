# Poster pitch

Five segments, each rehearsed as **stand-alone content** — not a script run
end to end. Which ones you use, and in what order, depends on who stops.
Structure follows the Harvard Catalyst poster-presentation framework.

Rehearse each aloud until you can start from any one of them cold.

---

## 0. Gauge first, then pick segments

Greet, smile, let them browse a beat. Then one open question — the answer tells
you which segments to run and at what depth.

- "How much do you work with hyperspectral?"
- "Have you run into quantum machine learning before, or is this a first?"
- "What brought you over — the imagery side or the quantum side?"

Two rough audiences and what each wants:

**Remote sensing / GEOINT** — they know the imagery better than the poster does.
Go straight to band selection and the uniform-beats-supervised result. They will
argue with it. Let them; that is the best outcome available.

**Curious about quantum, no background** — run the Hook, then the monitor, then
the Big Idea. Skip the band-selection detail entirely.

---

## 1. Hook — broad to specific, under a minute

> Tracking data centre construction from satellite is something a lot of people
> are trying to do right now. The imagery part is well understood — you can see
> clearing, foundations, steel, roofing.
>
> What I was handed was a narrower question: could a *quantum* classifier do
> that detection, and what would it actually cost to run? That second half turns
> out to be the interesting one, and the answer isn't where people look.

Then stop. Let them ask.

---

## 2. Background — the field, briefly

Name the work that got you here, and be specific about what each contributed:

- **Rybotycki, Gupta & Gawron (2025)** ran a quantum change-detection model on
  real IBM hardware and reported what it cost. That is the case study this
  builds on.
- **Krawec (FAS)** established the eight construction stages — the taxonomy in
  column one.
- **Möttönen et al. (2005)** is where the O(2ⁿ) state-preparation cost comes
  from, and that term is the whole argument.

End on why it matters: nobody had asked which variable actually drives that cost.

---

## 3. Big Idea — own it

This is the segment to sound most confident in. It is your claim, not a citation.

> My view is that the field is measuring the wrong variable. Every conversation
> about quantum cost is about qubit count and fidelity. For this workload the
> thing that actually sets cost is a **preprocessing decision made before you
> touch a circuit at all** — how many spectral bands you keep. Amplitude
> embedding ties bands to qubits to state-preparation depth, so band selection
> propagates straight through to what you pay.
>
> Accuracy plateaus at 16 to 32 bands. Cost keeps doubling past that. Going from
> 16 to 64 bands *loses* accuracy and costs five times the CNOTs.

Expect pushback. Take it — that is how the idea gets sharper.

---

## 4. Methods, challenges, and the surprises

Harvard's framing: what limits did you expect, what surprised you, how did you
handle it. You have three genuine surprises. Use them — they are the most
memorable content you have.

**The limitation I expected:** no public hyperspectral scene exists over a
hyperscale data centre. I used Pavia University as a proxy because its material
classes — painted metal, asphalt, bitumen, bare soil — are what such a campus is
built from. Say this *before* anyone challenges it.

**Surprise one — the clever method lost.** I expected supervised
mutual-information band selection to beat evenly spaced bands. It didn't, at any
band count above 16. Plain MI also picked eight adjacent blue bands that were
near-duplicates of each other. Handling it: I wrote a decorrelated variant that
rejects redundant neighbours — and uniform spacing *still* won. Which is good
news operationally, because a fixed multispectral sensor gives you even spacing
for free.

**Surprise two — a wrong answer that looked right.** Comparing PennyLane and
Qiskit, my first run returned −0.033 against a true value of −0.167. No error
was raised; the number sat well inside the plausible range. The cause was an
endianness mismatch — I'd remapped the qubits *and* reversed the amplitude
vector, which silently applies an X to every qubit. Handling it: I checked the
analytic values against each other before trusting anything sampled. That single
check is what caught it.

**Surprise three — the source paper's arithmetic.** Their reported per-sample
cost inverts their own spreadsheet's units, which compute samples per minute.
Recomputing from their own totals gives 28 seconds and about $45 a sample, not
2 minutes 13 seconds.

---

## 5. Results and future directions — keep short, then ask

> Where it leaves me: band selection is the first-order cost variable, and the
> quantum model's margin over logistic regression is small and gone by 64 bands.
>
> Next is reproducing the original authors' code on ONERA image pairs, which
> closes the bi-temporal change-detection gap at the same time. Then a
> foundation-model baseline — Scale-MAE rather than logistic regression — and
> eventually a hardware run.

Then hand it to them. This is the segment where you get value back:

- "You work with these scenes more than I do — would you expect uniform spacing
  to hold on a different sensor?"
- "Is there a scene type where you'd expect this to fall over?"

---

## The monitor — never run it silently

The demo is the most memorable thing here and the most distortable. The bare
reading is "quantum is random and broken," which is wrong and sticky.

Every time it loops, the line goes with it:

> The estimate is unbiased — it converges on the exact value. It isn't
> unreliable. Certainty is just purchasable: four times the shots buys twice the
> precision. That's why inference, not training, is the recurring cost.

**Analogy for non-experts:** it's an exit poll. Ask a hundred people and you get
a number; ask ten thousand and you get a tighter one. The circuit charges you
per person asked, every single time you want an answer.

**Analogy for amplitude embedding:** sixteen bands fit in four qubits because
you're storing them as amplitudes rather than slots. The catch is that *loading*
them costs exponentially — and a simulator does it in one step, so the bill is
invisible until you touch hardware.

---

## Answers to have ready

**"So should we be using quantum for this?"**
> Not on these numbers — the margin over logistic regression is gone by 64
> bands. What this establishes is the cost structure, which is worth knowing
> before anyone invests.

**"Did you reproduce the original paper?"**
> No. I implemented the architecture from the paper's description rather than
> running their repo, and went straight at the band-selection question.
> Reproducing it on ONERA is the next thing, and it closes the bi-temporal gap
> too.

**"Objective two just found the frameworks agree — wasn't that a dead end?"**
> It rules out the framework as a variable. Once you know the simulators agree
> exactly, every difference you see is shot noise or your own convention error.
> That's what makes the cost argument hold, because shots are what you pay for.

**"Why Pavia and not a real data centre?"**
> No public hyperspectral scene exists over one. I used a benchmark I could
> validate against. The data centre framing is the motivation, not the result.

---

## Logistics

Stay at the poster the whole session. Keep water. Face the person, not the
poster — use it as a visual aid, pointing to figures as you go, and check in
periodically rather than talking through. Watch for waning interest and offer to
go deeper or wrap. Eye-contact norms vary; stay alert to discomfort.

Before the session: record a 3–5 minute run-through, watch it back, and get one
colleague to listen live.
