# Random Chess on the GPU

A CUDA project that simulates thousands of random chess games in parallel on the GPU
and collects statistics about the outcomes.

## What This Project Is

The primary goal is to **learn CUDA programming** by building something nontrivial.
The vehicle for that learning is a massively parallel random chess simulator.

Each game works like this:
1. Start from the standard opening position.
2. Enumerate all legal moves for the side to move.
3. Pick one uniformly at random.
4. Repeat until the game ends (checkmate, stalemate, or draw).

By running many thousands of these games simultaneously on the GPU, we can collect
statistics about randomly played chess. For example:

- How often does white win? Black? Draw?
- What is the average game length?
- How often does en passant occur?
- How often does castling occur?
- How often is a game decided by checkmate vs. ending in stalemate?
- What is the material distribution at the end of a game?

These statistics are interesting on their own, but the real point is that implementing
a full legal move generator and game driver in CUDA exercises a wide range of GPU
programming concepts: thread divergence, shared memory, random number generation,
parallel reduction for aggregating statistics, and more.

## Progress

- [x] Project setup and README
