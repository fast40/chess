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

## Board Representation

We use **12 bitboards** (one `uint64_t` per piece-type-color combination), which is
the standard representation in CPU chess engines like Stockfish. No known GPU chess engine
has used this representation before — the Zeta GPU engine uses quad bitboards (4 values
encoding piece info vertically). We chose 12-bitboard for clarity: "where are white's
knights?" is a single array read.

Each game state is ~128 bytes: 96 bytes for 12 piece bitboards, 24 bytes for occupancy
bitboards, and 8 bytes for metadata (side to move, castling rights, en passant, clocks).

## Progress

- [x] Project setup and README
- [x] Board state struct with 12-bitboard representation
- [x] Board initialization (standard starting position)
- [x] ASCII board printing for debugging
- [x] Tests for piece counts, occupancy consistency, and game state
