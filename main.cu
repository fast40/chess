/**
 * @file main.cu
 * @brief GPU random chess simulator — plays thousands of games in parallel and reports statistics.
 *
 * This is the main entry point for the project. Each CUDA thread plays one complete
 * random chess game from the standard starting position, choosing moves uniformly at
 * random from the legal move list. Results are collected and statistics are reported.
 *
 * NOTE: [pedagogical] This file demonstrates several core CUDA concepts:
 *   - Kernel launch configuration (blocks × threads)
 *   - Per-thread random number generation with curand
 *   - Device memory allocation and host-device transfers
 *   - CUDA event-based timing
 */

#include "board.cuh"
#include "game.cuh"
#include <cstdio>
#include <curand_kernel.h>

// === Per-Game Statistics ===

/**
 * Statistics collected from a single game, written by each GPU thread.
 */
struct GameStats {
    GameResult result;
    uint16_t game_length; // total half-moves (plies)
};

// === CUDA Kernel ===

/**
 * Each thread plays one complete random chess game.
 *
 * NOTE: [pedagogical] The kernel is launched with enough threads to play all requested
 * games. Each thread gets a unique index (global_id) which seeds its random number
 * generator. The thread then runs an independent game loop: generate legal moves, pick
 * one at random, make the move, check for game over.
 */
__global__ void play_random_games(GameStats* results, int num_games, unsigned long long seed) {
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (global_id >= num_games) return;

    // NOTE: [pedagogical] curand_init sets up the random state for this thread. The three
    // arguments are: a global seed (same for all threads), a sequence number (unique per
    // thread, ensuring different random streams), and an offset (0 means start at the
    // beginning of the sequence). This guarantees each thread gets a statistically
    // independent random stream.
    curandState rng;
    curand_init(seed, global_id, 0, &rng);

    // === Game Loop ===
    BoardState board;
    initialize_board(board);

    int move_count = 0;
    constexpr int MAX_GAME_LENGTH = 1000;

    while (move_count < MAX_GAME_LENGTH) {
        MoveList legal;
        generate_legal_moves(board, legal);

        GameResult result = check_game_over(board, legal);
        if (result != ONGOING) {
            results[global_id] = {result, (uint16_t)move_count};
            return;
        }

        // Pick a random legal move uniformly
        // NOTE: [pedagogical] curand(&rng) returns a uniformly distributed unsigned int.
        // Taking modulo legal.count gives us an index into the legal move list. This has
        // a tiny bias when legal.count doesn't divide 2^32 evenly, but the bias is
        // negligible for our purposes (< 0.00001% for typical move counts of 20-40).
        int choice = curand(&rng) % legal.count;
        make_move(board, legal.moves[choice]);
        move_count++;
    }

    // Hit max game length without a result — treat as a draw
    results[global_id] = {DRAW_50_MOVE, (uint16_t)move_count};
}

// === Statistics Reporting ===

/**
 * Aggregate and print statistics from completed games.
 */
void report_statistics(const GameStats* results, int num_games, float elapsed_ms) {
    int white_wins = 0, black_wins = 0;
    int stalemates = 0, fifty_move = 0;
    long total_length = 0;
    int max_length = 0;

    for (int i = 0; i < num_games; i++) {
        switch (results[i].result) {
            case WHITE_WINS:     white_wins++; break;
            case BLACK_WINS:     black_wins++; break;
            case DRAW_STALEMATE: stalemates++; break;
            case DRAW_50_MOVE:   fifty_move++; break;
            default: break;
        }
        total_length += results[i].game_length;
        if (results[i].game_length > max_length) {
            max_length = results[i].game_length;
        }
    }

    int total_draws = stalemates + fifty_move;
    float avg_length = (float)total_length / num_games;

    printf("\n=== Results (%d games) ===\n", num_games);
    printf("  White wins:   %6d (%5.2f%%)\n", white_wins, 100.0f * white_wins / num_games);
    printf("  Black wins:   %6d (%5.2f%%)\n", black_wins, 100.0f * black_wins / num_games);
    printf("  Draws:        %6d (%5.2f%%)\n", total_draws, 100.0f * total_draws / num_games);
    printf("    Stalemate:  %6d (%5.2f%%)\n", stalemates, 100.0f * stalemates / num_games);
    printf("    50-move:    %6d (%5.2f%%)\n", fifty_move, 100.0f * fifty_move / num_games);
    printf("\n");
    printf("  Avg game length: %.1f half-moves (%.1f full moves)\n",
           avg_length, avg_length / 2.0f);
    printf("  Max game length: %d half-moves\n", max_length);

    printf("\n=== Performance ===\n");
    printf("  Total time:    %.2f ms\n", elapsed_ms);
    printf("  Games/second:  %.0f\n", 1000.0f * num_games / elapsed_ms);
    printf("  Avg time/game: %.3f ms\n", elapsed_ms / num_games);
    printf("\n");
}

// === Main ===

int main(int argc, char** argv) {
    int num_games = 10000;
    if (argc > 1) {
        num_games = atoi(argv[1]);
    }

    printf("Random Chess GPU Simulator\n");
    printf("Playing %d games on the GPU...\n", num_games);

    // === Device Memory Allocation ===
    GameStats* d_results;
    cudaMalloc(&d_results, num_games * sizeof(GameStats));

    // === Kernel Launch Configuration ===
    // NOTE: [pedagogical] We choose 256 threads per block, which is a common choice that
    // balances occupancy and register usage. The number of blocks is rounded up to cover
    // all games. Threads beyond num_games will early-return in the kernel.
    int threads_per_block = 256;
    int num_blocks = (num_games + threads_per_block - 1) / threads_per_block;

    // === Timing with CUDA Events ===
    // NOTE: [pedagogical] CUDA events provide accurate GPU timing. cudaEventRecord inserts
    // a timestamp into the GPU command stream, and cudaEventElapsedTime computes the
    // difference. This measures only GPU execution time, not host overhead.
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    unsigned long long seed = time(nullptr);

    cudaEventRecord(start);
    play_random_games<<<num_blocks, threads_per_block>>>(d_results, num_games, seed);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // Check for kernel errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    float elapsed_ms;
    cudaEventElapsedTime(&elapsed_ms, start, stop);

    // === Copy Results to Host ===
    GameStats* h_results = new GameStats[num_games];
    cudaMemcpy(h_results, d_results, num_games * sizeof(GameStats), cudaMemcpyDeviceToHost);

    report_statistics(h_results, num_games, elapsed_ms);

    // === Cleanup ===
    delete[] h_results;
    cudaFree(d_results);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
