#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>

typedef struct {
    uint64_t state;
    uint64_t inc;

    float prev_norm;
} prng_state;

void prng_seed_r(prng_state* rng , uint64_t initstate, uint64_t initseq);
void prng_seed(uint64_t initstate, uint64_t initseq);
uint32_t prng_rand_r(prng_state* rng);
uint32_t prng_rand(void);
float prng_randf_r(prng_state* rng);
float prng_randf(void);

float prng_rand_norm_r(prng_state* rng);
float prng_random_norm(void);



int main(void) {
    for (uint32_t i = 0; i < 10; i++) {
        printf("%f\n", prng_randf());
    }
    return 0; 
}



// Default state
static prng_state s_prng_state = {
    0x853c49e6748fea9bULL, 0xda3e39cb94b95bdbULL
};

void prng_seed_r(prng_state* rng , uint64_t initstate, uint64_t initseq) {
    rng->state = 0U;
    rng->inc = (initseq << 1u) | 1u;
    prng_rand_r(rng);
    rng->state += initstate;
    prng_rand_r(rng);
}

void prng_seed(uint64_t initstate, uint64_t initseq) {
    prng_seed_r(&s_prng_state, initstate, initseq);
}

uint32_t prng_rand_r(prng_state* rng) {
    uint64_t oldstate = rng->state;
    // Standard LCG step
    rng->state = oldstate * 6364136223846793005ULL + rng->inc;
    // Output permutation (XSH RR)
    uint32_t xorshifted = ((oldstate >> 18u) ^ oldstate) >> 27u;
    uint32_t rot = oldstate >> 59u;
    return (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
}

uint32_t prng_rand(void) {
    return prng_rand_r(&s_prng_state);
}

float prng_randf_r(prng_state* rng) 
{
    return(float)prng_rand_r(rng) / (float)UINT32_MAX; 
}
float prng_randf(void) {
   return prng_randf_r(&s_prng_state);
}

float prng_rand_norm_r(prng_state* rng);
float prng_random_norm(void);

