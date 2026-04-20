#ifndef PSGRDN_H
#define PSGRDN_H

#include <stdint.h>
#include <math.h>

#define PI 3.14159265358979

typedef struct {
    uint64_t state;
    uint64_t inc;
    float prev_norm;
} prng_state;

void     prng_seed_r(prng_state* rng, uint64_t initstate, uint64_t initseq);
void     prng_seed(uint64_t initstate, uint64_t initseq);
uint32_t prng_rand_r(prng_state* rng);
uint32_t prng_rand(void);
float    prng_randf_r(prng_state* rng);
float    prng_randf(void);
float    prng_rand_norm_r(prng_state* rng);
float    prng_rand_norm(void);

#endif // PSGRDN_H
