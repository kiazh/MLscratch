#include <stdio.h>
#include "arena.h"
#include "psgrdn.h"
#include <math.h>
#include <stdbool.h>
#include <stdint.h> 
#include <string.h>
#include "arena.c"
#include "psgrdn.c"
#include "base.h"

typedef struct {
    uint32_t rows, cols;
    float* data; 
} matrix;

matrix* mat_create(mem_arena* arena, uint32_t rows, uint32_t cols); 
void mat_clear(matrix* mat);
void mat_copy(matrix* dst, matrix* src);
void mat_fill(matrix* mat, f32 x);
void mat_scale(matrix* mat, f32 scale);
b32 mat_add(matrix* out, const matrix* a, const matrix* b);
b32 mat_sub(matrix* out, const matrix* a, const matrix* b);
b32 mat_mul(
    matrix* out, const matrix* a, const matrix* b,
    b8 zero_out, b8 tranpose_a, b8 transpose_b
);
b32 mat_relu(matrix* out, const matrix* in);
b32 mat_softmax(matrix* out, const matrix* in);
b32 mat_cross_entropy_loss(matrix* out, const matrix* p, const matrix* q);
b32 mat_softmax_add_grad(matrix* out, const matrix* softmax_out);
b32 mat_cross_entropy_loss_grad(matrix* out, const matrix* p, const matrix* q);


int main(void) {
    mem_arena* pern_arena = arena_create(GiB(1), MiB(1));

    arena_destroy(pern_arena);

    return 0;
}