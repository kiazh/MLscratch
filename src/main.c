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
#define _CRT_SECURE_NO_WARNINGS

typedef struct{
    uint32_t rows, cols;
    float *data;
} matrix;

matrix *mat_create(mem_arena *arena, uint32_t rows, uint32_t cols);
matrix* mat_load(mem_arena* arena, u32 rows,u32 cols, const char* filename);
void mat_clear(matrix *mat);
b32 mat_copy(matrix *dst, matrix *src);
void mat_fill(matrix *mat, f32 x);
void mat_scale(matrix *mat, f32 scale);
b32 mat_add(matrix *out, const matrix *a, const matrix *b);
b32 mat_sub(matrix *out, const matrix *a, const matrix *b);
b32 mat_mul(
    matrix *out, const matrix *a, const matrix *b,
    b8 zero_out, b8 transpose_a, b8 transpose_b);
f32 mat_sum(const matrix *mat);
b32 mat_relu(matrix *out, const matrix *in);
b32 mat_softmax(matrix *out, const matrix *in);
b32 mat_cross_entropy_loss(matrix *out, const matrix *p, const matrix *q);
b32 mat_softmax_add_grad(matrix *out, const matrix *softmax_out);
b32 mat_cross_entropy_loss_grad(matrix *out, const matrix *p, const matrix *q);

void draw_mnist_digit(f32* data);

int main(void){
    mem_arena *perm_arena = arena_create(GiB(1), MiB(1));

    matrix* train_images  = mat_load(perm_arena, 60000, 784, "data/train_images.npy");
    matrix* test_images   = mat_load(perm_arena, 10000, 784, "data/test_images.npy");
    matrix* train_labels = mat_create(perm_arena, 60000, 10);
    matrix* test_labels = mat_create(perm_arena, 10000, 10);


    draw_mnist_digit(train_images->data);

    arena_destroy(perm_arena);

    return 0;
}

void draw_mnist_digit(f32* data){
    for (u32 y = 0; y < 28; y++ ){
        for (u32 x = 0; x <28; x++){
            f32 num = data[x + y * 28];
            u32 col = 232 + (u32)(num * 24);
            printf("\x1b[48;5;%dm  ", col);
        }
    }
}

matrix *mat_create(mem_arena *arena, uint32_t rows, uint32_t cols){
    matrix *mat = PUSH_STRUCT(arena, matrix);

    mat->rows = rows;
    mat->cols = cols;
    mat->data = PUSH_ARRAY(arena, float, rows *cols);

    return mat;
}

static void npy_skip_header(FILE *f) {
    u8 buf[8];
    fread(buf, 1, 8, f); // magic(6) + major(1) + minor(1)
    u32 header_len;
    if (buf[6] == 1) {
        u16 hlen; fread(&hlen, 2, 1, f); header_len = hlen;
    } else {
        fread(&header_len, 4, 1, f);
    }
    fseek(f, header_len, SEEK_CUR);
}

matrix* mat_load(mem_arena* arena, u32 rows, u32 cols, const char* filename){
    FILE* f = fopen(filename, "rb");
    if (!f) {
        fprintf(stderr, "mat_load: could not open '%s'\n", filename);
        return NULL;
    }

    matrix* mat = mat_create(arena, rows, cols);

    npy_skip_header(f);

    u64 size = sizeof(f32) * rows * cols;
    fread(mat->data, 1, size, f);

    fclose(f);

    return mat;
}

b32 mat_copy(matrix *dst, matrix *src){
    if (dst->rows != src->rows || dst->cols != src->cols)
    {
        return false;
    }
    memcpy(dst->data, src->data, sizeof(f32) * dst->rows * dst->cols);
    return true;
}

void mat_clear(matrix *mat){
    memset(mat->data, 0, sizeof(f32) * (u64)mat->rows * mat->cols);
}

void mat_fill(matrix *mat, f32 x){
    u64 size = (u64)mat->rows * mat->cols;
    for (u64 i = 0; i < size; i++)
    {
        mat->data[i] = x;
    }
}

void mat_scale(matrix *mat, f32 scale){
    u64 size = (u64)mat->rows * mat->cols;
    for (u64 i = 0; i < size; i++)
    {
        mat->data[i] *= scale;
    }
}

f32 mat_sum(const matrix *mat){
    u64 size = (u64)mat->rows * mat->cols;
    f32 sum = 0.0f;
    for (u64 i = 0; i < size; i++)
    {
        sum += mat->data[i];
    }
    return sum;
}

b32 mat_add(matrix *out, const matrix *a, const matrix *b){
    if (a->rows != b->rows || a->cols != b->cols)
    {
        return false;
    }
    if (out->rows != a->rows || out->cols != a->cols)
    {
        return false;
    }

    u64 size = (u64)a->rows * a->cols;
    for (u64 i = 0; i < size; i++)
    {
        out->data[i] = a->data[i] + b->data[i];
    }
    return true;
}

b32 mat_sub(matrix *out, const matrix *a, const matrix *b){
    if (a->rows != b->rows || a->cols != b->cols){
        return false;
    }
    if (out->rows != a->rows || out->cols != a->cols){
        return false;
    }

    u64 size = (u64)a->rows * a->cols;
    for (u64 i = 0; i < size; i++){
        out->data[i] = a->data[i] - b->data[i];
    }
    return true;
}

void _mat_mul_nn(matrix *out, const matrix *a, const matrix *b)
{
    for (u64 i = 0; i < out->rows; i++){
        for (u64 k = 0; k < a->cols; k++){
            for (u64 j = 0; j < out->cols; j++){
                out->data[j + i * out->cols] +=
                    a->data[k + i * a->cols] *
                    b->data[j + k * b->cols];
            }
        }
    }
}
void _mat_mul_nt(matrix *out, const matrix *a, const matrix *b){
    for (u64 i = 0; i < out->rows; i++){
        for (u64 j = 0; j < out->cols; j++){
            for (u64 k = 0; k < a->cols; k++){
                out->data[j + i * out->cols] +=
                    a->data[k + i * a->cols] *
                    b->data[k + j * b->cols];
            }
        }
    }
}
void _mat_mul_tn(matrix *out, const matrix *a, const matrix *b){
    for (u64 k = 0; k < a->rows; k++){
        for (u64 i = 0; i < out->rows; i++){
            for (u64 j = 0; j < out->cols; j++){
                out->data[j + i * out->cols] +=
                    a->data[i + k * a->cols] *
                    b->data[j + k * b->cols];
            }
        }
    }
}
void _mat_mul_tt(matrix *out, const matrix *a, const matrix *b){
    for (u64 i = 0; i < out->rows; i++){
        for (u64 k = 0; k < a->rows; k++){
            for (u64 j = 0; j < out->cols; j++){
                out->data[j + i * out->cols] +=
                    a->data[i + k * a->cols] *
                    b->data[k + j * b->cols];
            }
        }
    }
}

b32 mat_mul(
    matrix *out, const matrix *a, const matrix *b,
    b8 zero_out, b8 transpose_a, b8 transpose_b){

    u32 a_rows = transpose_a ? a->cols : a->rows;
    u32 a_cols = transpose_a ? a->rows : a->cols;
    u32 b_rows = transpose_b ? b->cols : b->rows;
    u32 b_cols = transpose_b ? b->rows : b->cols;

    if (a_cols != b_rows)
    {
        return false;
    }
    if (out->rows != a_rows || out->cols != b_cols)
    {
        return false;
    }
    if (zero_out)
    {
        mat_clear(out);
    }

    u32 transpose = (transpose_a << 1) | transpose_b; 
    switch(transpose) {
        case 0b00: {_mat_mul_nn(out, a, b); } break;
        case 0b01: {_mat_mul_nt(out, a, b); } break;
        case 0b10: {_mat_mul_tn(out, a, b); } break;
        case 0b11: {_mat_mul_tt(out, a, b); } break;
    }
    return true;
}

b32 mat_relu(matrix *out, const matrix *in){
    if (out->rows != in->rows || out->cols != in->cols){
        return false;
    }

    u64 size = (u64)out->rows * out->cols;
    for (u64 i = 0; i < size; i++){
        out->data[i] = MAX(0, in->data[i]);
    }
    return true; 
}


b32 mat_softmax(matrix *out, const matrix *in){
    if (out->rows != in->rows || out->cols != in->cols){
        return false;
    }

    for (u32 r = 0; r < in->rows; r++){
        f32 *in_row  = in->data  + r * in->cols;
        f32 *out_row = out->data + r * out->cols;

        f32 max_val = in_row[0];
        for (u32 c = 1; c < in->cols; c++){
            if (in_row[c] > max_val) max_val = in_row[c];
        }

        f32 sum = 0.0f;
        for (u32 c = 0; c < in->cols; c++){
            out_row[c] = expf(in_row[c] - max_val);
            sum += out_row[c];
        }

        f32 inv_sum = 1.0f / sum;
        for (u32 c = 0; c < out->cols; c++){
            out_row[c] *= inv_sum;
        }
    }

    return true;
}


b32 mat_cross_entropy_loss(matrix *out, const matrix *p, const matrix *q){
    if (p->rows != q->rows || p->cols != q->cols){
        return false; 
    }
    if (out->rows != q->rows || out->cols != q->cols){
        return false; 
    }
    u64 size = (u64)out->rows * out->cols;
    for (u64 i = 0; i < size; i++) {
        out->data[i] = p->data[i] == 0.0f ?
        0.0f : p->data[i] * -logf(q->data[i] + 1e-7f);
    }
    
    return true;

}

b32 mat_softmax_add_grad(matrix *out, const matrix *softmax_out){
    if (out->rows != softmax_out->rows || out->cols != softmax_out->cols){
        return false;
    }
    u64 size = (u64)out->rows * out->cols;
    for (u64 i = 0; i < size; i++){
        f32 s = softmax_out->data[i];
        out->data[i] += s * (1.0f - s);
    }
    return true;
}

b32 mat_cross_entropy_loss_grad(matrix *out, const matrix *p, const matrix *q){
    if (p->rows != q->rows || p->cols != q->cols){
        return false;
    }
    if (out->rows != p->rows || out->cols != p->cols){
        return false;
    }
    u64 size = (u64)out->rows * out->cols;
    for (u64 i = 0; i < size; i++){
        out->data[i] = -p->data[i] / (q->data[i] + 1e-7f);
    }
    return true;
}
