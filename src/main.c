#include <stdio.h>
#include <stdbool.h>
#include <stdint.h>
#include <string.h>
#include <math.h>

#include "arena.c"
#include "psgrdn.c"

#include "arena.h"
#include "psgrdn.h"
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
b32 mat_softmax_add_grad(matrix *out, const matrix *softmax_out, const matrix *labels);
b32 mat_cross_entropy_loss_grad(matrix *out, const matrix *p, const matrix *q);

typedef enum {
    MV_FLAG_NONE = 0,

    MV_FLAG_REQUIRES_GRAD = (1 << 0),
    MV_FLAG_PARAMETER = (1 << 1),
    MV_FLAG_INPUT = (1 << 2),
    MV_FLAG_OUTPUT = (1 << 3),
    MV_FLAG_DESIRED_OUTPUT = (1 << 4),
    MV_FLAG_COST = (1 << 5),
} model_var_flags;

typedef enum {
    MV_OP_NULL = 0,
    MV_OP_CREATE,

    _MV_OP_UNARY_START,

    MV_OP_RELU,
    MV_OP_SOFTMAX, 

    MV_OP_ADD,
    MV_OP_SUB,
    MV_OP_MATMUL,
    MV_OP_CROSS_ENTROPY,
}model_var_op;

#define MODEL_VAR_MAX_INPUTS 2
#define MV_NUM_INPUTS(op) ((op) < _MV_OP_UNARY_START ? 0: ((op) < _MV_OP_BINARY_START ? 1 : 2))


typedef struct model_var{
    u32 index;
    u32 flags;

    matrix* val;
    matrix* grad;

    model_var_op op;
    struct model_var* inputs[MODEL_VAR_MAX_INPUTS];
} model_var;

typedef struct {
    model_var** vars;
    u32 size;
} model_program;

typedef struct {
    u32 num_vars;

    model_var* input;
    model_var* output;
    model_var* desired_output;
    model_var* cost;

    model_program forward_prog;
    model_program cost_prog; 
} model_context;

typedef struct {
    matrix* train_images;
    matrix* train_labels;
    matrix* test_images;
    matrix* test_labels; 

    u32 epochs;
    u32 batch_size;
    f32 learning_rate;
}model_training_desc;

model_var* mv_create(
    mem_arena* arena, model_context* model, 
    u32 rows, u32 cols, u32 flags
);
model_var* mv_relu(
    mem_arena* arena, model_context* model, 
    model_var* input, u32* flags
);
model_var* mv_softmax(
    mem_arena* arena, model_context* model, 
    model_var* input, u32 flags
);
model_var* mv_add(
    mem_arena* arena, model_context* model, 
    model_var* a, model_var* b, u32 flags
);
model_var* mv_sub(
    mem_arena* arena, model_context* model, 
    model_var* a, model_var* b, u32 flags
);
model_var* mv_matmul(
    mem_arena* arena, model_context* model, 
    model_var* a, model_var* b, u32 flags
);
model_var* mv_cross_entropy(
    mem_arena* arena, model_context* model, 
    model_var* p, model_var* q, u32 flags
);

model_program model_prog_create(mem_arena* arena, model_context* model, model_var* out_var); 
void model_prog_computer(model_program* prog);
void model_prog_compute_grads(model_program* prog);

model_context* model_create(model_program* prog);
void model_compile(mem_arena* arena, model_context* model);
void model_feedforward(model_context* model); 
void model_train(
    model_context* model, const model_training_desc* training_desc
);


void draw_mnist_digit(f32* data);

int main(void){
    mem_arena *perm_arena = arena_create(GiB(1), MiB(1));

    matrix* train_images  = mat_load(perm_arena, 60000, 784, "data/train_images.npy");
    matrix* test_images   = mat_load(perm_arena, 10000, 784, "data/test_images.npy");
    matrix* train_labels = mat_create(perm_arena, 60000, 10);
    matrix* test_labels = mat_create(perm_arena, 10000, 10);
    {
        matrix* train_labels_file = mat_load(perm_arena, 60000, 1, "data/train_labels.npy");
        matrix* test_labels_file = mat_load(perm_arena, 10000, 1, "data/test_labels.npy");

        for (u32 i = 0; i < 60000; i++){
            u32 num = train_labels_file->data[i];
            train_labels->data[i * 10 + num] = 1.0f;
        }
        for (u32 i = 0; i < 10000; i++){
            u32 num = (u32)test_labels_file->data[i];
            test_labels->data[i * 10 + num] = 1.0f;
        }

         draw_mnist_digit(train_images->data);
         for (u32 i = 0; i < 10; i++){
            printf("%.0f ", train_labels->data[i]);
         }
         printf("\n");
    }

    arena_destroy(perm_arena);

    return 0;
}

void draw_mnist_digit(f32* data){
    for (u32 y = 0; y < 28; y++ ){
        for (u32 x = 0; x <28; x++){
            f32 num = data[x + y * 28];
            u32 col = 232 + (u32)(num * 23.0f);
            printf("\x1b[48;5;%dm  ", col);
        }
        printf("\x1b[0m\n");
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

b32 mat_softmax_add_grad(matrix *out, const matrix *softmax_out, const matrix *labels){
    if (out->rows != softmax_out->rows || out->cols != softmax_out->cols){
        return false;
    }
    if (labels->rows != softmax_out->rows || labels->cols != softmax_out->cols){
        return false;
    }
    u64 size = (u64)out->rows * out->cols;
    for (u64 i = 0; i < size; i++){
        out->data[i] += softmax_out->data[i] - labels->data[i];
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

model_var* mv_create(
    mem_arena* arena, model_context* model, 
    u32 rows, u32 cols, u32 flags
){
    model_var* out = PUSH_STRUCT(arena, model_var);

    out->index = model -> num_vars++;
    out->flags = flags;
    out-> op = MV_OP_CREATE;
    out->val = mat_create(arena,rows,cols);

    if (flags & MV_FLAG_REQUIRES_GRAD) {
        out->grad = mat_create(arena, rows, cols);
    }

    if (flags & MV_FLAG_INPUT){model->input = out; }
    if (flags & MV_FLAG_OUTPUT){model->output = out; }
    if (flags & MV_FLAG_DESIRED_OUTPUT){model-> desired_output = out; }
    if (flags & MV_FLAG_COST){model->cost = out; }

    return out; 
}

model_var *_mv_unary_impl(
    mem_arena *arena, model_context *model,
    model_var *input, u32 rows, u32 cols,
    u32 flags, model_var_op op)
{
    if (flags & MV_FLAG_REQUIRES_GRAD)
    {
        flags |= MV_FLAG_REQUIRES_GRAD;
    }
    model_var *out = mv_create(arena, model, rows, cols, flags);

    out->op = op; 
    out->inputs[0] = input;
    return out; 
}

model_var* _mv_binary_impl(
    mem_arena *arena, model_context *model,
    model_var *a, model_var *b, u32 rows, u32 cols,
    u32 flags, model_var_op op)
{
    if ((a->flags & MV_FLAG_REQUIRES_GRAD) || (b->flags & MV_FLAG_REQUIRES_GRAD))
    {
        flags |= MV_FLAG_REQUIRES_GRAD;
    }
    model_var *out = mv_create(arena, model, rows, cols, flags);

    out->op = op; 
    out->inputs[0] = a;
    out->inputs[1] = b;
    return out; 
}

model_var* mv_relu(
    mem_arena* arena, model_context* model, 
    model_var* input, u32* flags
){
    return _mv_unary_impl(
        arena, model, input, input->val->rows,
        input->val->cols, flags, MV_OP_RELU
    );
}

model_var* mv_softmax(
    mem_arena* arena, model_context* model, 
    model_var* input, u32 flags
){
    return mv_unary_impl(
        arena, model, input, input->val->rows,
        input->val->cols, flags, MV_OP_SOFTMAX
    );
}


model_var* mv_add(
    mem_arena* arena, model_context* model, 
    model_var* a, model_var* b, u32 flags
){
    if(a->val->rows != b->val->rows || a->val->cols != b->val->cols){
        return NULL;
    }


    return _mv_binary_impl (
        arena, model, a, b,
        a->val->rows, a->val->cols,
        flags, MV_OP_ADD
    );
}

model_var* mv_sub(
    mem_arena* arena, model_context* model, 
    model_var* a, model_var* b, u32 flags
){
    if(a->val->cols != b->val->rows){
        return NULL;
    }


    return _mv_binary_impl (
        arena, model, a, b,
        a->val->rows, b->val->cols,
        flags, MV_OP_MATMUL
    );
}

model_var* mv_matmul(
    mem_arena* arena, model_context* model, 
    model_var* a, model_var* b, u32 flags
);
model_var* mv_cross_entropy(
    mem_arena* arena, model_context* model, 
    model_var* p, model_var* q, u32 flags
){
    if(p->val->rows != q->val->rows || p->val->cols != q->val->cols){
        return NULL;
    }
    return _mv_binary_impl (
        arena, model, p, q,
        p->val->rows, q->val->cols,
        flags, MV_OP_CROSS_ENTROPY
    );
}