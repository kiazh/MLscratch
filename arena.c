#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <stdbool.h>

#define KiB(n) ((u64)(n) << 10)
#define MiB(n) ((u64)(n) << 20)
#define GiB(n) ((u64)(n) << 30)

#define MIN(a, b) (((a) < (b)) ? (a) : (b))
#define MAX(a, b) (((a) > (b)) ? (a) : (b))
#define ALIGN_UP_POW2(n, p) (((u64)(n) + ((u64)(p) - 1)) & (~((u64)(p) - 1)))

typedef struct {
    uint64_t capacity;
    uint64_t pos;
} mem_arena;

mem_arena arena_create(uint64_t capacity); 
void arena_destroy(mem_arena* arena);

void* arena_push(mem_arena* arean, uint64_t size);
void arena_pop(mem_arena* arena, uint64_t size);
void area_pop_to(mem_arena* arena, uint64_t pos);
void arena_clear(mem_arena* arena);