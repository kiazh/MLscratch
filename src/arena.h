#ifndef ARENA_H
#define ARENA_H

#include <stdint.h>
#include <stdbool.h>

#ifndef KiB
#define KiB(n) ((uint64_t)(n) << 10)
#endif
#ifndef MiB
#define MiB(n) ((uint64_t)(n) << 20)
#endif
#ifndef GiB
#define GiB(n) ((uint64_t)(n) << 30)
#endif

#ifndef MIN
#define MIN(a, b) (((a) < (b)) ? (a) : (b))
#endif
#ifndef MAX
#define MAX(a, b) (((a) > (b)) ? (a) : (b))
#endif
#ifndef ALIGN_UP_POW2
#define ALIGN_UP_POW2(n, p) (((uint64_t)(n) + ((uint64_t)(p) - 1)) & (~((uint64_t)(p) - 1)))
#endif

#define ARENA_BASE_POS (sizeof(mem_arena))
#define ARENA_ALIGN (sizeof(void*))

typedef struct {
    uint64_t reserve_size;
    uint64_t commit_size;
    uint64_t pos;
    uint64_t commit_pos;
} mem_arena;

typedef struct {
    mem_arena* arena;
    uint64_t start_pos;
} mem_arena_temp;

mem_arena* arena_create(uint64_t reserve_size, uint64_t commit_size);
void arena_destroy(mem_arena* arena);
void* arena_push(mem_arena* arena, uint64_t size, int32_t non_zero);
void arena_pop(mem_arena* arena, uint64_t size);
void arena_pop_to(mem_arena* arena, uint64_t pos);
void arena_clear(mem_arena* arena);

mem_arena_temp arena_temp_begin(mem_arena* arena);
void arena_temp_end(mem_arena_temp temp);
mem_arena_temp arena_scratch_get(mem_arena** conflicts, uint32_t num_conflicts);
void arena_scratch_release(mem_arena_temp scratch);

#define PUSH_STRUCT(arena, T)       (T*)arena_push((arena), sizeof(T), false)
#define PUSH_STRUCT_NZ(arena, T)    (T*)arena_push((arena), sizeof(T), true)
#define PUSH_ARRAY(arena, T, n)     (T*)arena_push((arena), sizeof(T) * (n), false)
#define PUSH_ARRAY_NZ(arena, T, n)  (T*)arena_push((arena), sizeof(T) * (n), true)

uint32_t plat_get_pagesize(void);
void*    plat_mem_reserve(uint64_t size);
int32_t  plat_mem_commit(void* ptr, uint64_t size);
int32_t  plat_mem_decommit(void* ptr, uint64_t size);
int32_t  plat_mem_release(void* ptr, uint64_t size);

#endif // ARENA_H
