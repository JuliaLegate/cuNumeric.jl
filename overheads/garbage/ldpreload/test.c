#define _GNU_SOURCE
#include <dlfcn.h>
#include <malloc.h>
#include <stdint.h>
#include <stdio.h>

static int init = 0;
static size_t count = 0;
static uint64_t iter = 0;

#define ITERS 50

static void* (*real_malloc)(size_t) = NULL;

static void* (*real_free)(void*) = NULL;

static inline void mtrace_init(void) {
  if (!init) {
    real_free = dlsym(RTLD_NEXT, "free");
    if (NULL == real_free) {
      fprintf(stderr, "Error in `dlsym`: %s\n", dlerror());
    }
    real_malloc = dlsym(RTLD_NEXT, "malloc");
    if (NULL == real_malloc) {
      fprintf(stderr, "Error in `dlsym`: %s\n", dlerror());
    }
    init = 1;
  }
}
void* malloc(size_t size) {
  mtrace_init();
  void* p = NULL;
  // fprintf(stderr, "malloc(%lu) = ", size);
  p = real_malloc(size);
  // fprintf(stderr, "%p\n", p);
  if (p) {
    count += malloc_usable_size(p);
  }

  iter++;
  if (!(iter % ITERS)) {
    fprintf(stderr, "malloc %lu %lu \n", iter, count);
  }
  return p;
}

void free(void* ptr) {
  mtrace_init();
  size_t size = 0;
  if (ptr) {
    size = malloc_usable_size(ptr);
  }
  //	fprintf(stderr, "free(%p) with size %lu \n", ptr, size);
  real_free(ptr);
  count -= size;
  iter++;

  if (!(iter % ITERS)) {
    fprintf(stderr, "free %lu %lu \n", iter, count);
  }
}
