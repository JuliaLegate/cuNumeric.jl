#ifndef CALLBACK_H
#define CALLBACK_H
namespace callback {
void mapper_register_oom_callback(void* function);
void mapper_remove_usage(uint64_t remove);
void mapper_alloc(uint64_t size);
};  // namespace callback
#endif