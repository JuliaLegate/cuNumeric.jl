#include <vector>
#include <random>
#include <ctime>

#include "jlcxx/jlcxx.hpp"

class GetVector {
    private:
        std::vector<double> values;
        
    public:
        GetVector(size_t N) : values(N) {}
        
        void set(size_t index, double val) {
            values[index] = val;
        }
        
};

JLCXX_MODULE define_julia_module(jlcxx::Module& mod) {
    mod.add_type<GetVector>("CppVector")
        .constructor<size_t>()
        .method("set", &GetVector::set);

}