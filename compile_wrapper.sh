#!/bin/bash
bash /pool/david/cunumeric-play/new-build-separation/cuNumeric.jl/scripts/build_cpp_wrapper.sh \
     /pool/david/cunumeric-play/new-build-separation/cuNumeric.jl/ \
     /home/david/.julia/artifacts/4e8c4426602dd01a7d7ae60a24d91c42431c07fe/ \
     /home/david/.julia/artifacts/8e6715bc0af6fb274653a47d96c9425fa48b42ce/ \
     /home/david/.julia/artifacts/2117b531439d782e1cb0ebffcc4dcf11d274cc39/ \
     /home/david/.julia/artifacts/e83428c736a7117825caecd465ca39de72e835b8/lib \
     /pool/david/cunumeric-play/new-build-separation/cuNumeric.jl/deps/cunumeric_jl_wrapper \
     main 1
