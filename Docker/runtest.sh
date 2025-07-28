docker run --rm -it --name=tester -v $PWD:/workspace ghcr.io/julialegate/cunumeric.jl > docker.log 2>&1 &
docker exec -i tester /workspace/test.sh
docker stop tester