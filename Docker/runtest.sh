docker run --rm -dit --name=tester -v $PWD:/workspace ghcr.io/julialegate/cunumeric.jl > docker.log 2>&1
while ! docker ps | grep -q tester; do
	sleep 0.5
done
docker exec -i tester /workspace/test.sh
docker stop tester
