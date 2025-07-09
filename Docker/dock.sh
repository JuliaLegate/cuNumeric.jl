DOCKER_BUILDKIT=1  docker build --platform=linux/x86_64 -f Dockerfile.conda --progress=plain -t cunumeric-dev .
echo $GH_TOKEN | docker login ghcr.io -u $GH_USER --password-stdin
docker tag cunumeric-dev ghcr.io/julialegate/cunumeric.jl:latest
docker push ghcr.io/julialegate/cunumeric.jl:latest