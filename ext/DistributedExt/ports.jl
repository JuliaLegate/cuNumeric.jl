"""Return a Realm P2P address for each Distributed.jl worker."""
function legate_peers(worker_ids=Distributed.workers())
    if isempty(worker_ids)
        error("No Julia workers found! Call addprocs(...) first.")
    end

    worker_address_expr = quote
        using Sockets

        let local_process = Distributed.LPROC
            # Distributed.start_worker stores the address of its listening socket here.
            # Querying the worker directly avoids OS-specific socket/process inspection.
            port = local_process.bind_port
            if iszero(port)
                error("Julia worker ", Distributed.myid(), " has no listening port")
            end

            # Realm needs a different interface because Julia already owns bind_addr:port.
            hostname_ip = Sockets.getaddrinfo(Sockets.gethostname()).host
            string(Sockets.IPv4(hostname_ip), ':', port)
        end
    end

    peers = Dict{Int,String}()
    for worker_id in worker_ids
        peers[worker_id] = Distributed.remotecall_fetch(
            Core.eval, worker_id, Main, worker_address_expr
        )
    end

    return peers
end

function setup_legate_env(worker_addrs)
    self_addr = worker_addrs[Distributed.myid()]

    # Realm expects the complete peer set, including this worker.
    peer_addrs = join(sort!(collect(values(worker_addrs))), " ")

    # Set environment variables
    ENV["WORKER_SELF_INFO"] = "$self_addr"
    ENV["WORKER_PEERS_INFO"] = "$peer_addrs"
    ENV["REALM_UCP_BOOTSTRAP_PLUGIN"] = "realm_ucp_bootstrap_p2p.so"
    ENV["REALM_UCP_BOOTSTRAP_MODE"] = "p2p"

    # Optional: print to check
    println("Self: ", ENV["WORKER_SELF_INFO"])
    println("Peers: ", ENV["WORKER_PEERS_INFO"])
    println("Bootstrap plugin: ", ENV["REALM_UCP_BOOTSTRAP_PLUGIN"])
    println("Bootstrapping mode: ", ENV["REALM_UCP_BOOTSTRAP_MODE"])
    return nothing
end
