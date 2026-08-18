"""Return the Realm P2P address for one Julia worker."""
function worker_addr(worker_id::Integer)
    address_expr = quote
        using Sockets

        let local_process = Distributed.LPROC
            port = local_process.bind_port
            if iszero(port)
                error("Julia worker ", Distributed.myid(), " has no listening port")
            end

            # Realm needs a different interface because Julia already owns bind_addr:port.
            hostname_ip = Sockets.getaddrinfo(Sockets.gethostname()).host
            string(Sockets.IPv4(hostname_ip), ':', port)
        end
    end

    # Core.eval works before DistributedExt or cuNumeric is loaded on the worker.
    return Distributed.remotecall_fetch(Core.eval, worker_id, Main, address_expr)
end

"""Return the Realm P2P address of every Julia worker."""
function legate_peers(worker_ids=Distributed.workers())
    isempty(worker_ids) && error("No Julia workers found! Call addprocs(...) first.")
    return Dict(worker_id => worker_addr(worker_id) for worker_id in worker_ids)
end

function setup_legate_env(worker_addrs)
    self_addr = worker_addrs[Distributed.myid()]
    peers_info = join(sort!(collect(values(worker_addrs))), " ")

    ENV["WORKER_SELF_INFO"] = self_addr
    ENV["WORKER_PEERS_INFO"] = peers_info # Realm expects self to be included.
    ENV["REALM_UCP_BOOTSTRAP_PLUGIN"] = "realm_ucp_bootstrap_p2p.so"
    ENV["REALM_UCP_BOOTSTRAP_MODE"] = "p2p"

    println("Self: ", self_addr)
    println("Peers: ", peers_info)
    println("Bootstrap plugin: realm_ucp_bootstrap_p2p.so")
    println("Bootstrapping mode: p2p")
    return nothing
end
