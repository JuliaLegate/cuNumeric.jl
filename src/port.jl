using Sockets

"""
    @everywhere_but_repl expr

Execute `expr` on all workers except process 1.
"""
# macro everywhere_but_repl(ex)
#     esc(
#         quote
#             @everywhere begin
#                 ex
#             end
#         end
#     )
# end

#
# Helper: find (IP:PORT) for a given PID by inspecting Linux sockets
#
function _ports_for_pid(pid)
    pidstr = string(pid)

    # 1. Get all lines from `ss -ltnp` that match this pid
    raw = read(`ss -ltnp`, String)
    lines = filter(l -> occursin("pid=$pidstr", l), split(raw, '\n'))

    addrs = String[]

    for ln in lines
        parts = split(ln)
        if length(parts) >= 4
            # 2. Extract the port from the Local Address:Port field
            locale = parts[4]          # e.g., "127.0.0.1:34567"
            port = split(locale, ":")[end]  # get only the port

            # 3. Use the host's real IP
            ip_int = Sockets.getaddrinfo(gethostname()).host
            ipstr = string(Sockets.IPv4(ip_int))

            # 4. Combine IP and port
            push!(addrs, "$ipstr:$port")
        end
    end

    return addrs
end

#
# Return "IP:PORT" for each worker in Distributed.jl
#
function legate_peers()
    w = workers()
    if isempty(w)
        error("No Julia workers found! Call addprocs(...) first.")
    end

    # Fetch PID of each worker from that worker
    pidmap = Dict(wi => remotecall_fetch(() -> getpid(), wi) for wi in w)

    peers = String[]

    for wi in w
        pid = pidmap[wi]
        ports = _ports_for_pid(pid)
        if isempty(ports)
            push!(peers, "UNKNOWN:0")
        else
            push!(peers, ports[1])
        end
    end

    return peers
end

function setup_legate_env()
    if myid() != 1
        all_addrs = legate_peers()
        self_addr = all_addrs[1]

        # Exclude self and join remaining peers
        peer_addrs = join(all_addrs[2:end], " ")

        # Set environment variables
        ENV["WORKER_SELF_INFO"] = "$self_addr"
        ENV["WORKER_PEERS_INFO"] = "$peer_addrs"
        ENV["BOOTSTRAP_P2P_PLUGIN"] = "ucp"
        ENV["REALM_UCP_BOOTSTRAP_MODE"] = "p2p"

        # Optional: print to check
        println("Self: ", ENV["WORKER_SELF_INFO"])
        println("Peers: ", ENV["WORKER_PEERS_INFO"])
        println("Bootstrap plugin: ", ENV["BOOTSTRAP_P2P_PLUGIN"])
        println("Bootstrapping mode: ", ENV["REALM_UCP_BOOTSTRAP_MODE"])
    end
end
