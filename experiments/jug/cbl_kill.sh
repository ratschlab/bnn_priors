#!/usr/bin/env bash
set -euo pipefail

for host in cartwright laplace poisson vapnik gosset julia fields \
    ramanujan banach riemann euler robbins vartak curie sagarmatha ariadne \
    grothendieck babbage neumann gan bernoulli; do
    ssh "$host" 'ps aux | grep ag919' || continue;
    echo "For host $host ([0,1,2,...])? Should I kill (write \`kill\`)?"
    read command
    if [ "$command" = "kill" ]; then
        ssh "$host" 'kill $(ps aux  |grep ag919 | sed '"'"'s/  */ /g'"'"' | cut -d" " -f2)' || echo 0;
    fi
done
