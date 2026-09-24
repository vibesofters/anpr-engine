# Phase 2E owner-operated manual staging runbook

Status: **BOUNDED PRIVATE STAGING COMPLETED AND STOPPED**. The procedure below
is retained for reproducibility; do not repeat it without a new bounded reason
and owner approval. This is a private, loopback-only measurement exercise,
not public deployment. Coolify can manage
the staging service, but its healthy status is not a CPU/memory benchmark.
Do not start if either code-only amd64 image, the approved private
model bundle, an owner-managed runtime credential, this exact Compose file,
and synthetic non-identifying test images cannot be delivered to the host by
an owner-controlled **private** method. Do not use a public registry, publish
the repository, copy weights into images, or send any secret/path to the agent.

**Stop conditions, before starting:** reserve at most 15 minutes and 12 test
requests. Skip staging if current available RAM is below 6 GiB, root free
space below 20 GiB, sustained load above 4 on the 8-vCPU host, or loopback
port 39083 is occupied. Stop immediately if available RAM falls below 6 GiB,
host CPU exceeds 70% or load exceeds 4 for 30 seconds, root free space falls
by more than 1 GiB **after image loading**, either ANPR container OOMs/restarts, or any other project
shows degradation. Do not raise the temporary caps to force a passing run.

## 1. What to click in Coolify

1. In the owner dashboard, note the Coolify version and selected server's OS
   version. Use the dedicated **staging** environment and the ANPR-only Compose
   service; do not add a public domain, expose the inference port, or change
   another project's resource. The web port must remain bound to host loopback.
2. The model-bundle source and internal service credential are runtime-only
   Coolify variables. The owner enters the credential privately; never reveal
   it in screenshots, logs, generated-Compose excerpts, or chat. Validate the
   Compose file before deployment. The generated preview may retain variable
   references; healthy model readiness is the practical mount/hash check.
3. For read-only host measurements, use **Server → Terminal** if it connects.
   If the browser's real-time/terminal connection does not work, do not disable
   security or monitoring to bypass it. Use a separate owner-controlled SSH
   session for the read-only commands below, or pause the benchmark. Do not
   enable Coolify's server-wide metrics merely to measure this one project.
4. Confirm in the dashboard's aggregate server view that other projects
   appear healthy before and during the short test, without opening their
   logs or configuration. A Coolify "Running (healthy)" badge proves startup,
   not low resource use or an end-to-end inference result.

**Coolify Stop safeguard:** Stop only the ANPR staging service. In the Stop
dialog, turn **off** "Run Docker Cleanup (remove unused images and builder
cache)" before confirming; it may be selected by default and can affect
shared-host resources. Confirm both ANPR components show `Exited` after a
refresh. Do not run Force Cleanup Containers or any Docker-wide prune.

## 2. Read-only commands in the server terminal

Run these before staging; send only their non-secret values, not terminal
screenshots. The commands do not list other containers or projects:

```sh
uname -s -m -r
sed -n '1,6p' /etc/os-release
getconf _NPROCESSORS_ONLN
awk '/^(MemTotal|MemAvailable|SwapTotal|SwapFree):/ {print $1,$2,$3}' /proc/meminfo
df -B1 --output=size,avail /
awk '{print $1,$2,$3}' /proc/loadavg
docker version --format '{{.Server.Version}}'
docker compose version --short
docker info --format 'arch={{.Architecture}} cgroup={{.CgroupVersion}} driver={{.Driver}} running={{.ContainersRunning}}'
ss -Hln '( sport = :39083 )'
```

The last command must have **no output**. Confirm Docker can enforce CPU,
memory, PID, read-only-root, tmpfs, bind-mount, and loopback-port limits on
this host; a failed capability check is a stop, not a reason to relax the
boundary. Do not run a Docker-wide inspection, cleanup, or stats command.

## 3. Minimum temporary ANPR-only staging actions

Choose **one** controller: the existing Coolify-managed staging service **or**
the manual Compose procedure below. Never start a second Compose project
alongside the Coolify service. For the Coolify-managed service, use its own
Deploy/Stop controls and read-only SSH diagnostics; do not run `docker compose
up`, `down`, or `--remove-orphans` against Coolify's service directory.

The numbered procedure below is only for a host on which no Coolify-managed
ANPR staging resource exists.

1. Privately load the **code-only** images built off-host, if not already
   loaded. On the owner's local Docker machine, `docker save -o
   <owner-private-image-archive> anpr-engine-web:phase2e-amd64-local
   anpr-engine-inference:phase2e-amd64-local`; privately transfer it and
   compare its SHA-256 before `docker load -i <owner-private-image-archive>`
   on the server. Verify `docker image inspect` gives exactly the
   Linux amd64 image IDs in the [image preflight](../reports/phase2e-amd64-image-preflight.md).
   Do not rebuild on the shared host. If private delivery cannot be arranged,
   stop here. The private model pair remains outside the images and is mounted
   read-only; readiness must fail on either SHA-256 mismatch.
2. Place the reviewed `deploy/docker/compose.phase2e-staging.yaml` in an
   ANPR-only temporary working directory. The owner prepares a mode-0600
   runtime env file there containing only `ANPR_PRIVATE_MODEL_BUNDLE_SOURCE`
   and `ANPR_INTERNAL_SERVICE_CREDENTIAL`; do not paste its values into the
   dashboard transcript or send it to the agent. The Compose file publishes
   the web container **only on host loopback** and publishes no inference
   port. Verify no public route/domain/proxy mapping exists. If the Compose
   project name `anpr-phase2e-staging` is already in use, stop; do not adopt
   or remove someone else's resources.
3. From that directory, start only this exact Compose project:

   ```sh
   docker compose -p anpr-phase2e-staging --env-file <owner-private-env-file> -f compose.phase2e-staging.yaml up -d --no-build --pull never
   docker compose -p anpr-phase2e-staging --env-file <owner-private-env-file> -f compose.phase2e-staging.yaml ps
   ```

   Wait for both health checks; record seconds from start to inference
   readiness as an upper bound on startup plus model loading; the service
   does not separately instrument model-only load time. The fixed
   **protective trial caps**, not final sizing, are
   inference 0.75 CPU/2 GiB, web 0.25 CPU/512 MiB, one Python process,
   one active inference, zero queue, and 128/64 MiB memory-backed temporary
   space. If loading fails within these caps, stop and report it.
4. Use only synthetic, non-identifying JPEG/PNG images through the loopback
   BFF. Establish one cold request, two warm serial requests, and two
   simultaneous submissions (same session) expecting one immediate busy
   rejection. Exercise cancellation, a bounded timeout, restart/recovery,
   health/readiness and a blank-image no-detection case. Test only safe status
   codes and timings; do not print response bodies, plate strings, cookies,
   image bytes, or headers containing credentials. Do not run automatic
   retries, a load generator, or more than 12 requests. Stop on the thresholds
   above. The BFF's existing 125-second deadline exceeds the Python
   120-second deadline; any proxy deadline is **unverified** because no
   public/proxied upload route is permitted in this run.
   This architecture supports one resident model pair; do not implement
   on-demand loading merely to create a comparison benchmark.

## 4. Sanitized measurements to send back

Send a small table with: observation time; OS and Coolify versions; cgroup
version/limit support; pre/peak/post available RAM, root free space, aggregate
CPU/load; ANPR **web and inference only** idle/peak CPU and memory (`docker
stats --no-stream` with their two exact Compose container IDs); startup and
model-readiness seconds; cold/warm/serial request durations and safe HTTP
status categories; simultaneous-request outcome and rejection delay;
cancellation, timeout, restart/recovery outcomes; ANPR tmpfs/disk-spooling
observations; health/shutdown result; whether any other project was affected;
and whether the stop threshold was crossed. Confirm model-hash readiness,
image IDs, loopback-only web port, no inference host port, and no retained
request artifacts. Mark **Coolify proxy buffering/timeouts not tested** if
the no-public-route boundary prevents measuring them. Do not send raw logs,
names of other projects, IPs, credentials, private mount paths, image content,
plate text, predictions, or corrections.

For the two-container readings, use only project-scoped IDs (do not run
unfiltered `docker stats`):

```sh
anpr_web_id=$(docker compose -p anpr-phase2e-staging --env-file <owner-private-env-file> -f compose.phase2e-staging.yaml ps -q web)
anpr_inference_id=$(docker compose -p anpr-phase2e-staging --env-file <owner-private-env-file> -f compose.phase2e-staging.yaml ps -q inference)
docker stats --no-stream --format '{{.Name}} {{.CPUPerc}} {{.MemUsage}} {{.PIDs}}' "$anpr_web_id" "$anpr_inference_id"
docker inspect --format '{{.Name}} memory={{.HostConfig.Memory}} nano_cpus={{.HostConfig.NanoCpus}} read_only={{.HostConfig.ReadonlyRootfs}} restarts={{.RestartCount}}' "$anpr_web_id" "$anpr_inference_id"
```

## 5. Stop and remove only ANPR staging resources

For a Coolify-managed resource, use the **Coolify Stop safeguard** in section 1;
do not use the commands below. They apply only to the manual Compose variant,
using the same Compose file and private env file:

```sh
docker compose -p anpr-phase2e-staging --env-file <owner-private-env-file> -f compose.phase2e-staging.yaml down --remove-orphans
docker compose -p anpr-phase2e-staging --env-file <owner-private-env-file> -f compose.phase2e-staging.yaml ps
```

Verify loopback port 39083 is no longer listening, then remove only the two
exact ANPR staging image tags and owner-transferred archives if no longer
needed. Remove only this staging directory's env file and synthetic fixtures
using the owner's normal secure method. Do **not** delete the canonical
private model bundle or run `docker system prune`, `docker volume prune`, a
wildcard removal, or any Docker-wide cleanup. Leave other Coolify resources
untouched. If cleanup fails, report the exact ANPR-only resource still present.
