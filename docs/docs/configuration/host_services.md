---
id: host_services
title: Host Services
---

Frigate normally uses a ZeroMQ (ZMQ) proxy to move messages between its
internal processes. The proxy can also relay messages to services running on
the Docker host. This allows external detectors, audio transcription engines,
or embedding generators to subscribe and publish data without running inside
the container.

Set the following environment variables when starting the container:

```
FRIGATE_PROXY_HOST_PUB=tcp://host.docker.internal:5556
FRIGATE_PROXY_HOST_SUB=tcp://host.docker.internal:5557
FRIGATE_PROXY_TIMEOUT_MS=1000  # optional, defaults to 1000ms
```

`FRIGATE_PROXY_HOST_PUB` should point to a PUB socket on the host that sends
messages into Frigate. `FRIGATE_PROXY_HOST_SUB` is the SUB socket on the host
that receives messages published by Frigate. The timeout variable controls how
long the proxy waits when connecting to the host before giving up.

:::note
The host service must bind to the referenced ports and the container must be
able to reach them. On Linux hosts running `ufw` or `firewalld`, you may need
to open the ports to allow the container to connect.
:::

With the proxy bridge enabled, host services can participate in Frigate's
messaging topics for detection, transcription, and embeddings without blocking
Frigate when the host service is unavailable.

