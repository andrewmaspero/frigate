---
id: macos_docker
title: macOS Docker Issues
---

# macOS Docker Networking and Permissions

When offloading tasks to host services on macOS, these problems are common:

## Can't connect to host service

- Docker Desktop for Mac doesn't support `--net=host`. Use
  `host.docker.internal` to reach services running on the host.
- Ensure the host service listens on `0.0.0.0` and the port is exposed.

## Port blocked or refused

- macOS may prompt to allow incoming connections on first run. Verify the
  service is allowed in **System Settings → Network → Firewall**.
- Confirm the port in the Frigate `endpoint` matches the service port.

## Permission errors

- Host services may require Full Disk Access to read media or models.
- If a service reports `Permission denied`, grant access under **System
  Settings → Privacy & Security**.

## Misc

- Restart Docker Desktop after changing firewall or network settings.
- Check that VPN or security software is not blocking the connection.
