# SQLite amalgamation

This directory contains the SQLite 3.50.1 amalgamation used by Android NDK
builds of the optional llama-agent SQLite backend.

The source archive was downloaded from:

<https://www.sqlite.org/2025/sqlite-amalgamation-3500100.zip>

Archive SHA-256:

```text
41716b44ac8777188c4c3f1f370f01c9cb9e3b6428eb5c981d086c35de2d9d3f
```

Linux builds continue to use the system SQLite development library. Android
builds compile this amalgamation as a private static target and disable loadable
extensions. SQLite is public-domain software; see `SQLITE-LICENSE.txt` for the
distribution notice.
