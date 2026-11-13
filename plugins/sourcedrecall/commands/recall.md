---
description: Search your memory and past coding sessions (sourcedrecall)
argument-hint: what to look for, e.g. how did we fix the login test
allowed-tools: mcp__plugin_sourcedrecall_sourcedrecall__history_search, mcp__plugin_sourcedrecall_sourcedrecall__memory_recall, mcp__plugin_sourcedrecall_sourcedrecall__memory_read
---

Look this up in the user's sourcedrecall memory: $ARGUMENTS

Call history_search with it (all_projects=false). The passages come back best match first whether or not they answer it: if none of them answers it, call it again with all_projects=true. If it is about the user rather than the code (who they are, what they prefer), call memory_recall too. If a passage is close but cut short, memory_read with its session reads the whole session.

Answer from what comes back, briefly, with the date and session of each thing you use. If nothing answers it, say so plainly; do not guess. If the result says no coding sessions are stored yet, pass on its note about importing past sessions.
