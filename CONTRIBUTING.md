# Contributing to VIctory

You can help with source code, synthetic tests, documentation, research, or
reproducible test reports. Most of the work needs no game files: the test
suite runs on synthetic data, and many tickets are marked as needing neither
the game nor a GPU.

## The one rule

Nothing from the game, or from any other proprietary title, enters this
repository, an issue, a comment or a pull request. That covers game files,
executable bytes, disassembly, shader dumps, captures, screenshots and logs
that contain game material. Read the [asset policy](docs/asset-policy.md)
before you share anything. Describe what you saw in words, with counts, our
own log strings and `module+rva` addresses.

## Find something to do

Work is planned on one issue, the
[map](https://github.com/JMAN730/victory/issues/35). It states where the
project is going (the title lit and controllable on the reference machine,
then playable), what has been decided so far, and what is still unclear. Its
sub-issues are the tickets.

Each ticket carries one label:

| Label | What it is | Who works it |
| --- | --- | --- |
| `wayfinder:task` | Work with done criteria: a fix whose shape is known, or an investigation | Anyone. Investigations need the game and the hardware |
| `wayfinder:research` | Facts from documentation or other public sources | Anyone, or an agent |
| `wayfinder:grilling` | A decision between named options | Settled with the maintainer; evidence is welcome as comments |
| `wayfinder:prototype` | A decision that needs something rough to react to | Anyone, with the maintainer |

A ticket is free when it is open, has no assignee, and is not marked as
blocked by another open ticket. GitHub shows the blocked ones on the map's
sub-issue list. Tickets that say "a good first ticket" need neither the game
nor a GPU.

Something not on the map? Open an issue with the bug or feature form. One
problem per issue.

## Claim a ticket

Comment on the ticket to say you are taking it, and which part if you are
not taking all of it. The maintainer assigns it to you; the assignee is the
claim. If the ticket names a pull request already in flight, help there
(review it, test it on your hardware) rather than starting a second one.

If you stop, say so on the ticket so someone else can pick it up.

## Set up

Follow the requirements in the [README](README.md#getting-started). On x86-64
Linux, `nix develop` or `nix-shell` supplies the pinned Rust toolchain, native
build tools and Windows cross-compiler; see the
[Linux guide](docs/linux-proton.md#development-with-nix). Then:

```powershell
cargo xtask install-hooks                # asset-policy check on every commit
cargo test --workspace --locked          # needs no game data
```

To run the title you also need `victory.toml` (copy `victory.example.toml`)
with every path outside the repository.

## Make the change

- Branch from `main`. One ticket per pull request; keep it focused.
- Follow the [code standards](docs/code-standards.md) and use the words in
  the [glossary](GLOSSARY.md).
- A fix comes with a test that fails without it, built from synthetic data.
- If the change leaves something approximate, mark it with a `// shortcut:`
  comment rather than a TODO.

## Before you open a pull request

Run what CI runs:

```powershell
cargo fmt --all
cargo clippy --workspace --all-targets --locked -- -D warnings
cargo clippy --workspace --all-targets --all-features --locked -- -D warnings
cargo test --workspace --locked
cargo xtask policy
```

CI repeats these on Windows, repeats clippy and the tests on Linux for the
portable crates, and checks the pull request description.

## Open the pull request

Use the template. `## What` and `## Tested` must both have text, and CI
fails the pull request if either is empty.

- **What:** the problem and how the change solves it. Link the ticket
  (`Closes #123`).
- **Tested:** the hardware, the OS and the commands. Say what you did not
  test. "Not tested" is a permitted answer; an untested change that says so
  is more useful than one that implies otherwise.
- If an AI tool wrote part of the change, say so in the description.

Commit subjects name the area and state the behaviour, for example
`GPU: no blending on integer render targets in host pipelines`. The
[code standards](docs/code-standards.md#commits) have the details.

## Keep references inside this repository

Describe a finding or a change from elsewhere by its subject. Do not write
issue or pull request numbers, links or @mentions that point at another
repository, in issues, pull requests or commit messages: GitHub turns them
into backlinks and notifications there. Credit a change that originated
elsewhere by keeping its author on the commit.

## Write a useful test report

Include the VIctory commit, build profile, Windows version, CPU, GPU and
driver. Give the command or settings needed to reproduce the problem, the
last stage that worked, and the exact error code if there is one. Replace
personal paths with placeholders. When you compare two builds, use the same
settings and give both commit IDs.

Say which tier your machine is in. The reference machine is Windows 11 with
an NVIDIA GPU that runs 32-lane waves and has at least 12 GB of video
memory; a result there decides whether a ticket is done. Reports from other
hardware are welcome and are read as tier 2.

For a GPU failure, follow
[Reporting a GPU failure](README.md#reporting-a-gpu-failure) in the README.

## Working with agents

`AGENTS.md` and `docs/agents/` tell coding agents where the tracker, the
labels and the domain documents live. An agent session follows the same
rules as a person: the asset policy, the code standards, one ticket at a
time, and a claim before any work.
