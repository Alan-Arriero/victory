# VIctory

The words this project uses for running an Xbox Series title natively on
Windows, and for planning the work to get there.

## Language

### Running the title

**Title**:
The console program being run, with the DLLs it ships beside it.
_Avoid_: Game (except when naming a specific one), app, guest

**Console**:
The Xbox Series machine the title was built for, and the behaviour it expects.
_Avoid_: Guest, target, Xbox (alone)

**Host**:
The Windows machine and its Direct3D 12 device that the title actually runs on.
_Avoid_: PC, native side

**Title memory**:
Memory the title allocated and addresses directly, which on the console the GPU reads in place.
_Avoid_: Guest memory, game memory

**Refusal**:
The shader translator declining a program, with a reason in our own words.
_Avoid_: Translation failure, unsupported shader, skip

**Dropped draw**:
A draw or dispatch the title issued that the host did not run.
_Avoid_: Skipped draw, missing draw

**Shortcut**:
A place where the code knowingly does less than the console does, marked as such in the source.
_Avoid_: TODO, hack, stub

**Console wave**:
The group of lanes a console shader executes together, 64 wide on this console.
_Avoid_: Wave64 (as a noun), wavefront

### Planning the work

**Reference machine**:
The hardware and OS a result must hold on for a ticket to be done.
_Avoid_: Target machine, test rig

**Tier**:
How much a class of hardware counts: tier 1 decides whether work is done, tier 2 is kept working but does not gate.
_Avoid_: Support level

**Lit and controllable**:
The first stage of the destination: the world is drawn with correct lighting and answers input, at any frame rate.
_Avoid_: Working, in-game, booting to gameplay

**Playable**:
The second stage of the destination: lit and controllable at a frame rate and for a duration a person would accept.
_Avoid_: Working, running, compatible

**Map**:
The single issue that states the destination and indexes what has been decided on the way to it.
_Avoid_: Roadmap, tracking issue, epic

**Ticket**:
A sub-issue of the map: one decision to make or one piece of work to do.
_Avoid_: Task (for decisions), story

**Baseline**:
A recorded run on the reference machine that later results are compared against.
_Avoid_: Benchmark, test report
