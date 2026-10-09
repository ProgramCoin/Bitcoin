
### Hardware Note: This project was developed and tested on an NVIDIA GeForce RTX 3050 Laptop GPU. While capable as a general-purpose GPU, the RTX 3050 provides relatively limited compute performance for Bitcoin's SHA-256 mining workload. As a result, the mining performance demonstrated by this project should not be interpreted as the maximum performance of the software on more powerful hardware


## Bitcoin Core CUDA solo miner

The runner supports Bitcoin Core `regtest`, a mainnet dry-run, and live
mainnet solo mining. Regtest remains the default. Both mainnet modes require a
valid mainnet payout address and use the live coinbase value and transaction
set supplied by `getblocktemplate`; they build the transaction Merkle root and
SegWit witness commitment before scanning. Only `--live-mainnet` can submit a
mainnet block. Always verify the configured payout address before starting.

### Regtest

Start a local regtest node in one PowerShell window:

```powershell
& "C:\Program Files\Bitcoin\daemon\bitcoind.exe" -regtest -server
```

From the project directory, build and run the miner:

```powershell
nvcc -O3 -arch=sm_86 -Xcompiler /Brepro -Xlinker /Brepro cuda_miner.cu -o cuda_miner.exe
python regtest_miner.py
```

Regtest sends its coinbase output to an `OP_TRUE` anyone-can-spend script and
is for local testing only. The runner first confirms that the node reports the
`regtest` chain and refuses to use that script otherwise.

### Mainnet

Use a fully synchronized mainnet Bitcoin Core node with RPC enabled locally.
Never expose Bitcoin Core RPC to the public internet. `--mainnet` on its own
never mines or submits; it must be combined with exactly one mode:

| Flags | Behaviour |
| --- | --- |
| `--mainnet --dry-run` | Full template construction, payout verification, CUDA scanning and candidate verification. Never calls `submitblock`; a verified candidate is saved to disk. Runs until interrupted. |
| `--mainnet --live-mainnet` | The same pipeline, and a verified candidate is saved and submitted with `submitblock`. Stops mining after the first accepted block, then watches that block until its coinbase matures. |
| `--mainnet --monitor-block <hash>` | Mines nothing. Reports that block's confirmations, any reorganization, and the wallet's view of its coinbase until it is spendable. |

`--live-mainnet` without `--mainnet`, or together with `--dry-run`, is
rejected before any RPC call.

#### Payout

Both modes require `--payout-address <address>`. The miner needs only the
public address; it never uses a private key, wallet passphrase or wallet RPC.
Bitcoin Core's `validateaddress` must report the address as valid on the
`main` chain and return a non-empty `scriptPubKey`, which becomes the only
value-carrying coinbase output. The full `coinbasevalue` from
`getblocktemplate` is paid to that script; the only other output is the
zero-value SegWit witness commitment.

`--expected-payout-script <hex>` is required with `--live-mainnet` and
recommended with `--dry-run`. The script Core resolves is compared
byte-for-byte with it and startup aborts on any difference, which protects
against a mistyped or substituted address. Get the value once and reuse it:

```powershell
& "C:\Program Files\Bitcoin\daemon\bitcoin-cli.exe" validateaddress "<your-mainnet-address>"
```

Live mainnet additionally verifies wallet ownership. It calls Bitcoin Core's
wallet RPC `getaddressinfo` for the payout address and requires `ismine` to be
`true` and the wallet's `scriptPubKey` to equal the validated, pinned script.
Startup aborts if no wallet is loaded, the wallet RPC fails, the address is
watch-only or foreign, or the scripts differ. Only the public script and the
`ismine` flag are read; descriptors and key metadata are never printed, and no
private key or passphrase is requested, so the wallet may stay locked. With
more than one wallet loaded, `getaddressinfo` is ambiguous and the check fails
closed; load only the wallet that owns the payout address. Dry-run and regtest
do not query the wallet.

#### Dry-run

```powershell
nvcc -O3 -arch=sm_86 -Xcompiler /Brepro -Xlinker /Brepro cuda_miner.cu -o cuda_miner.exe
python regtest_miner.py --mainnet --dry-run --bitcoin-conf "C:\Program Files\Bitcoin\bitcoin.conf" --payout-address "<your-mainnet-address>" --expected-payout-script "<scriptPubKey-hex>"
```

#### Live mining

```powershell
python regtest_miner.py --mainnet --live-mainnet --bitcoin-conf "C:\Program Files\Bitcoin\bitcoin.conf" --payout-address "<your-mainnet-address>" --expected-payout-script "<scriptPubKey-hex>"
```

`--mainnet`, `--live-mainnet`, `--payout-address` and
`--expected-payout-script` must all be present or the runner exits before any
RPC call.

At startup the runner prints the network, LIVE or DRY RUN mode, Core's chain,
synchronization status, peer count, height and best block hash, then the
payout address, the script Core resolved, the expected script and whether they
match. In live mode, once every check has passed, it prints a final summary
immediately before requesting the first template:

```
NETWORK: MAINNET
MODE: LIVE
CHAIN CHECK: PASS
SYNC CHECK: PASS
PEER CHECK: PASS
PAYOUT ADDRESS: <address>
PAYOUT SCRIPT: <script>
EXPECTED PAYOUT SCRIPT: <script>
SCRIPT MATCH: PASS
WALLET OWNERSHIP: PASS
SUBMISSION TRANSPORT: bitcoin-cli -stdin
LIVE MAINNET PREFLIGHT: PASS
```

If any check fails the miner exits with an error instead, the summary is not
printed, and no template is requested and no CUDA process is started.

A candidate is submitted only if all of the following hold; anything else
stops the miner or discards the work without calling `submitblock`:

- the startup preflight passed (see below), the payout script validated and
  matched the pinned script, and the wallet confirmed ownership;
- the template passed validation and its previous block was Core's tip, which
  is re-checked before every CUDA chunk;
- the nonce returned by CUDA was re-hashed on the CPU and its full 256-bit
  hash is at or below the target;
- the serialized block passed the local consistency checks: its header is the
  CPU-verified header, the coinbase txid is the first Merkle leaf, the
  recomputed Merkle root equals the header's, the previous block hash matches
  the template, and the coinbase outputs are exactly the full coinbase value
  to the validated payout script plus the optional zero-value commitment;
- the network is mainnet with `--live-mainnet`, without `--dry-run`.

A verified mainnet candidate, in dry-run and live mode alike, is first written
next to the script as `unsubmitted_block_<hash>.hex` (temporary file, then
rename, so the name never holds a partial block). In live mode it is then
submitted at once. There is no tip check in between: Core itself decides
whether the block extends its chain, and asking first would only delay the
broadcast. The file is forced to disk beside the submission, not ahead of it,
and is kept afterwards.

`submitblock` results are handled as follows:

- no result (JSON `null`): accepted. A `MAINNET BLOCK ACCEPTED` banner prints
  the block hash, height, payout address and script, coinbase txid and value,
  the active-chain status is reported, mining stops (regardless of `--blocks`)
  and the block is monitored (see below).
- `duplicate`: Core already has the block. `getblockheader` decides: on the
  active chain it is treated as accepted, otherwise as a side-chain block.
- `inconclusive` or `duplicate-inconclusive`: Core stored the block but another
  block holds that height. This is reported as `NOT ON ACTIVE CHAIN`, not as a
  rejection, and a fresh template is requested. Core does not relay a block
  that competes with its own tip, and it has not fully validated it either, so
  this result neither earns the reward nor proves anything beyond the miner's
  own checks.
- `prev-blk-not-found`: the work was stale; a fresh template is requested.
- any other string (for example `high-hash` or `bad-txnmrklroot`): the exact
  reason is printed and the miner exits with an error, since it indicates a
  block-construction problem that retrying will not fix.
- RPC failure or timeout: says nothing about whether Core received the block,
  so Core is asked (`getblockheader`). If the block is on the active chain it
  is accepted. Otherwise the identical block is resubmitted on a fixed backoff
  schedule (about six minutes in total); resubmitting is harmless. Only when
  every attempt fails is `SUBMISSION FAILED` printed, with the saved file and
  the command to resubmit it, and the miner exits with an error.

A candidate found in a chunk that was abandoned because the tip had already
changed is still CPU-verified when its reply is read. On mainnet it is saved
and reported as `STALE CANDIDATE`, and it is not submitted, for the reason
above: Core would store it without relaying it.

Acceptance by the local node is not a guaranteed reward. The block must reach
other nodes and stay on the active chain for 100 further blocks.

#### Monitoring an accepted block

After an accepted mainnet block the miner frees the GPU and watches the block.
`--mainnet --monitor-block <hash>` does the same later, for example after a
restart; it is read-only and needs neither `--dry-run` nor `--live-mainnet`.
Once a minute it reads the block's confirmations and the wallet's category for
its coinbase (`immature`, `generate` or `orphan`) and prints a line whenever
either changes: `ACTIVE CHAIN` with the blocks left until maturity,
`NOT ON ACTIVE CHAIN` if the block was reorganized out (it keeps watching in
case it returns), and `COINBASE MATURE` at 101 confirmations, when it exits.
RPC failures are reported once and retried.

#### Staying connected while mining

A background thread checks every `--monitor-interval` seconds (default 15)
that the Tor SOCKS5 proxy answers and that Core is on mainnet, outside initial
block download, networking, and connected to at least `--min-peers` peers
(default 1; also enforced at startup). It only records the result and never
touches the CUDA process.

- One failed check, or one failed tip check, changes nothing: the GPU keeps
  its current work and the chunk in flight is still read, so a candidate in
  it is kept.
- Two consecutive failed health checks, or three consecutive failed tip
  checks, pause mining at the next chunk boundary, when nothing is in flight.
- While paused the miner retries after 5, 10, 20, 40 and then every 60
  seconds, for at most `--recovery-timeout` seconds (default 3600), and then
  exits with an error. It never starts or restarts Bitcoin Core or Tor.
- When the checks pass again it resumes with a freshly fetched and validated
  template.

A failed `getblocktemplate` call is handled the same way. Regtest keeps
failing fast.

While a mainnet session is mining or monitoring a block, the runner asks
Windows not to sleep on idle. GPU load does not count as user activity, so an
unattended machine would otherwise suspend mid-session. No power setting is
changed and the request ends with the process; the display may still turn off,
and closing the lid or choosing Sleep still suspends the machine.

The block is passed to `bitcoin-cli` through `-stdin`, because a real mainnet
block is far larger than the Windows command-line limit.

#### Preflight

Before requesting mining work, the tool verifies a SOCKS5 no-auth handshake at
`127.0.0.1:9150` by default, then checks that Bitcoin Core is on mainnet,
outside initial block download, synchronized to its header tip, and connected
to peers. It checks that the GBT previous block matches Core's current tip
before preparing CUDA work, validates the address, cross-checks `bits` against
the GBT target, and scans only the template's allowed nonce range. A SOCKS5
endpoint check verifies the proxy protocol handshake; it does not independently
identify the proxy software as Tor.

If the configured SOCKS5 endpoint is unavailable, startup fails closed unless
`--tor-executable` names a standalone `tor.exe`. In that case the miner starts
that executable with the configured SOCKS port and waits for a successful
SOCKS5 handshake. The miner tracks and stops only the Tor process it started;
it does not stop an already-available Tor process. Host, port, executable,
startup timeout, and poll interval can be set with `--tor-host`, `--tor-port`,
`--tor-executable`, `--tor-startup-timeout`, and `--tor-poll-interval`.
Use `--require-onion-peers` to additionally require a connected peer identified
by Core as an onion peer.

For example, with a standalone Tor binary:

```powershell
python regtest_miner.py --mainnet --dry-run --bitcoin-conf "C:\Program Files\Bitcoin\bitcoin.conf" --payout-address "<your-mainnet-address>" --tor-executable "C:\path\to\tor.exe" --tor-port 9150 --require-onion-peers
```

These checks apply identically to `--live-mainnet`. Any failed or unreachable
RPC call during preflight or mining stops the miner rather than continuing.

It is a solo miner, not a pool miner. At the current educational CUDA rate,
finding a mainnet block is extraordinarily unlikely; GPU time and electricity
costs can exceed any expected return.

Omit `--bitcoin-conf` when Bitcoin Core uses its default data directory
(`%APPDATA%\Bitcoin\bitcoin.conf`): `bitcoin-cli` finds that file by itself,
and the runner exits at startup if the named file does not exist.
`--bitcoin-conf` makes the runner pass that file to every `bitcoin-cli` RPC
call. Bitcoin Core itself must also be started with that configuration if it
is not already loading it, for example:

```powershell
& "C:\Program Files\Bitcoin\daemon\bitcoind.exe" -conf="C:\Program Files\Bitcoin\bitcoin.conf" -server
```

The file path alone does not choose a custom data directory. If the node uses
one, pass the same directory to the runner with `--datadir <path>` so
`bitcoin-cli` uses the node's RPC cookie and data. Bitcoin Core's Tor/proxy
settings apply to the node's peer-to-peer connections; the miner talks to
Core locally through `bitcoin-cli` and does not route its RPC calls over Tor.
Keep RPC bound to localhost and do not share the config file, since it may
contain RPC credentials.

Build from a Visual Studio 2022 Developer PowerShell so `nvcc` can find
`cl.exe`.

By default, the runner uses
`C:\Program Files\Bitcoin\daemon\bitcoin-cli.exe`, the `bitcoin-cli` default
network datadir, and mines one block. Use `--datadir <path>` if the node was
started with a custom datadir, `--bitcoin-conf <path>` to select a non-default
configuration file, `--blocks 0` to keep mining until interrupted, or
`--chunk-size <count>` to change the GPU nonce batch size (default:
250,000,000). Use
`--bitcoin-cli <path>` and `--cuda-miner <path>` if the executables are in
different locations. Mainnet dry-run always continues until interrupted, and
live mainnet stops after its first accepted block; `--blocks` does not affect
either mainnet mode.

Run the unit tests, which mock every Bitcoin Core RPC call, with:

```powershell
python -m unittest discover -v
```

The runner starts `cuda_miner.exe --serve` once, when the first template is
ready, and reuses that process, its CUDA context and its device buffers for
every nonce chunk and every later template. Each chunk is one request line on
the process's stdin (`SCAN <id> <80-byte-hex> <start> <count>`) and one reply
line on its stdout (`<id> NONE` or `<id> FOUND <nonce> <hash>`); diagnostics
go to stderr. New work replaces the old header without a restart. With
`--version-rolling` the request is `SCANV <id> <80-byte-hex> <start> <count>
<versions>` and a hit is reported as `<id> FOUND <nonce> <hash> <variant>`
(see "Version rolling" below).

The 32-bit nonce space of one header lasts only a few seconds. When it is
exhausted the validated template is kept and only the coinbase extranonce is
rolled, which changes the Merkle root and gives a fresh nonce space in well
under a millisecond; the rolled coinbase goes through the same payout check as
any other before it could be submitted. A template older than 30 seconds is
replaced at the next rollover. On mainnet its replacement is fetched and
validated on a background thread while the GPU keeps scanning, and handed over
as a finished, immutable work item; that thread never talks to the CUDA
process.

The chain-tip check runs once per chunk. The first chunk of a template is
checked before it is sent, unless the same tip was confirmed within the last
two seconds (after an extranonce rollover, or when a prefetched template
takes over), in which case it is treated like a later chunk. For every later chunk the runner sends the request
first and asks Core for its best block while the GPU is already scanning, so
the GPU does not idle during the RPC; the chunk's result is read only after
that check has returned. If the check reports a new tip, the chunk in flight
is abandoned and a new template is fetched; its reply is never used as a scan
result (see `STALE CANDIDATE` above). The cost is that after a new block the
GPU finishes at most one chunk of stale work before it can start on the new
template. On regtest a tip check that fails, or does not answer within 10
seconds, stops the miner; on mainnet it counts as one failed check. The process is ended by closing its stdin, on normal exit, on
errors and on Ctrl+C, and is killed if it does not exit within ten seconds.
Any malformed reply, mismatched request id, CUDA error or unexpected exit
stops the miner; the process is not restarted and nothing is submitted.
Rebuild `cuda_miner.exe` after updating: an executable built before `--serve`
existed is rejected at startup.

The scan kernel computes exact SHA256d and applies the full 256-bit target
comparison, with two shortcuts that do not change any result:

- The first hash resumes after rounds 0-3 of the header's second block. Those
  rounds do not depend on the nonce apart from one added term, so their state
  is computed once per header.
- The second hash stops after round 60, when the most significant 32 bits of
  the final hash are already determined. A nonce is rejected if that word is
  above the target's most significant word and accepted if it is below; when
  the two are equal the hash is completed and the full comparison decides.

The hash reported for a candidate always comes from the complete, unshortened
computation, and the runner re-hashes it on the CPU. Each shortcut can be
turned off at build time (`-DMINER_RESUME_FIRST_HASH=0`,
`-DMINER_EARLY_SECOND_HASH=0`), and `-DMINER_NONCES_PER_THREAD=<n>` changes
how many nonces each GPU thread hashes per launch (default 1) and
`-DMINER_THREADS_PER_BLOCK=<n>` the threads per block (default 256).
Interleaved runs of 128, 256 and 512 threads and of 1, 2 and 4 nonces per
thread all landed within 0.6% of each other, 128 threads being the slowest,
so the defaults were kept.

The kernel is at the instruction floor for this formulation of SHA256d: about
2,640 GPU instructions per hash (1,230 shifts and rotates, 665 three-input
logic operations, 681 additions), 39 registers, no spills and no stack frame.
The card executes them as fast as its clock allows, so the hashrate follows
the clock, and what remains is to do less work per hash, which is what
version rolling does.

#### Version rolling (optional)

`--version-rolling` makes the GPU hash 16 versions of every header for each
nonce. BIP 320 reserves bits 13-28 of the block version for miners; variant
`i` is the header with `i` added to that field, so the versions used are
`20000000`, `20002000`, ... `2001e000` for Core's usual template version.
The variants differ only in the first 64 header bytes, that is in the
midstate. The second SHA-256 block (Merkle-root tail, time, bits, nonce) is
the same for all of them, and so is its message schedule, which is therefore
expanded once per nonce instead of once per hash. That removes about 390 of
the 2,640 instructions for 15 of every 16 hashes.

| Versions per nonce | Sustained MH/s | Hashes per MHz of GPU clock |
| --- | --- | --- |
| 1 (default) | 701 | baseline |
| 2 | 758 | +8.8% |
| 4 | 795 | +13.2% |
| 8 | 808 | +15.7% |
| 16 (`--version-rolling`) | 813 | +17.0% |

Measured in one interleaved run at 86-87 C through `--serve`, six 20-second
slots each. The right-hand column divides out the clock the driver happened
to allow in each slot, which is what makes runs on this thermally limited
card comparable; it repeats to about 0.2%.

What changes and what does not:

- It is off unless the flag is given. Without it the runner sends the same
  `SCAN` requests as before, and the scan kernel they use compiles to
  byte-identical machine code.
- A block found this way carries the rolled version, which `CANDIDATE FOUND`
  prints. Nothing else in the header differs from the unrolled one: same
  previous block, Merkle root, time and bits. The coinbase, payout script,
  witness commitment and transaction set are untouched.
- The GPU names the variant it hit. The runner rebuilds that 80-byte header,
  hashes it on the CPU and requires the same hash, at or below the target,
  exactly as for an unrolled candidate; a wrong variant fails closed. Every
  later check, the saved-block file and the submission rules are unchanged.
- A chunk still holds `--chunk-size` hashes, so the chain tip is checked as
  often. One header now lasts about 85 seconds instead of 6, so the
  extranonce is rarely rolled and a template is replaced about every 85
  seconds instead of every 30 to 36; it is still dropped at once when the
  tip changes. An older template only means slightly older transactions.
- A template whose version already uses bits 13-28 is mined unrolled, with a
  message saying so. Bitcoin Core does not set them.
- Unlike the plain scan, the version scan can report nonce `ffffffff` itself,
  so that nonce needs no separate CPU check.

Rolled versions are valid by consensus (a version of 4 or more is all that
is required) and many mainnet blocks carry them. Bitcoin Core 27 confirmed
it for this miner's own blocks: built from a live mainnet template with each
rolled version and offered through `getblocktemplate` in `proposal` mode,
which validates without storing or submitting, every one was accepted, while
a block with version 1 was refused as `bad-version` and one with a wrong
Merkle root as `bad-txnmrklroot`. Core's template does not list the version
as mutable; that list is advice from a template server to its clients, not a
consensus rule.

To use it, add `--version-rolling` to the command line, or to the `python`
line of the launcher.

The host thread sleeps while a kernel runs (`cudaDeviceScheduleBlockingSync`)
instead of spinning a CPU core. On the thermally limited laptop GPU this was
measured at about 693 MH/s sustained against 643 MH/s with the default
spin-wait, because the idle core leaves the GPU more of the shared cooling.
The build flags `/Brepro` make the executable byte-for-byte reproducible from
the same source and toolchain (CUDA 13.4, MSVC 19.51).

The CUDA executable without arguments retains its synthetic-header benchmark,
and `--scan-header <80-byte-hex> --start <n> --count <n>` still performs a
single one-shot scan.

## Operating notes

### What to expect from a GPU

Solo mining Bitcoin on a graphics card is a lottery ticket, not an income.
At about 725 MH/s and a difficulty of about 1.3 x 10^14 (October 2026):

| Quantity | Value |
| --- | --- |
| Share of the network's hashrate | about 1 part in 1.3 trillion |
| Chance per day of continuous mining | about 1 in 9 billion |
| Chance per year of continuous mining | about 1 in 25 million |
| Average time to find one block | about 25 million years |

Each hash is independent, so the chance does not improve with time already
spent. Graphics cards found blocks routinely from mid-2010 until ASICs arrived
in 2013; the smallest solo winners reported since then used ASICs of about
1 TH/s and up, which is more than a thousand times this rate. The electricity
used costs far more than the expected reward. The project's value is a
correct, verified miner, not its earnings.

### Measured performance (RTX 3050 Laptop GPU)

| Measurement | Result |
| --- | --- |
| First seconds from idle | about 810-830 MH/s |
| Sustained, GPU at 86-87 C | about 700-750 MH/s |
| Sustained with `--version-rolling` | about 790-815 MH/s |
| Effective rate as a share of the raw rate | about 99.9% |
| GPU idle between work items | about 0.1% |
| Gap at an extranonce rollover | under 1 ms |
| CUDA reply to `submitblock` sent | 0.1-16 ms (regtest, up to a full-weight block) |

The card is limited by heat, not by the code: the driver holds 86-87 C by
lowering clock and power, and the hashrate follows. Better cooling is the
largest remaining gain. On a laptop, use a hard flat surface with the rear
raised, the highest fan profile and mains power, and stop if the temperature
passes 90 C. To watch it from a second window:

```powershell
nvidia-smi --query-gpu=temperature.gpu,power.draw,clocks.gr,utilization.gpu --format=csv -l 10
```

### Reading the output

```
Hashing nonce 00000000..0ee6b27f of 00000000..ffffffff | 250,000,000 hashes | ... H/s chunk | ... H/s average
```

- Every header has its own 32-bit nonce space, so each pass starts again at
  nonce `00000000`. Seeing the same range on consecutive lines does not mean
  the same input is being hashed: each pass uses a different extranonce and
  therefore a different Merkle root and header.
- `average` is the average of the current pass only, not of the session. The
  line is printed on the first chunk of a pass and then at most every five
  seconds, so at high hashrates only the first chunk of each pass is shown.
- `Mining mainnet block N` names the block being attempted, which is always
  Bitcoin Core's current tip plus one.
- `Nonce space exhausted; switching to the refreshed template.` appears about
  every 30 seconds with the same height: same block, updated transactions.
- `Template became stale; requesting new work.` followed by a height one
  higher means the network found a block and work moved on to the next one.
- A single `[MONITOR] Tip check failed (1/3)` is harmless. `Mining paused`
  means connectivity was lost and the runner is waiting to recover.

### Running it

- Start it in its own console window. A process started as a background job
  (for example with `start /b`) ignores Ctrl+C, so it could not be stopped
  cleanly.
- Stop it with one Ctrl+C and wait for `Stopped by user.` This closes the
  CUDA process and releases the idle-sleep request. Do not close the window
  while it is mining.
- Bitcoin Core must already be running and synchronized; the runner does not
  start it. With `onlynet=onion` through Tor Browser's proxy, closing Tor
  Browser drops every peer.
- A small launcher script is a convenient place for the payout address, the
  pinned script and `--min-peers`. Keep it out of version control if the
  address should not be tied to the repository: this project ignores
  `start_mining.bat` for that reason. It needs no private key, password or
  RPC credential.

### Wallet

- The live preflight asks the wallet whether it owns the payout address. That
  works with an encrypted wallet that is locked; the wallet never has to be
  unlocked for mining.
- Encrypting a descriptor wallet (Bitcoin Core 27) keeps existing addresses
  and their scripts, so a pinned payout script stays valid. It also creates a
  new seed for future addresses, so make a new backup afterwards; a backup
  taken before encryption is unencrypted and does not contain the new seed.
- Exactly one wallet may be loaded. With a second one loaded the ownership
  check cannot be answered and the runner refuses to start.

### What has and has not been verified

Verified: payout construction against live mainnet templates, the complete
submission path on regtest up to a full-weight block (including a lost reply,
failed calls, and Bitcoin Core being stopped at submission time), pause and
recovery with a real Tor interruption, block monitoring through a real
reorganization to maturity, and clean shutdown on Ctrl+C. For version
rolling: every variant against the CPU over randomized headers, targets and
ranges including the last nonce, acceptance of the rolled versions by Bitcoin Core in proposal
mode, and a 200-second mainnet dry-run (no repeated or skipped nonce range,
no `submitblock` call).

Not verified: an actual mainnet block submission, a submitted block with a
rolled version, unattended runs longer than
about fifteen minutes, and loss of Bitcoin Core's own peers (the tests
interrupted the Tor proxy, not the node's connections).
