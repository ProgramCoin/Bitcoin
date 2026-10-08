
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
go to stderr. New work replaces the old header without a restart.

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
how many nonces each GPU thread hashes per launch (default 1, which measured
best).

The host thread sleeps while a kernel runs (`cudaDeviceScheduleBlockingSync`)
instead of spinning a CPU core. On the thermally limited laptop GPU this was
measured at about 693 MH/s sustained against 643 MH/s with the default
spin-wait, because the idle core leaves the GPU more of the shared cooling.
The build flags `/Brepro` make the executable byte-for-byte reproducible from
the same source and toolchain (CUDA 13.4, MSVC 19.51).

The CUDA executable without arguments retains its synthetic-header benchmark,
and `--scan-header <80-byte-hex> --start <n> --count <n>` still performs a
single one-shot scan.