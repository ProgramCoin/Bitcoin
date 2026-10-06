
### Hardware Note: This project was developed and tested on an NVIDIA GeForce RTX 3050 Laptop GPU. While capable as a general-purpose GPU, the RTX 3050 provides relatively limited compute performance for Bitcoin's SHA-256 mining workload. As a result, the mining performance demonstrated by this project should not be interpreted as the maximum performance of the software on more powerful hardware


## Bitcoin Core CUDA solo miner

The runner supports Bitcoin Core `regtest` and a mainnet dry-run. Regtest
remains the default. Mainnet dry-run requires a valid mainnet payout address
and uses the live coinbase value and transaction set supplied by
`getblocktemplate`; it builds the transaction Merkle root and SegWit witness
commitment before scanning. Always verify the configured payout address
before starting.

### Regtest

Start a local regtest node in one PowerShell window:

```powershell
& "C:\Program Files\Bitcoin\daemon\bitcoind.exe" -regtest -server
```

From the project directory, build and run the miner:

```powershell
nvcc -O3 -arch=sm_86 cuda_miner.cu -o cuda_miner.exe
python regtest_miner.py
```

Regtest sends its coinbase output to an `OP_TRUE` anyone-can-spend script and
is for local testing only.

### Mainnet

Use a fully synchronized mainnet Bitcoin Core node with RPC enabled locally.
Never expose Bitcoin Core RPC to the public internet. Mainnet scanning is
currently dry-run only: it consumes live templates and the real network target,
but will never call `submitblock`. Live mainnet submission remains gated until
the required regtest and mainnet validation has actually succeeded. From the
project directory, build and pass a mainnet address that you control:

```powershell
nvcc -O3 -arch=sm_86 cuda_miner.cu -o cuda_miner.exe
python regtest_miner.py --mainnet --dry-run --bitcoin-conf "C:\Program Files\Bitcoin\bitcoin.conf" --payout-address "<your-mainnet-address>"
```

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

It is a solo miner, not a pool miner. At the current educational CUDA rate,
finding a mainnet block is extraordinarily unlikely; GPU time and electricity
costs can exceed any expected return.

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
different locations. Mainnet dry-run always continues until interrupted;
`--blocks` does not limit that mode.

The CUDA executable without arguments retains its synthetic-header benchmark.