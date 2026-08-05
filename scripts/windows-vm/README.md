# The Windows build box

`build-all-platforms.sh` runs its windows stage on a Windows 11 guest hosted on
the Linux build machine (`petes-alienware-pc`, a.k.a. pc1) under KVM. This
directory holds everything needed to rebuild that guest from nothing.

It replaces a GCP Windows VM that was deleted, which used to abort every release
run. Running it beside the Linux stage on hardware we already own removes the
cloud bill and the instance-lifecycle dance, at the cost of the guest only being
reachable from inside the house.

The host keeps running Linux as its main OS; Windows is just a guest, so the
machine remains available for training work. Autostart is disabled, and
`build-all-platforms.sh` starts the guest for the windows stage then shuts it
down on exit (via `WINDOWS_LIBVIRT_DOMAIN` in `.env`) so qemu does not keep
eight host cores busy between releases.

## Reaching it

The guest sits on libvirt's NAT network (192.168.122.0/24) and has no route from
the outside, so every connection hops through its host. That hop lives in
`~/.ssh/config` on the Mac, which is why `.env` names a host rather than an
address:

```
Host moonshine-win
  HostName 192.168.122.189
  User pete
  ProxyJump petes-alienware-pc
  ServerAliveInterval 15
  ServerAliveCountMax 8
```

`ssh moonshine-win` then works from the Mac, and `WINDOWS_CLOUD_HOST` in `.env`
is set to that alias. The address is pinned by a static DHCP reservation on the
libvirt network, so it will not drift.

NAT was chosen over bridging deliberately: building a bridge reconfigures the
host's physical NIC, and getting that wrong over SSH would cut off the only way
into a machine that lives in another room.

## Rebuilding it

The guest holds nothing that is not either in git or reinstallable, so the
recovery procedure for any serious problem is to recreate it:

```bash
# On the Linux host:
cd ~/windows-vm
SSH_PUBKEY="$(cat ~/.ssh/id_ed25519.pub)" \
VM_PASSWORD="$(cat .vm-password)" \
./create-vm.sh
```

That wipes any existing `moonshine-win` domain, builds an answer-file ISO from
the templates here, and boots Windows Setup against it. The install is hands-off
and takes 15-25 minutes. Then, from the Mac:

```bash
scp scripts/windows-vm/provision-toolchain.ps1 moonshine-win:C:/
ssh moonshine-win "powershell -NoProfile -ExecutionPolicy Bypass -File C:\provision-toolchain.ps1"
ssh moonshine-win "Restart-Computer -Force"   # so sshd picks up the new PATH
```

The reboot matters: sshd caches the environment it started with, so tools
installed after it came up are invisible to SSH sessions until it restarts.

## Watching an install

The guest is headless and silent until sshd is up, so when something goes wrong
during Setup the only way to see it is the virtual screen:

```bash
./screenshot.sh          # writes a PNG and prints its path
./send-keys.sh --enter "dir e:\\"           # type into the guest
./send-keys.sh --key KEY_F10 --modifier KEY_LEFTSHIFT   # cmd prompt in WinPE
```

Setup ignores a malformed answer file without reporting it and quietly reverts
to asking questions, which looks exactly like the file never being found.
`create-vm.sh` runs `xmllint` over the generated file to catch that, but if
Setup ever stalls on a dialog, the dialog names the setting that is missing.

## Files

| File | Role |
| --- | --- |
| `create-vm.sh` | Builds the answer-file ISO and defines the libvirt domain |
| `autounattend.xml.in` | Answers every Windows Setup and OOBE prompt |
| `bootstrap.ps1.in` | Runs at first logon; brings up sshd with the key authorized |
| `provision-toolchain.ps1` | Installs the build toolchain over SSH afterwards |
| `screenshot.sh` | Captures the guest's screen |
| `send-keys.sh` | Types into the guest when SSH is not available |

## What is installed

Visual Studio 2022 Build Tools with the v143 x64 toolset (which `test-core.bat`
and `publish-binary.bat` pin explicitly), CMake, Git, GitHub CLI, uv, Python,
and PowerShell 7. PowerShell is the SSH login shell because
`build-all-platforms.sh` sends its remote command in PowerShell syntax.

The repository is cloned to `C:\Users\pete\moonshine`. It is public over HTTPS,
so the guest needs no credentials to fetch it.

## Two things still to decide

**The Windows licence.** This is Windows 11 Enterprise **Evaluation**, installed
from Microsoft's Evaluation Center. It is genuine Microsoft media and needs no
key, but it is a 90-day licence that expires around 31 October 2026. Before then
it needs either `slmgr /rearm` (good for a few more 90-day periods) or a real
licence key. If you would rather run a properly licensed box from the start,
reinstall from retail media and activate it with a purchased key; nothing else
in this directory changes.

**Release upload credentials.** A plain CI run works today, but the release path
(`build-all-platforms.sh` in release mode) passes `-Upload`, and those steps
need `gh` authenticated for GitHub releases and a PyPI token for `twine`.
Neither is set up on the guest yet, so a release run will get through the builds
and fail at the first upload.

## Resources

8 vCPUs, 16 GB RAM, and a 200 GB sparse disk, out of the host's 24 cores and
60 GB. The topology is deliberately one socket with 8 cores under it: QEMU
otherwise presents each vCPU as a separate socket, and Windows client editions
are licensed for at most two, so the guest ignores the rest. Getting this wrong
is not an error, just a much slower build -- it cost about 35% on `test-core`.

A full CI run takes roughly 22 minutes.
