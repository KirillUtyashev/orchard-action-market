import os
import sys
import stat
import time
import paramiko
from dotenv import load_dotenv

# ---------- helpers ----------

def _remote_exists(sftp, path):
    try:
        sftp.stat(path)
        return True
    except FileNotFoundError:
        return False

def _remote_isdir(sftp, path):
    try:
        return stat.S_ISDIR(sftp.stat(path).st_mode)
    except FileNotFoundError:
        return False

def _remote_mkdirs(sftp, path):
    parts = []
    head, tail = os.path.split(path)
    while tail:
        parts.append(tail)
        head, tail = os.path.split(head)
    # Build from root downward
    curr = "" if path.startswith("/") else "."
    for segment in reversed(parts):
        curr = os.path.join(curr, segment) if curr else segment
        if not _remote_exists(sftp, curr):
            sftp.mkdir(curr)

def _should_skip(name, exclude_names, exclude_exts):
    if name in exclude_names:
        return True
    _, ext = os.path.splitext(name)
    return ext in exclude_exts

# ---------- main upload ----------

def upload_folder(
        sftp,
        local_dir,
        remote_dir,
        overwrite=False,
        newer_only=True,
        exclude_names=("__pycache__", ".git", ".DS_Store"),
        exclude_exts=(".pyc",),
        preserve_times=True,
):
    """
    Recursively upload local_dir -> remote_dir.
    - overwrite=False: only upload if missing or different
    - newer_only=True: upload if local mtime > remote mtime (when sizes equal)
    """
    local_dir = os.path.abspath(local_dir)
    _remote_mkdirs(sftp, remote_dir)

    for root, dirs, files in os.walk(local_dir):
        # Skip excluded directories by mutating dirs in-place
        dirs[:] = [d for d in dirs if not _should_skip(d, exclude_names, set())]

        rel = os.path.relpath(root, local_dir)
        rdir = remote_dir if rel == "." else os.path.join(remote_dir, rel).replace("\\", "/")
        if not _remote_exists(sftp, rdir):
            _remote_mkdirs(sftp, rdir)

        # Upload files
        for fname in files:
            if _should_skip(fname, exclude_names, set(exclude_exts)):
                continue

            lpath = os.path.join(root, fname)
            rpath = os.path.join(rdir, fname).replace("\\", "/")

            lstat = os.stat(lpath)
            need_upload = False

            if not _remote_exists(sftp, rpath):
                need_upload = True
            elif not overwrite:
                # Compare size (fast) then mtime (optional)
                rstat = sftp.stat(rpath)
                if rstat.st_size != lstat.st_size:
                    need_upload = True
                elif newer_only and int(lstat.st_mtime) > int(rstat.st_mtime):
                    need_upload = True
            else:
                need_upload = True

            if need_upload:
                sftp.put(lpath, rpath)
                if preserve_times:
                    # Preserve mtime/atime on remote
                    sftp.utime(rpath, (int(lstat.st_atime), int(lstat.st_mtime)))
                print(f"Uploaded: {rpath}")
            else:
                print(f"Skipped:  {rpath}")

def upload_folder_to_vm_with_key(
        vm_ip,
        username,
        private_key_path,
        working_dir,
        local_folder,
        remote_folder,   # relative to working_dir or absolute if startswith("/")
        overwrite=False,
        newer_only=True,
):
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    key = paramiko.RSAKey.from_private_key_file(private_key_path)
    ssh.connect(hostname=vm_ip, username=username, pkey=key)

    # Keepalive
    ssh.get_transport().set_keepalive(5)

    sftp = ssh.open_sftp()

    # Resolve remote path
    if os.path.isabs(remote_folder):
        remote_dir = remote_folder
    else:
        remote_dir = os.path.join(working_dir, remote_folder).replace("\\", "/")

    upload_folder(
        sftp,
        local_dir=local_folder,
        remote_dir=remote_dir,
        overwrite=overwrite,
        newer_only=newer_only,
        exclude_names=("__pycache__", ".git", ".DS_Store"),
        exclude_exts=(".pyc",),
        preserve_times=True,
    )

    sftp.close()
    ssh.close()
    print("Upload completed.")

# ---------- CLI wrapper (env like your existing script) ----------

def main(args):
    """
    Usage:
      python upload_folder.py <vm_env_suffix> <local_folder> <remote_folder_rel_or_abs> [--overwrite] [--all]
    Examples:
      python upload_folder.py A ./policyitchk policyitchk
      python upload_folder.py B ./graphs /home/MARL/orchard-action-market/graphs --overwrite
      python upload_folder.py A ./src src --all
    Flags:
      --overwrite : always replace remote files
      --all       : newer_only = False (upload if size differs; ignore mtimes)
    """
    load_dotenv(dotenv_path=".env")
    if len(args) < 3:
        raise SystemExit("Need <vm_env_suffix> <local_folder> <remote_folder>")

    vm_suffix = args[0]
    local_folder = args[1]
    remote_folder = args[2]

    overwrite = "--overwrite" in args
    newer_only = not ("--all" in args)

    upload_folder_to_vm_with_key(
        vm_ip=os.getenv(f"VM_IP_{vm_suffix}"),
        username=os.getenv("USERNAME"),
        private_key_path=os.getenv("PRIVATE_KEY_PATH"),
        working_dir="/home/MARL/orchard-action-market",
        local_folder=local_folder,
        remote_folder=remote_folder,
        overwrite=overwrite,
        newer_only=newer_only,
    )


if __name__ == "__main__":
    main(sys.argv[1:])
