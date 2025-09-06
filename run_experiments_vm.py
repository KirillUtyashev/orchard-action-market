import paramiko
import sys
from dotenv import load_dotenv
import stat
import os


def download_logs(sftp, local_log_dir, remote_log_dir, working_dir):
    os.makedirs(local_log_dir, exist_ok=True)
    remote_files = sftp.listdir(working_dir + "/" + remote_log_dir)

    for filename in remote_files:
        if filename.endswith(".log") or filename.endswith(".txt"):
            remote_path = f"{working_dir}/{remote_log_dir}/{filename}"
            local_path = os.path.join(local_log_dir, filename)

            if os.path.exists(local_path):
                print(f"Skipped {filename} (already exists locally).")
                continue  # Skip downloading

            sftp.get(remote_path, local_path)
            print(f"Downloaded {filename}")


def download_most_recent_log(sftp, local_log_dir, remote_log_dir, working_dir):
    os.makedirs(local_log_dir, exist_ok=True)
    remote_path = os.path.join(working_dir, remote_log_dir)
    remote_files_attr = sftp.listdir_attr(remote_path)

    # Filter only .log or .txt files
    log_files = [
        f for f in remote_files_attr
        if f.filename.endswith(".log") or f.filename.endswith(".txt")
    ]

    if not log_files:
        print("No log or txt files found.")
        return

    # Find the most recently modified file
    latest_file = max(log_files, key=lambda f: f.st_mtime)

    remote_file_path = os.path.join(remote_path, latest_file.filename)
    local_file_path = os.path.join(local_log_dir, latest_file.filename)

    if os.path.exists(local_file_path):
        print(f"Skipped {latest_file.filename} (already exists locally).")
        return

    sftp.get(remote_file_path, local_file_path)
    print(f"Downloaded most recent file: {latest_file.filename}")


def download_all_folders(sftp, local_dir, remote_dir, working_dir):
    os.makedirs(local_dir, exist_ok=True)
    remote_path = os.path.join(working_dir, remote_dir)

    # List all entries with attributes
    entries = sftp.listdir_attr(remote_path)

    # Filter only directories
    folders = [e for e in entries if stat.S_ISDIR(e.st_mode)]

    if not folders:
        print("No folders found in remote directory.")
        return

    for folder in folders:
        folder_name = folder.filename
        local_folder_path = os.path.join(local_dir, folder_name)

        if os.path.exists(local_folder_path):
            print(f"Skipped {folder_name} (already exists locally).")
            continue

        remote_folder_path = os.path.join(remote_path, folder_name)
        os.makedirs(local_folder_path, exist_ok=True)

        # Download all .png files from the most recent folder
        files = sftp.listdir(remote_folder_path)
        for filename in files:
            remote_file_path = os.path.join(remote_folder_path, filename)
            local_file_path = os.path.join(local_folder_path, filename)

            if os.path.exists(local_file_path):
                print(f"Skipped {filename} (already exists locally).")
                continue

            sftp.get(remote_file_path, local_file_path)
            print(f"Downloaded {filename}")


def download_most_recent_folder(sftp, local_dir, remote_dir, working_dir):
    os.makedirs(local_dir, exist_ok=True)
    remote_path = os.path.join(working_dir, remote_dir)

    # List all entries with attributes
    entries = sftp.listdir_attr(remote_path)

    # Filter only directories
    folders = [e for e in entries if stat.S_ISDIR(e.st_mode)]

    if not folders:
        print("No folders found in remote directory.")
        return

    # Find the most recently modified folder
    latest_folder = max(folders, key=lambda e: e.st_mtime)
    latest_folder_name = latest_folder.filename

    print(f"Most recently updated folder: {latest_folder_name}")

    remote_folder_path = os.path.join(remote_path, latest_folder_name)
    local_folder_path = os.path.join(local_dir, latest_folder_name)
    os.makedirs(local_folder_path, exist_ok=True)

    # Download all .png files from the most recent folder
    files = sftp.listdir(remote_folder_path)
    for filename in files:
        remote_file_path = os.path.join(remote_folder_path, filename)
        local_file_path = os.path.join(local_folder_path, filename)

        if os.path.exists(local_file_path):
            print(f"Skipped {filename} (already exists locally).")
            continue

        sftp.get(remote_file_path, local_file_path)
        print(f"Downloaded {filename}")

def run_command(ssh, env_script, working_dir, args):
    # Build remote experiment command
    command = (
        f"cd {working_dir} && "
        f"git stash && git pull && "
        f"source {env_script} && "
        f"python -u run_experiments.py {' '.join(args)}"
    )

    print(f"Executing remote command:\n{command}\n")

    # Run SSH command
    stdin, stdout, stderr = ssh.exec_command(command)
    print("Experiment started. Waiting for completion...")

    # Read and print stdout
    print("----- STDOUT -----")
    for line in stdout:
        print(line.strip())

    # Read and print stderr
    print("----- STDERR -----")
    error_output = stderr.read().decode().strip()
    if error_output:
        print(error_output)
    exit_status = stdout.channel.recv_exit_status()
    if exit_status != 0:
        print(f"Error: Remote command failed with exit status {exit_status}")
        error_output = stderr.read().decode().strip()
        print("Error details:\n", error_output)
        # Optionally, raise an exception
        raise RuntimeError(f"Remote execution failed with exit status {exit_status}")
    print(f"Experiment completed with exit status {exit_status}")

    for line in stdout:
        print(line.strip())


def run_experiment_and_fetch_logs_with_key(vm_ip, username, private_key_path, working_dir, env_script, args, remote_log_dir, local_log_dir, remote_graph_dir, local_graph_dir,
                                           remote_weights_dir, local_weights_dir):
    print(username)
    print(private_key_path)
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    # Load private key
    key = paramiko.RSAKey.from_private_key_file(private_key_path)

    ssh.connect(
        hostname=vm_ip,
        username=username,
        pkey=key
    )

    # Set SSH keepalive (equivalent to -o ServerAliveInterval=30)
    transport = ssh.get_transport()
    transport.set_keepalive(5)

    # run_command(ssh, env_script, working_dir, args)

    # 2. Fetch logs via SFTP
    sftp = ssh.open_sftp()

    download_most_recent_log(sftp, local_log_dir, remote_log_dir, working_dir)
    # download_logs(sftp, local_log_dir, remote_log_dir, working_dir)
    download_most_recent_folder(sftp, local_graph_dir, remote_graph_dir, working_dir)
    download_most_recent_folder(sftp, local_weights_dir, remote_weights_dir, working_dir)
    # download_all_folders(sftp, local_weights_dir, remote_weights_dir, working_dir)

    sftp.close()
    ssh.close()

    print("Pipeline completed.")


def main(args):
    load_dotenv(dotenv_path=".env")

    local_arg = args[0]
    remaining_args = args[1:]

    print(f"Using local argument: {local_arg}")
    print(f"Passing to remote: {remaining_args}")

    run_experiment_and_fetch_logs_with_key(
        vm_ip=os.getenv(f"VM_IP_{local_arg}"),
        username=os.getenv("USERNAME"),
        private_key_path=os.getenv("PRIVATE_KEY_PATH"),
        working_dir='/home/MARL/orchard-action-market',
        env_script='venv/bin/activate',
        args=remaining_args,
        remote_log_dir="logs",
        local_log_dir="../logs",
        remote_graph_dir="graphs",
        local_graph_dir="../graphs",
        remote_weights_dir="policyitchk",
        local_weights_dir="../policyitchk"
    )


if __name__ == "__main__":
    main(sys.argv[1:])



