import paramiko
import sys
from dotenv import load_dotenv
import os


def run_experiment_and_fetch_logs_with_key(vm_ip, username, private_key_path, working_dir, env_script, args, remote_log_dir, local_log_dir):
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

    # Build remote experiment command
    # command = (
    #     f"cd {working_dir} && "
    #     f"git stash && git pull && "
    #     f"source {env_script} && "
    #     f"python -u run_experiments.py {' '.join(args)}"
    # )
    #
    # print(f"Executing remote command:\n{command}\n")
    #
    # # Run SSH command
    # stdin, stdout, stderr = ssh.exec_command(command)
    # print("Experiment started. Waiting for completion...")
    #
    # # Read and print stdout
    # print("----- STDOUT -----")
    # for line in stdout:
    #     print(line.strip())
    #
    # # Read and print stderr
    # print("----- STDERR -----")
    # error_output = stderr.read().decode().strip()
    # if error_output:
    #     print(error_output)
    # exit_status = stdout.channel.recv_exit_status()
    # if exit_status != 0:
    #     print(f"Error: Remote command failed with exit status {exit_status}")
    #     error_output = stderr.read().decode().strip()
    #     print("Error details:\n", error_output)
    #     # Optionally, raise an exception
    #     raise RuntimeError(f"Remote execution failed with exit status {exit_status}")
    # print(f"Experiment completed with exit status {exit_status}")
    #
    # for line in stdout:
    #     print(line.strip())

    # 2. Fetch logs via SFTP
    sftp = ssh.open_sftp()
    os.makedirs(local_log_dir, exist_ok=True)
    remote_files = sftp.listdir(working_dir + "/" + remote_log_dir)

    for filename in remote_files:
        if filename.endswith(".log"):
            remote_path = f"{working_dir + "/" + remote_log_dir}/{filename}"
            local_path = os.path.join(local_log_dir, filename)

            if os.path.exists(local_path):
                print(f"Skipped {filename} (already exists locally).")
                continue  # Skip downloading

            sftp.get(remote_path, local_path)
            print(f"Downloaded {filename}")

    sftp.close()
    ssh.close()

    print("Pipeline completed.")


def main(args):
    load_dotenv(dotenv_path=".env")

    local_arg = args[0]
    remaining_args = args[1:]

    print(f"Using local argument: {local_arg}")
    print(f"Passing to remote: {remaining_args}")

    if local_arg == "1":
        machine = "VM_IP"
        path = "PRIVATE_KEY_PATH"
        venv = ".venv"
    elif local_arg == "2":
        machine = "VM_IP_1"
        path = "PRIVATE_KEY_PATH_1"
        venv = "venv"
    else:
        machine = "VM_IP_2"
        path = "PRIVATE_KEY_PATH_1"
        venv = "venv"

    run_experiment_and_fetch_logs_with_key(
        vm_ip=os.getenv(machine),
        username=os.getenv("USERNAME"),
        private_key_path=os.getenv(path),
        working_dir='/home/MARL/orchard-action-market',
        env_script=f'{venv}/bin/activate',
        args=remaining_args,
        remote_log_dir="logs",
        local_log_dir="../train_scripts/logs"
    )


if __name__ == "__main__":
    main(sys.argv[1:])



