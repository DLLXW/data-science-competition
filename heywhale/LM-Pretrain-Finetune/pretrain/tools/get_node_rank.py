import socket
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="PyTorch distributed training launch "
                                        "helper utility that will spawn up "
                                        "multiple distributed processes")

    # Optional arguments for the launch helper
    parser.add_argument("--num_nodes", type=int, default=1,
                        help="The number of nodes to use for distributed "
                             "training")
    parser.add_argument('--node_ips', type=str, default='',
                        help='nodes ip list for training, devided by ",", length >= num_nodes')
    args = parser.parse_args()
    return args

def get_worker_index(ip_list, num_nodes):
    if len(ip_list) != num_nodes:
        return 0
    local_ip = socket.gethostbyname(socket.gethostname())
    for i, ip in enumerate(ip_list):
        if local_ip == ip:
            return i
    return 0

def main():
    args = parse_args()
    ip_list = args.node_ips.split(",")
    node_rank = get_worker_index(ip_list, args.num_nodes)
    print(node_rank)

if __name__ == '__main__':
    main()
