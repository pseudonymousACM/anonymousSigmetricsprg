import hashlib

def initiator_identifier(ip: str, port: int, group_size: int) -> bool:
    """
    Determines if a given IP and port can act as an initiator based on a group size.

    Args:
        ip (str): The IP address of the node.
        port (int): The port number of the node.
        group_size (int): The size of the group (e.g., 3 or 5).

    Returns:
        bool: True if the node is an initiator, False otherwise.
    """
    # Combine IP and port into a single string
    identifier = f"{ip}:{port}"

    # Generate a hash of the identifier
    hash_value = hashlib.sha256(identifier.encode()).hexdigest()

    # Convert the hash to an integer
    hash_int = int(hash_value, 16)

    # Check if the hash mod group_size is zero
    return hash_int % group_size == 0
